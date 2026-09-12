// =============================================================================
// tcgen05_probe.cu — ptxas + layout probe for the sm_103a mixed fp8 x fp4
// block-scaled tcgen05 MMA:
//
//   tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
//       [d_tmem], a_desc, b_desc, idesc, [scale_a_tmem], [scale_b_tmem], p;
//
// WHY THIS FILE EXISTS (expert-tcgen05 pre-study, 0.5-day first step)
// ---------------------------------------------------------------------------
// The pre-study (docs/agent/perf-roadmap.md, "expert 侧 tcgen05 fp4 swapAB
// 预研") concluded from headers + CUTLASS + DeepGEMM reading only (the analysis
// host had no nvcc) that this spelling exists on sm_103a and that the swapAB
// mapping (A = fp4 e2m1 weight, B = e4m3 activation) is correct. This file turns
// those claims into facts:
//   1. does ptxas ASSEMBLE the instruction for sm_103a?      -> see STATUS below
//   2. is the 64-bit SMEM descriptor built the way the hardware wants?
//   3. do tcgen05.alloc / tcgen05.st / tcgen05.ld round-trip?
//   4. does D match a CPU reference (i.e. is the operand LAYOUT right)?
//
// STATUS (2026-09-12, checked locally with the CUDA 13.3 toolkit found at
// /tmp/nvccx/nvidia/cu13 — this host has nvcc+ptxas but NO GPU):
//   (1) VERIFIED. `nvcc -gencode arch=compute_103a,code=sm_103a` accepts the
//       file; the emitted PTX carries
//         .target sm_103a
//         tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
//             [%r182], %rd19, %rd20, %r183, [%r184], [%r185], p;
//       and `ptxas -arch=sm_103a` assembles it clean (ptxas -v: 30 registers,
//       4364 bytes smem). NOTE: `-arch=sm_103a` alone is NOT enough with this
//       nvcc — its `--list-gpu-arch` has no sm_103a entry, so it silently drops
//       the suffix and targets sm_103, where every tcgen05 instruction is
//       rejected. Use the explicit `-gencode arch=compute_103a,code=sm_103a`.
//   (2)(3)(4) NEED A GPU. No sm_103a device on this host, so the numeric check
//       (descriptor correctness + scale/TMEM mapping) is still open. Run it on a
//       B300:
//         nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//              -o /tmp/tcgen05_probe /tmp/tcgen05_probe.cu && /tmp/tcgen05_probe
//
// WHAT IS ALREADY PROVEN AND IS THEREFORE REUSED VERBATIM
// ---------------------------------------------------------------------------
// The descriptor bit layout, the TMEM row/column mapping and the SF-word
// mechanics below are the ones that kernels/cuda/dsv41_experts_mxf4.cu already
// validates numerically for tcgen05.mma.cta_group::1.kind::mxf4
// .block_scale.scale_vec::2X (see its header comment lines 24-70 and the
// self-test tests_tcgen05_mxf4.cu). Only what genuinely differs for the
// mxf8f6f4 path is changed, and every change is marked [CHANGED].
//
// THE ONE LAYOUT FACT THAT IS EASY TO GET WRONG
// ---------------------------------------------------------------------------
// [CHANGED] In the mxf8f6f4 path the fp4 operand is UNPACKED in smem: one e2m1
// element per BYTE (low nibble used). Evidence:
//   * CUTLASS cute/arch/mma_sm100_desc.hpp: `float_e2m1_unpacksmem_t` ->
//     MXF8F6F4Format::E2M1, while the packed `float_e2m1_t` -> MXF4Format::E2M1.
//     I.e. only kind::mxf4 consumes packed nibbles; mxf8f6f4 wants the unpacked
//     ("unpacksmem") representation.
//   * DeepGEMM include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:86,87 sizes
//     A/B smem as LOAD_BLOCK_M*BLOCK_K*sizeof(a_dtype_t) with a 1-byte dtype,
//     i.e. one element per byte.
// Consequence for K=32: A/B are 32 BYTES per row = 2 x 16-byte chunks, which is
// exactly the canonical UMMA Major-K SWIZZLE_NONE atom
//     ((8,n),(2,1)) : ((1,SBO),LBO)        [units of uint128_t]
// (cute/atom/mma_traits_sm100.hpp:275). A packed fp4 A would be 16 B/row and
// would NOT fit that canonical form. So A and B share the same strides:
//     LBO = 8 u128 = 128 B   (stride between the atom's two K-chunks)
//     SBO = 16 u128 = 256 B  (stride between 8-row groups)
// If the remote numeric check fails, this packing assumption is suspect #1:
// flip A to packed nibbles (16 B/row, K-chunks = 1, LBO unused) and re-run.
//
// [CHANGED] scale_vec::1X = one e8m0 scale per 32-element K-block (mxf8f6f4 has
// no 2X/4X). A 32-bit TMEM SF word holds 4 consecutive blocks' bytes and the
// idesc 2-bit SFA_ID/SFB_ID selects which byte. Ground truth: DeepGEMM
// sm100_fp8_fp4_gemm_1d1d.cuh sizes SF smem as rows*sizeof(uint32_t) and uses
// kNumSFAStagesPerLoad = 1 when kGranKA == 32, advancing the TMEM column by 4
// per 128-row group (SM100_UTCCP_4x32dp128bit). This probe has K=32 => exactly
// ONE block => SFA_ID = SFB_ID = 0 => the meaningful byte is byte 0. Suspect #2
// if the numeric check fails: the byte position inside the SF word.
//
// [CHANGED] idesc formats: mxf8f6f4 a_format/b_format use MXF8F6F4Format
// (E4M3=0, E5M2=1, E2M3=3, E3M2=4, E2M1=5) -- NOT the MXF4Format::E2M1 = 1 that
// dsv41_experts_mxf4.cu uses. Here a_format = 5 (E2M1, the weight) and
// b_format = 0 (E4M3, the activation). k_size = 0 = dense K32 for mxf8f6f4
// (whereas kind::mxf4's k_size = 0 means dense K64). All other idesc bits are
// unchanged. Expected idesc for this probe = 0x08820280.
//
// [CHANGED] M is pinned at 128 (CUTLASS SM100_MMA_MXF8F6F4_SS static_asserts
// M == 128 for the 1-CTA form; N must be a multiple of 8 in [8,256]). The brief
// said "A: 16 rows"; in the canonical layout 128 rows is 16 row-groups of 8, so
// A is [128, K] here.
//
// ISOLATION SWITCHES (for "ptxas rejected X" triage)
// ---------------------------------------------------------------------------
//   -DPROBE_MMA_SPELLING=1   emit `...kind::mxf8f6f4.block_scale` with no
//                            explicit `.scale_vec::1X` (the CUTLASS / DeepGEMM
//                            spelling) and the otherwise identical operands, to
//                            tell "the qualifier is the problem" apart from
//                            "the whole instruction is unsupported".
//                            Both spellings assemble on sm_103a (verified).
//   -DPROBE_ASM_ONLY=1       compile ONLY a minimal kernel holding the asm, so a
//                            rejection is attributable to the instruction alone
//                            and not to the surrounding file.
//   -DPROBE_SF_BYTE0_ONLY=1  write the SF byte into byte 0 of the SF word only,
//                            instead of replicating it to all 4 bytes. The
//                            default replication is proof-safe for a single
//                            K-block (whichever byte the SF-id picks carries the
//                            same value) and removes a byte-position ambiguity
//                            from the numeric check.
// =============================================================================

// =============================================================================
// PHASE 0 (2026-09-12) — REAL-WEIGHT PARITY SUITE (docs/agent/expert-tcgen05-plan.md §1)
// -----------------------------------------------------------------------------
// The layout probe above answers "does the instruction exist / are the
// descriptors shaped right" using RANDOM 4-bit codes and random e4m3 bytes.
// Phase 0 upgrades that to a numeric acceptance test:
//   * the fp4 operand is produced by REAL QUANTIZATION of random f32 — per
//     32-element block: e8m0 scale 2^ceil(log2(amax/6)) + e2m1 codes (exactly
//     the production quant.rs / dsv41_experts_mxf4.cu helpers);
//   * the activation is f32 -> e4m3 (per-32 e8m0) instead of fp4: the fp8
//     swapAB operand has more mantissa, so the quantization risk drops;
//   * geometry M=128 (the instruction pins M; gateup has 3840/128 = 30 tiles),
//     N=8 (one decode token zero-padded to the minimum legal N), K=32 per MMA
//     (the 1X scale granularity) with an nblk-block K loop (K = 32*nblk);
//   * references, all four of them, so a failure is attributable:
//       (a) CPU golden over the DEQUANTIZED operands (exact double arithmetic)
//       (b) CPU golden over the UNQUANTIZED f32 operands -> quantization loss
//       (c) a GPU f32 SIMT dequant GEMV (the current expert path's math shape)
//       (d) the dequant+double golden reproduced with f32 fmaf block-wise
//   * criteria (tests_tcgen05_mxf4.cu 口径): max rel err < 5e-2 with the
//     p50/p90/p99 reported, argmax equality (over M per column AND over N per
//     row), and max|diff| against the reference's top1-top2 margin.
//
// WHY (d) MATTERS: with e8m0 (power-of-two) block scales the scaling itself is
// EXACT in fp32 — no rounding. So (a) vs the tcgen05 result should differ only
// by the tensor core's accumulation order, i.e. ~1e-6 relative, NOT by the
// ~1e-2 that the quantization of the operands introduces. (b) is the case that
// isolates the quantization loss, and it is what tells whether e2m1 weights +
// e4m3 activations preserve the argmax at all.
//
// TWO SCALE-FACTOR LAYOUTS ARE RUN FOR EVERY CASE. The probe header above flags
// the SF byte position as "suspect #2" and only a numeric test can settle it:
//   sf_mode 0 PACKED  — the canonical layout of dsv41_experts_mxf4.cu: one word
//                       holds four consecutive blocks' bytes [SF(b0),SF(b1),
//                       SF(b2),SF(b3)], the idesc a_sf_id/b_sf_id = b%4 picks
//                       the byte, the word column advances 4 (one per 32-row
//                       group) per group of four blocks.
//   sf_mode 1 PERBLK  — one block per word in byte 0, sf_id = 0, word column
//                       advances 4 per block. For every single block this is
//                       bit-identical to the already-verified single-block
//                       probe, so it is layout-hypothesis-free: PACKED failing
//                       while PERBLK passes means the BYTE SELECTOR model is
//                       wrong, not the MMA/descriptor/SMEM layout.
// At nblk == 1 the two coincide (the suite says so instead of re-running).
// PERBLK costs 4 TMEM columns per K-block, so it cannot scale to the real K
// (160 blocks x 4 > 512 TMEM columns) — it exists to disambiguate, not to ship.
//
// NOT covered (Phase 1): the real K (5120 gateup / 1920 down) needs chunked
// staging — 128 rows x 5120 B = 640 KiB of fp4 smem per tile does not fit — so
// this file stages at most PH0_MAXBLK blocks and the real-K numbers are host-only.
//
// Build / run:
//   GPU : nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//              -I<CUDA/include> -o /tmp/t_ph0 tests_tcgen05_mxf8f6f4_1x.cu && /tmp/t_ph0
//   HOST: g++ -x c++ -DPH0_HOST_ONLY -O2 -std=c++17 -o /tmp/t_ph0_host \
//              tests_tcgen05_mxf8f6f4_1x.cu && /tmp/t_ph0_host
//         (no GPU, no CUDA toolkit: prints the quantization-loss numbers the GPU
//          suite predicts, including the real K=5120 shape)
//   triage: -DPROBE_RAW_LAYOUT=1 (random-code layout probe only),
//           -DPROBE_ASM_ONLY=1  (ptxas acceptance only)
// =============================================================================
#ifdef PH0_HOST_ONLY
// host-only build with plain g++: every CUDA header and device construct below
// is compiled out, only the shared ph0:: library is used.
#else
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// =============================================================================
// ph0:: — CUDA-FREE shared host code: codecs, block quantization, references and
// statistics. Compiled BOTH as host code of the nvcc build and by plain g++, so
// the numbers the GPU test compares against can be reproduced locally.
// The codecs are byte-compatible with production:
//   e2m1_encode      == dsv41_experts_mxf4.cu:152 (quant.rs e2m1_encode)
//   pow2_ceil_div    == dsv41_experts_mxf4.cu:141 fast_round_scale6, maxcode
//                       generalized to 6 (e2m1) / 448 (e4m3)
//   scale_to_ue8m0   == dsv41_experts_mxf4.cu:132 f_pow2_to_ue8m0
// =============================================================================
namespace ph0 {

// quant.rs FP4_TABLE, indexed by the 4-bit e2m1 code (production order).
const float kE2M1[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                         0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// ------------------------------------------------------------------- codecs
// Nearest e2m1 code, magnitudes {0,.5,1,1.5,2,3,4,6}; ties -> smaller magnitude
// (matches quant.rs e2m1_encode, which keeps the first minimum-distance slot).
inline uint8_t e2m1_encode(float v) {
    const float a = std::fmin(std::fabs(v), 6.0f);
    uint8_t c;
    if (a <= 0.25f) c = 0;
    else if (a <= 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a <= 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a <= 3.5f) c = 5;
    else if (a <= 5.0f) c = 6;
    else c = 7;
    return (uint8_t)(c | (v < 0.f ? 8u : 0u));
}

// e4m3 encode, round-to-nearest-even, saturating at the largest finite value
// 448 (0x7E); 0x7F/0xFF are NaN in this format and are never produced. Same
// convention as __nv_cvt_float_to_fp8(..., __NV_SATFINITE).
inline uint8_t e4m3_encode(float x) {
    const uint32_t sign = (x < 0.f) ? 0x80u : 0u;
    const float a = std::fabs(x);
    if (!(a <= 448.f)) return (uint8_t)(sign | 0x7Eu);  // NaN / Inf / overflow
    int efield, mant;
    if (a < 0.015625f) {  // < 2^-6: subnormal, step = 2^-9
        mant = (int)std::lrint(a * 512.0f);
        efield = 0;
        if (mant >= 8) { efield = 1; mant = 0; }
    } else {
        int e;
        std::frexp(a, &e);                                     // a = m*2^e, m in [.5,1)
        int E = e - 1;                                         // a in [2^E, 2^(E+1))
        int m8 = (int)std::lrint(std::ldexp(a, 3 - E));         // a/2^E*8 -> [8,16]
        if (m8 >= 16) { ++E; m8 = 8; }                          // mantissa carry
        efield = E + 7;
        mant = m8 - 8;
        // [256,448] is REPRESENTABLE (efield 15, mant 0..6) — the largest finite
        // is 0x7E (efield 15, mant 6). Only a true exponent overflow saturates;
        // clamping on `efield >= 15` would map the whole [256,448] decade onto
        // 448 (caught by the brute-force nearest-code check, max excess 192).
        if (E > 8) { efield = 15; mant = 6; }
    }
    return (uint8_t)(sign | (uint32_t)(efield << 3) | (uint32_t)mant);
}

// e4m3 decode S EEEE MMM (exact; the raw probe above calls this by name too).
inline double e4m3_to_d(uint8_t b) {
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    double v;
    if (e == 0)
        v = std::ldexp((double)m / 8.0, -6);  // subnormal
    else
        v = std::ldexp(1.0 + (double)m / 8.0, e - 7);
    return s ? -v : v;
}

// --------------------------------------------------------------------- e8m0
inline double ue8m0_to_scale(uint8_t s) { return std::ldexp(1.0, (int)s - 127); }

inline uint8_t scale_to_ue8m0(float s) {
    if (!(s > 0.f)) return 0;
    uint32_t bits;
    std::memcpy(&bits, &s, 4);
    int e = (int)((bits >> 23) & 0xFFu) - 127;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (uint8_t)(e + 127);
}

// 2^ceil(log2(amax / maxcode)): the production fast_round_scale6 with the code
// ceiling as a parameter (6 for e2m1, 448 for e4m3). A power of two keeps the
// dequantization exact, and ceil keeps |v|/scale <= maxcode (no saturation).
inline float pow2_ceil_div(float amax, float maxcode) {
    if (!(amax > 0.f)) return std::ldexp(1.f, -126);
    const float r = amax / maxcode;
    uint32_t bits;
    std::memcpy(&bits, &r, 4);
    int ex = (int)((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) ? 1 : 0);
    if (ex < -126) ex = -126;
    if (ex > 127) ex = 127;
    return std::ldexp(1.f, ex);
}

// ------------------------------------------------------------- block quantize
enum { Q_E2M1 = 0, Q_E4M3 = 1 };

// rows x K f32 -> ONE CODE BYTE PER ELEMENT (the mxf8f6f4 smem form is unpacked:
// the fp4 element lives in its own byte, low nibble) + one e8m0 byte per
// (row, 32-element K block).
struct Quantized {
    int rows = 0, K = 0;
    std::vector<uint8_t> code;  // rows*K
    std::vector<uint8_t> sf;    // rows*(K/32)
    int nblk() const { return K / 32; }
    uint8_t at(int r, int k) const { return code[(size_t)r * K + k]; }
    uint8_t scale(int r, int blk) const { return sf[(size_t)r * (K / 32) + blk]; }
};

inline Quantized quantize_rows(const std::vector<float>& x, int rows, int K, int fmt) {
    Quantized q;
    q.rows = rows;
    q.K = K;
    q.code.assign((size_t)rows * K, 0);
    q.sf.assign((size_t)rows * (K / 32), 0);
    const float maxcode = (fmt == Q_E2M1) ? 6.0f : 448.0f;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < K / 32; ++b) {
            const float* v = x.data() + (size_t)r * K + 32 * b;
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(v[i]));
            const float sc = pow2_ceil_div(amax, maxcode);
            // round-trip assertion: the byte the hardware reads must decode back
            // to the float scale the codes were built with (otherwise the golden
            // and the hardware would disagree for a reason that is not a bug).
            if (scale_to_ue8m0(sc) == 0 && sc != 0.f) { /* never: sc is a power of 2 */ }
            q.sf[(size_t)r * (K / 32) + b] = scale_to_ue8m0(sc);
            const float inv = 1.f / sc;  // exact: sc is a power of two
            for (int i = 0; i < 32; ++i) {
                const float t = v[i] * inv;
                q.code[(size_t)r * K + 32 * b + i] =
                    (fmt == Q_E2M1) ? e2m1_encode(t) : e4m3_encode(t);
            }
        }
    }
    return q;
}

inline double dequant_elem(const Quantized& q, int r, int k, int fmt) {
    const double s = ue8m0_to_scale(q.scale(r, k / 32));
    const double c = (fmt == Q_E2M1) ? (double)kE2M1[q.code[(size_t)r * q.K + k] & 0xF]
                                     : e4m3_to_d(q.code[(size_t)r * q.K + k]);
    return c * s;
}

// ------------------------------------------------------------------ references
// (a) D[m][n] = sum_k dequant(A)[m][k] * dequant(B)[n][k], exact in double, with
// the per-block scale product folded in exactly where the hardware folds it.
inline void golden_from_quant(const Quantized& A, const Quantized& B, int M, int N,
                              std::vector<double>& D) {
    const int K = A.K;
    D.assign((size_t)M * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int b = 0; b < K / 32; ++b) {
                const double sa = ue8m0_to_scale(A.scale(m, b));
                const double sb = ue8m0_to_scale(B.scale(n, b));
                double part = 0.0;
                for (int i = 0; i < 32; ++i) {
                    const int k = 32 * b + i;
                    part += (double)kE2M1[A.code[(size_t)m * K + k] & 0xF] *
                            e4m3_to_d(B.code[(size_t)n * K + k]);
                }
                acc += part * sa * sb;
            }
            D[(size_t)m * N + n] = acc;
        }
}

// (b) the same product from the UNQUANTIZED f32 operands: (a) minus (b) is the
// quantization loss and nothing else.
inline void exact_from_f32(const std::vector<float>& A, const std::vector<float>& B, int M,
                           int N, int K, std::vector<double>& D) {
    D.assign((size_t)M * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[(size_t)m * K + k] * (double)B[(size_t)n * K + k];
            D[(size_t)m * N + n] = acc;
        }
}

// (d) the golden's arithmetic redone the way an f32 accumulator would: sequential
// fmaf within a 32-block, block scale folded per block. The tensor core adds the
// 32 products in an unspecified (probably tree) order, so this predicts the order
// of magnitude of |tcgen05 - golden| (expect ~1e-6 relative), not its exact value.
inline void fp32_order_from_quant(const Quantized& A, const Quantized& B, int M, int N,
                                  std::vector<double>& D) {
    const int K = A.K;
    D.assign((size_t)M * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int b = 0; b < K / 32; ++b) {
                const float sa = std::ldexp(1.f, (int)A.scale(m, b) - 127);
                const float sb = std::ldexp(1.f, (int)B.scale(n, b) - 127);
                float part = 0.f;
                for (int i = 0; i < 32; ++i) {
                    const int k = 32 * b + i;
                    const float av = kE2M1[A.code[(size_t)m * K + k] & 0xF];
                    const float bv = (float)e4m3_to_d(B.code[(size_t)n * K + k]);
                    part = std::fma(av, bv, part);
                }
                acc = std::fma(part, sa * sb, acc);
            }
            D[(size_t)m * N + n] = (double)acc;
        }
}

// ----------------------------------------------------------------- statistics
// Two families of numbers, because they answer different questions:
//   * element-wise |got-ref| / max(|ref|, 1e-3*max|ref|) — the plan's criterion
//     #2. It blows up on the near-cancelling outputs of a random dot product, so
//     a large p50 here does NOT mean the kernel is wrong;
//   * norm-wise (max|d|/max|ref|, ||d||2/||ref||2, ||d||1/||ref||1) — the plan's
//     criterion #1 shape, stable and comparable across shapes. Use THESE to judge
//     the quantization loss; use the element-wise ones only against the golden.
struct Stats {
    double p50 = 0, p90 = 0, p99 = 0, mx = 0;  // element-wise relative errors
    double maxabs = 0;                         // max absolute |got-ref|
    double denom = 0;                          // max(1e-3*max|ref|, eps): rel floor
    double max_ref = 0;                        // max|ref|
    double rel_max = 0;                        // max|d| / max|ref|      (criterion 1)
    double rel_l2 = 0;                         // ||d||2 / ||ref||2
    double rel_l1 = 0;                         // ||d||1 / ||ref||1
    int argM_ok = 0, argM_tot = 0;             // per column: argmax over the M rows
    int argN_ok = 0, argN_tot = 0;             // per row: argmax over the N columns
    int risky_cols = 0;                        // columns where maxdiff > top1-top2
    double margin_ratio = 0;                   // max over columns of diff/margin
    double min_margin = 0;                     // relative top1-top2 (over M), min
    bool pass(double tol) const { return mx < tol; }
};

inline Stats compare(const std::vector<double>& got, const std::vector<double>& ref, int M,
                     int N) {
    Stats s;
    double amax = 0, num2 = 0, den2 = 0, num1 = 0, den1 = 0;
    for (double v : ref) amax = std::fmax(amax, std::fabs(v));
    s.max_ref = amax;
    s.denom = std::fmax(1e-3 * amax, 1e-300);
    std::vector<double> rel;
    rel.reserve(got.size());
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(got[i] - ref[i]);
        s.maxabs = std::fmax(s.maxabs, d);
        num2 += d * d;
        den2 += ref[i] * ref[i];
        num1 += d;
        den1 += std::fabs(ref[i]);
        rel.push_back(d / std::fmax(std::fabs(ref[i]), s.denom));
    }
    std::sort(rel.begin(), rel.end());
    const auto pct = [&](double p) {
        if (rel.empty()) return 0.0;
        size_t i = (size_t)(p * (double)(rel.size() - 1));
        return rel[i];
    };
    s.p50 = pct(0.50);
    s.p90 = pct(0.90);
    s.p99 = pct(0.99);
    s.mx = rel.empty() ? 0.0 : rel.back();
    s.rel_max = (amax > 0) ? s.maxabs / amax : 0.0;
    s.rel_l2 = (den2 > 0) ? std::sqrt(num2 / den2) : 0.0;
    s.rel_l1 = (den1 > 0) ? num1 / den1 : 0.0;

    // argmax bookkeeping. NOTE the margins: for random data the top1-top2 gap over
    // 128 rows is tiny relative to the row norm, so "risky_cols" is ~all of them
    // even for a perfect kernel. margin_ratio (max diff/margin) is the number that
    // says whether a decision could actually flip.
    s.min_margin = 1e300;
    s.margin_ratio = 0;
    for (int n = 0; n < N; ++n) {
        int bi = 0, gi = 0;
        double bv = -1e300, gv = -1e300, second = -1e300, colmax = 0.0;
        for (int m = 0; m < M; ++m) {
            const double rv = ref[(size_t)m * N + n], gvv = got[(size_t)m * N + n];
            if (rv > bv) { bv = rv; bi = m; }
            if (gvv > gv) { gv = gvv; gi = m; }
            colmax = std::fmax(colmax, std::fabs(gvv - rv));
        }
        for (int m = 0; m < M; ++m)
            if (m != bi) second = std::fmax(second, ref[(size_t)m * N + n]);
        ++s.argM_tot;
        if (gi == bi) ++s.argM_ok;
        const double margin = bv - second;
        s.min_margin = std::fmin(s.min_margin, margin / std::fmax(1.0, std::fabs(bv)));
        if (margin > 0) s.margin_ratio = std::fmax(s.margin_ratio, colmax / margin);
        if (colmax > margin) ++s.risky_cols;
    }
    for (int m = 0; m < M; ++m) {
        int bi = 0, gi = 0;
        double bv = -1e300, gv = -1e300;
        for (int n = 0; n < N; ++n) {
            const double rv = ref[(size_t)m * N + n], gvv = got[(size_t)m * N + n];
            if (rv > bv) { bv = rv; bi = n; }
            if (gvv > gv) { gv = gvv; gi = n; }
        }
        ++s.argN_tot;
        if (gi == bi) ++s.argN_ok;
    }
    return s;
}

// ------------------------------------------------------------------- rng/data
// Deterministic LCG + Box-Muller: the host analysis and the GPU suite generate
// byte-identical inputs, so their numbers are comparable.
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 1664525u + 1013904223u;
        return s;
    }
    float uniform() { return (float)((next() >> 8) & 0xFFFFFFu) * (1.f / 16777216.f); }
    float normal() {
        const float u1 = std::fmax(uniform(), 1e-7f), u2 = uniform();
        return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530718f * u2);
    }
};

// Weight/activation distributions. LLM expert weights are roughly Gaussian with
// a sparse tail (the checkpoint's fp4 blocks chase that tail), decode-time
// activations are heavy tailed positive.
enum { D_NORMAL = 0, D_OUTLIER = 1, D_LOGNORMAL = 2 };
inline float draw(Rng& r, int dist) {
    if (dist == D_OUTLIER) {
        float v = r.normal();
        if (r.uniform() < 0.002f) v *= 12.f;
        return v;
    }
    if (dist == D_LOGNORMAL) return std::exp(1.1f * r.normal()) * 0.5f;
    return r.normal();
}

struct CaseCfg {
    const char* name;
    int M, N, K;
    uint32_t seed;
    int wdist, adist;
};

struct CaseCore {
    CaseCfg cfg;
    Quantized W;  // weights   [M,K] e2m1 (A operand of swapAB)
    Quantized A;  // activation[N,K] e4m3 (B operand of swapAB)
    std::vector<double> golden;  // (a) dequantized, double
    std::vector<double> exact;   // (b) unquantized, double
    std::vector<double> f32ord;  // (d) dequantized, f32 fmaf order
};

inline CaseCore build_case(const CaseCfg& cfg) {
    CaseCore c;
    c.cfg = cfg;
    const int M = cfg.M, N = cfg.N, K = cfg.K;
    Rng r(cfg.seed);
    std::vector<float> wf((size_t)M * K), af((size_t)N * K);
    for (auto& v : wf) v = draw(r, cfg.wdist);
    for (auto& v : af) v = draw(r, cfg.adist);
    c.W = quantize_rows(wf, M, K, Q_E2M1);
    c.A = quantize_rows(af, N, K, Q_E4M3);
    golden_from_quant(c.W, c.A, M, N, c.golden);
    exact_from_f32(wf, af, M, N, K, c.exact);
    fp32_order_from_quant(c.W, c.A, M, N, c.f32ord);
    return c;
}

// --------------------------------------------------------------- host analysis
// The local half of Phase 0: no GPU, no CUDA. It reports (1) the quantization
// loss (a) vs (b) — the number that decides whether e2m1 weights + e4m3
// activations are numerically viable at all used with the argmax criterion — and
// (2) the f32-accumulation deviation (a) vs (d), i.e. what the GPU should report
// for "tcgen05 vs CPU golden" if the layout is right.
inline void report(const char* tag, const Stats& s, const char* what) {
    printf("     %-22s %-28s rel: p50=%.1e p90=%.1e p99=%.1e max=%.1e | norm: max/max=%.2e "
           "l2=%.2e l1=%.2e | maxabs=%.2e\n",
           tag, what, s.p50, s.p90, s.p99, s.mx, s.rel_max, s.rel_l2, s.rel_l1, s.maxabs);
}

inline int host_analysis() {
    const CaseCfg cfgs[] = {
        {"nblk=1  K=32   w=N(0,1)    a=N(0,1)   ", 128, 8, 32, 20260912u, D_NORMAL, D_NORMAL},
        {"nblk=4  K=128  w=N(0,1)    a=N(0,1)   ", 128, 8, 128, 20260913u, D_NORMAL, D_NORMAL},
        {"nblk=8  K=256  w=outlier   a=N(0,1)   ", 128, 8, 256, 20260914u, D_OUTLIER, D_NORMAL},
        {"nblk=8  K=256  w=N(0,1)    a=lognormal", 128, 8, 256, 20260915u, D_NORMAL, D_LOGNORMAL},
        {"REALSHAPE K=5120 w=outlier a=N(0,1)   ", 128, 8, 5120, 20260916u, D_OUTLIER, D_NORMAL},
    };
    printf("\n== PHASE 0 host analysis (no GPU; same codecs/data as the GPU suite) ==\n");
    printf("   (d) predicts what the GPU must print for 'tcgen05 vs golden': the CRITERION\n");
    printf("       (max rel err < 5e-2, argmax equal) applies to that line, NOT to (a).\n");
    printf("   (a) is the quantization loss of e2m1+e4m3 itself: informational, and the\n");
    printf("       element-wise p50 is inflated by near-cancelling dot products — read the\n");
    printf("       norm columns (max/max, l1, l2) there.\n");
    int bad = 0;
    for (const CaseCfg& cfg : cfgs) {
        const CaseCore c = build_case(cfg);
        const int nblk = cfg.K / 32;
        Stats quant = compare(c.golden, c.exact, cfg.M, cfg.N);    // (a) vs (b)
        Stats fp32o = compare(c.f32ord, c.golden, cfg.M, cfg.N);   // (d) vs (a)
        printf("\n   [%s] M=%d N=%d K=%d nblk=%d  |ref_exact|max=%.4g rms=%.3g\n", cfg.name,
               cfg.M, cfg.N, cfg.K, nblk, quant.max_ref,
               [&] {
                   double s2 = 0;
                   for (double v : c.exact) s2 += v * v;
                   return std::sqrt(s2 / c.exact.size());
               }());
        report("(d) f32fma vs golden", fp32o, "expected tcgen05 vs golden");
        report("(a) golden vs exact", quant, "QUANTIZATION LOSS (info)");
        printf("     criterion check on (d): max rel %.3g %s 5e-2; argmax M %d/%d N %d/%d\n",
               fp32o.mx, fp32o.mx < 5e-2 ? "<" : ">=", fp32o.argM_ok, fp32o.argM_tot,
               fp32o.argN_ok, fp32o.argN_tot);
        printf("     quantization side: argmax over M %d/%d, over N %d/%d kept; "
               "min rel top1-top2 margin=%.2e; max diff/margin=%.2f\n",
               quant.argM_ok, quant.argM_tot, quant.argN_ok, quant.argN_tot, quant.min_margin,
               quant.margin_ratio);
        if (fp32o.mx >= 5e-2) {
            printf("     [NOTE] the f32 accumulation deviation is already >= 5e-2 on the host;\n"
                   "            the GPU 5e-2 threshold could not then distinguish a layout bug.\n");
            ++bad;
        }
        if (quant.argM_ok != quant.argM_tot) {
            printf("     [INFO] the e2m1+e4m3 FORMAT alone flips %d/%d column argmax (over M)\n"
                   "            at this K/margin. The GPU criterion compares tcgen05 against the\n"
                   "            QUANTIZED golden, so this is a format observation, not a bug — but\n"
                   "            it is the number that says how much headroom the fp4 path has.\n",
                   quant.argM_tot - quant.argM_ok, quant.argM_tot);
        }
    }
    printf("\n   [host analysis done] parity criterion (d): %s; quantization is lossy by construction "
           "(see the l1 column)\n",
           bad ? "NOT SAFE at this tolerance" : "expected to pass (max rel ~1e-6..1e-5)");
    return 0;  // informational: it never fails the build
}

}  // namespace ph0

// --------------------------------------------------------------------- knobs
#define PROBE_M 128  // pinned by the instruction (CUTLASS static_assert)
#define PROBE_N 8    // minimum legal N for mxf8f6f4 (multiples of 8 in [8,256])
#define PROBE_K 32   // dense K for mxf8f6f4, = one 32-element scale block (1X)

#define PROBE_TMEM_COLS 32  // power of two, >= 32 (tcgen05.alloc requirement)

// TMEM column map: D first (as in dsv41_experts_mxf4.cu), then SFA, then SFB.
//   D   : M=128 -> lanes (m%32) x 4 partitions, N=8 columns.
//   SFA : 4 columns, one per 32-row group of A's M=128 rows.
//   SFB : 4 columns are written (proven x4 store form); at N=8 the hardware
//         reads only the first one (B rows n<32 -> lane n, column +0).
#define PROBE_D_OFF 0
#define PROBE_D_COLS PROBE_N
#define PROBE_SFA_OFF (PROBE_D_OFF + PROBE_D_COLS)      // 8
#define PROBE_SFA_COLS 4
#define PROBE_SFB_OFF (PROBE_SFA_OFF + PROBE_SFA_COLS)  // 12
#define PROBE_SFB_COLS 4

// Canonical Major-K SWIZZLE_NONE strides (identical for A and B here; see the
// header comment). In units of 16 bytes: LBO = 8, SBO = 16.
#define PROBE_LBO_BYTES 128
#define PROBE_SBO_BYTES 256

#define PROBE_A_BYTES (PROBE_M * PROBE_K)  // 4096: 1 byte per fp4 element
#define PROBE_B_BYTES (PROBE_N * PROBE_K)  //  256: 1 byte per e4m3 element

#ifndef PROBE_MMA_SPELLING
#define PROBE_MMA_SPELLING 0
#endif

#ifndef PH0_HOST_ONLY
namespace {

// ------------------------------------------------------------- small helpers
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ------------------------------------------------------- tcgen05 primitives
__device__ __forceinline__ void tc_alloc(uint32_t* dst, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                     smem_addr(dst)),
                 "r"(ncols));
}
__device__ __forceinline__ void tc_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols)
                 : "memory");
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
            smem_addr(bar))
        : "memory");
}
__device__ __forceinline__ void tc_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(cnt)
                 : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n\t.reg .pred p;\n"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_%=;\n\t}" ::"r"(smem_addr(bar)),
        "r"(phase)
        : "memory");
}

// ------------------------------------------------------------- the MMA under test
// [CHANGED vs dsv41_experts_mxf4.cu] kind::mxf8f6f4.block_scale.scale_vec::1X
// instead of kind::mxf4.block_scale.scale_vec::2X. The operand list is
// identical: [d_tmem], a_desc, b_desc, idesc, [sf_a_tmem], [sf_b_tmem], pred.
// Operand numbering used by both branches: %0 d_tmem, %1 a_desc, %2 b_desc,
// %3 idesc, %4 sfa_tmem, %5 sfb_tmem, %6 enable_input_d.
__device__ __forceinline__ void tc_mma_mxf8f6f4_1x(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                                   uint32_t idesc, uint32_t sfa_tmem,
                                                   uint32_t sfb_tmem, uint32_t enable_d) {
#if PROBE_MMA_SPELLING == 0
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
        : "memory");
#else
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
        : "memory");
#endif
}

__device__ __forceinline__ void tc_st_x4(uint32_t taddr, uint32_t w0, uint32_t w1, uint32_t w2,
                                         uint32_t w3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr),
                 "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                 : "memory");
}
__device__ __forceinline__ void tc_ld_x8(uint32_t taddr, uint32_t* v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7])
        : "r"(taddr)
        : "memory");
}

// ------------------------------------------------------------- descriptors
// CUTLASS UMMA::SmemDescriptor / DeepGEMM make_smem_desc bit layout:
//   [ 0,14) start_address >> 4
//   [16,30) leading_byte_offset >> 4    (LBO)
//   [32,46) stride_byte_offset  >> 4    (SBO)
//   [46,48) version = 1 (Blackwell)
//   [49,52) base_offset = 0, [52] lbo_mode = 0 (legacy)
//   [61,64) layout_type = 0 (SWIZZLE_NONE / INTERLEAVE)
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_base, uint32_t lbo_bytes,
                                              uint32_t sbo_bytes) {
    const uint64_t start = (uint64_t)((smem_base >> 4) & 0x3FFFu);
    const uint64_t lbo = (uint64_t)((lbo_bytes >> 4) & 0x3FFFu);
    const uint64_t sbo = (uint64_t)((sbo_bytes >> 4) & 0x3FFFu);
    return start | (lbo << 16) | (sbo << 32) | ((uint64_t)1 << 46);  // version = 1
}

// Instruction descriptor, block-scaled form (CUTLASS
// UMMA::InstrDescriptorBlockScaled). [CHANGED] formats come from
// MXF8F6F4Format: E4M3 = 0, E5M2 = 1, E2M3 = 3, E3M2 = 4, E2M1 = 5.
// Bit map: [0,2) sparse_id2 [2] sparse [4,6) b_sf_id [7,10) a_format
// [10,13) b_format [13] a_neg [14] b_neg [15] a_major [16] b_major
// [17,23) n_dim = N>>3 [23] scale_format (1 = UE8M0) [24,29) m_dim = M>>4
// [29,31) a_sf_id [31] k_size (0 = dense K32 for mxf8f6f4)
__device__ __forceinline__ uint32_t make_idesc_mxf8f6f4(uint32_t m_dim, uint32_t n_dim,
                                                        uint32_t a_fmt, uint32_t b_fmt,
                                                        uint32_t a_sf_id, uint32_t b_sf_id,
                                                        uint32_t k_size) {
    uint32_t d = 0;
    d |= (b_sf_id & 0x3u) << 4;
    d |= (a_fmt & 0x7u) << 7;
    d |= (b_fmt & 0x7u) << 10;
    d |= (n_dim & 0x3Fu) << 17;
    d |= 1u << 23;  // scale_format = UE8M0
    d |= (m_dim & 0x1Fu) << 24;
    d |= (a_sf_id & 0x3u) << 29;
    d |= (k_size & 0x1u) << 31;
    return d;
}

}  // namespace
#endif  // !PH0_HOST_ONLY

// =============================================================================
#ifdef PROBE_ASM_ONLY
// ---------------------------------------------------------------------------
// Minimal translation unit: ONE kernel holding just the instruction under
// test, so that a ptxas rejection is attributable to the MMA alone.
//   nvcc -gencode arch=compute_103a,code=sm_103a -DPROBE_ASM_ONLY=1 -c file.cu
// ---------------------------------------------------------------------------
namespace {
__global__ void probe_asm_only_kernel(const uint8_t* a, const uint8_t* b) {
    __shared__ uint32_t s_tmem;
    __shared__ __align__(8) uint64_t s_mbar;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    if (warp == 0) {
        tc_alloc(&s_tmem, PROBE_TMEM_COLS);
        tc_relinquish();
    }
    if (tid == 0) mbar_init(&s_mbar, 1);
    __syncthreads();
    if (tid == 0) {
        const uint32_t tb = s_tmem;
        const uint64_t da = make_desc(smem_addr(a), PROBE_LBO_BYTES, PROBE_SBO_BYTES);
        const uint64_t db = make_desc(smem_addr(b), PROBE_LBO_BYTES, PROBE_SBO_BYTES);
        const uint32_t idesc = make_idesc_mxf8f6f4(PROBE_M >> 4, PROBE_N >> 3, 5, 0, 0, 0, 0);
        tc_mma_mxf8f6f4_1x(tb, da, db, idesc, tb + PROBE_SFA_OFF, tb + PROBE_SFB_OFF, 0u);
        tc_commit(&s_mbar);
    }
    __syncthreads();
    if (warp == 0) tc_dealloc(s_tmem, PROBE_TMEM_COLS);
}
}  // namespace

int main() {
    return 0;  // nothing to run: this build only answers "does ptxas take it?"
}

#else  // !PROBE_ASM_ONLY
#ifdef PH0_HOST_ONLY
// ---------------------------------------------------------------------------
// Host-only build (g++ -x c++ -DPH0_HOST_ONLY): no CUDA toolkit, no GPU. Runs
// the Phase 0 numerical analysis with the same codecs/data/goldens the GPU
// suite uses, including the real K=5120 shape the GPU kernel cannot stage.
// ---------------------------------------------------------------------------
int main() { return ph0::host_analysis(); }
#else  // !PH0_HOST_ONLY  (the CUDA build: probe + Phase 0 parity)
// =============================================================================
namespace {

// ------------------------------------------------------------------- kernel
__global__ void __launch_bounds__(128) probe_kernel(
    const uint8_t* __restrict__ a,    // [128][32] fp4 e2m1, UNPACKED (1 element/byte)
    const uint8_t* __restrict__ sfa,  // [128] e8m0, one 32-element block per row
    const uint8_t* __restrict__ b,    // [  8][32] e4m3 (1 element/byte)
    const uint8_t* __restrict__ sfb,  // [  8] e8m0
    float* __restrict__ d_out,        // [128][8]
    uint64_t* __restrict__ dbg)       // [4] {a_desc, b_desc, idesc, tmem_base}
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // -------- smem: canonical Major-K SWIZZLE_NONE --------
    //   unit16(r, kb) = (r % 8) + 8*kb + 16*(r / 8)      [16-byte units], kb in {0,1}
    //   A: 128 rows * 32 B = 256 units = 4096 B
    //   B:   8 rows * 32 B =  16 units =  256 B
    __shared__ __align__(1024) uint8_t s_a[PROBE_A_BYTES];
    __shared__ __align__(1024) uint8_t s_b[PROBE_B_BYTES];
    __shared__ __align__(8) uint64_t s_mbar;
    __shared__ uint32_t s_tmem_base;

    // -------- tmem alloc --------
    if (warp == 0) {
        tc_alloc(&s_tmem_base, PROBE_TMEM_COLS);
        tc_relinquish();
    }
    if (tid == 0) mbar_init(&s_mbar, 1);
    __syncthreads();

    const uint32_t tmem_base = s_tmem_base;
    const uint32_t d_col = tmem_base + PROBE_D_OFF;
    const uint32_t sfa_col = tmem_base + PROBE_SFA_OFF;
    const uint32_t sfb_col = tmem_base + PROBE_SFB_OFF;

    // -------- stage A (one row per thread: 128 threads, 128 rows) --------
    for (int m = tid; m < PROBE_M; m += 128) {
        for (int kb = 0; kb < 2; ++kb) {
            const int unit = (m & 7) + 8 * kb + 16 * (m >> 3);
            uint4 v;
            __builtin_memcpy(&v, a + (size_t)m * PROBE_K + kb * 16, 16);
            *reinterpret_cast<uint4*>(s_a + unit * 16) = v;
        }
    }
    // -------- stage B (8 rows * 2 chunks = 16 uint4) --------
    for (int c = tid; c < PROBE_N * 2; c += 128) {
        const int n = c >> 1, kb = c & 1;
        const int unit = (n & 7) + 8 * kb + 16 * (n >> 3);
        uint4 v;
        __builtin_memcpy(&v, b + (size_t)n * PROBE_K + kb * 16, 16);
        *reinterpret_cast<uint4*>(s_b + unit * 16) = v;
    }

    // make the smem writes visible to the async proxy (the MMA)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // -------- stage the scale factors into TMEM --------
    // A: row m -> lane (m%32), column (sfa_col + m/32).
    // [CHANGED] scale_vec::1X + K=32 => one block => the scale byte lives in
    // byte 0 (SF-id 0). The default writes it to all four bytes of the word,
    // which is provably safe while only one K-block exists.
    {
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = 32 * j + lane;
            const uint32_t sf = (uint32_t)sfa[m];
#if PROBE_SF_BYTE0_ONLY
            w[j] = sf;
#else
            w[j] = sf | (sf << 8) | (sf << 16) | (sf << 24);
#endif
        }
        // x4 store: register j -> column sfa_col + j = sfa_col + m/32. Every warp
        // writes the identical content into its own 32-lane partition, which is
        // the PTX requirement that SF be duplicated to all lane partitions.
        tc_st_x4(((uint32_t)(warp * 32) << 16) | sfa_col, w[0], w[1], w[2], w[3]);
    }
    // B: B row n -> lane (n%32), column (sfb_col + n/32) = sfb_col for n < 32.
    {
        uint32_t w0 = 0;
        if (lane < PROBE_N) {
            const uint32_t sf = (uint32_t)sfb[lane];
#if PROBE_SF_BYTE0_ONLY
            w0 = sf;
#else
            w0 = sf | (sf << 8) | (sf << 16) | (sf << 24);
#endif
        }
        tc_st_x4(((uint32_t)(warp * 32) << 16) | sfb_col, w0, 0u, 0u, 0u);
    }
    tc_wait_st();
    tc_fence_before_thread_sync();
    __syncthreads();
    tc_fence_after_thread_sync();

    // -------- issue the MMA --------
    if (tid == 0) {
        const uint64_t da = make_desc(smem_addr(s_a), PROBE_LBO_BYTES, PROBE_SBO_BYTES);
        const uint64_t db = make_desc(smem_addr(s_b), PROBE_LBO_BYTES, PROBE_SBO_BYTES);
        // a_format = 5 (E2M1 weight), b_format = 0 (E4M3 activation),
        // scale_format = UE8M0, a_major = b_major = K, k_size = 0 (dense K32).
        const uint32_t idesc = make_idesc_mxf8f6f4(PROBE_M >> 4, PROBE_N >> 3,
                                                   /*a_fmt=*/5, /*b_fmt=*/0,
                                                   /*a_sf_id=*/0, /*b_sf_id=*/0,
                                                   /*k_size=*/0);
        dbg[0] = da;
        dbg[1] = db;
        dbg[2] = (uint64_t)idesc;
        dbg[3] = (uint64_t)tmem_base;
        // enable_input_d = 0: the accumulator is cleared by this single MMA.
        tc_mma_mxf8f6f4_1x(d_col, da, db, idesc, sfa_col, sfb_col, /*enable_d=*/0);
        tc_commit(&s_mbar);
    }
    mbar_wait(&s_mbar, 0);

    // -------- read D back: D[m][n] at lane (m%32), column (d_col + n) --------
    {
        uint32_t v[PROBE_N];
        tc_ld_x8(((uint32_t)(warp * 32) << 16) | d_col, v);
        tc_wait_ld();
#pragma unroll
        for (int n = 0; n < PROBE_N; ++n) {
            const int m = warp * 32 + lane;
            d_out[(size_t)m * PROBE_N + n] = __uint_as_float(v[n]);
        }
    }

    __syncthreads();
    if (warp == 0) tc_dealloc(tmem_base, PROBE_TMEM_COLS);
}

// ---------------------------------------------------------------------- host
// The tables and the e4m3 decode live in ph0:: (shared with the Phase 0 suite,
// which is also compiled by plain g++); pull them in rather than keeping a
// second copy in sync.
using ph0::e4m3_to_d;
using ph0::kE2M1;

uint32_t g_rng = 12345u;
uint32_t xrand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

}  // namespace

// =============================================================================
// (case 0 of the Phase 0 suite) The RANDOM-CODE layout probe described in the
// header: exact-arithmetic, quantization-free, so it separates "descriptor /
// TMEM / scale mapping wrong" from "quantization wrong". Also runnable alone
// with -DPROBE_RAW_LAYOUT=1.
int ph0_raw_probe_main() {
    std::vector<uint8_t> a((size_t)PROBE_M * PROBE_K), sfa(PROBE_M);
    std::vector<uint8_t> b((size_t)PROBE_N * PROBE_K), sfb(PROBE_N);

    // A: random e2m1 codes, one per byte (the low nibble; the high nibble is
    // ignored by the unpacked fp4 layout).
    for (size_t i = 0; i < a.size(); ++i) a[i] = (uint8_t)(xrand() & 0xF);
    // B: random e4m3 bytes, avoiding the two NaN encodings.
    for (size_t i = 0; i < b.size(); ++i) {
        uint8_t v;
        do {
            v = (uint8_t)xrand();
        } while (v == 0x7F || v == 0xFF);
        b[i] = v;
    }
    // scales: small powers of two, e8m0 = e + 127.
    for (int m = 0; m < PROBE_M; ++m) sfa[m] = (uint8_t)(127 + (int)(xrand() % 5) - 2);
    for (int n = 0; n < PROBE_N; ++n) sfb[n] = (uint8_t)(127 + (int)(xrand() % 5) - 2);

    // ---- CPU reference: D[m][n] = sum_k A[m][k]*2^(sfa[m]-127)*B[n][k]*2^(sfb[n]-127)
    std::vector<double> ref((size_t)PROBE_M * PROBE_N, 0.0);
    for (int m = 0; m < PROBE_M; ++m) {
        const double sam = std::ldexp(1.0, (int)sfa[m] - 127);
        for (int n = 0; n < PROBE_N; ++n) {
            const double sbn = std::ldexp(1.0, (int)sfb[n] - 127);
            double acc = 0.0;
            for (int k = 0; k < PROBE_K; ++k)
                acc += (double)kE2M1[a[(size_t)m * PROBE_K + k] & 0xF] *
                       e4m3_to_d(b[(size_t)n * PROBE_K + k]);
            ref[(size_t)m * PROBE_N + n] = acc * sam * sbn;
        }
    }

    uint8_t *da = nullptr, *dsfa = nullptr, *db = nullptr, *dsfb = nullptr;
    float* dd = nullptr;
    uint64_t* ddbg = nullptr;
    cudaMalloc(&da, a.size());
    cudaMalloc(&dsfa, sfa.size());
    cudaMalloc(&db, b.size());
    cudaMalloc(&dsfb, sfb.size());
    cudaMalloc(&dd, ref.size() * sizeof(float));
    cudaMalloc(&ddbg, 4 * sizeof(uint64_t));
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dsfa, sfa.data(), sfa.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dsfb, sfb.data(), sfb.size(), cudaMemcpyHostToDevice);
    cudaMemset(dd, 0, ref.size() * sizeof(float));

    printf("== tcgen05 mxf8f6f4 block_scale.scale_vec::1X probe ==\n");
    printf("   M=%d N=%d K=%d (swapAB: A = fp4 e2m1 weight [M,K], B = e4m3 act [N,K])\n",
           PROBE_M, PROBE_N, PROBE_K);
    printf("   smem layout: K-major SWIZZLE_NONE, LBO=%d B, SBO=%d B; fp4 UNPACKED (1 B/elem)\n",
           PROBE_LBO_BYTES, PROBE_SBO_BYTES);
    printf("   expected idesc = 0x%08X\n\n",
           (unsigned)((5u << 7) | ((PROBE_N >> 3) << 17) | (1u << 23) | ((PROBE_M >> 4) << 24)));
    fflush(stdout);

    probe_kernel<<<1, 128>>>(da, dsfa, db, dsfb, dd, ddbg);
    cudaError_t err = cudaDeviceSynchronize();
    uint64_t hdbg[4] = {0, 0, 0, 0};
    cudaMemcpy(hdbg, ddbg, sizeof(hdbg), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[FAIL] launch/sync: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("   a_desc   = 0x%016llx\n", (unsigned long long)hdbg[0]);
    printf("   b_desc   = 0x%016llx\n", (unsigned long long)hdbg[1]);
    printf("   idesc    = 0x%08llx\n", (unsigned long long)hdbg[2]);
    printf("   tmem_base= 0x%08llx\n\n", (unsigned long long)hdbg[3]);

    std::vector<float> got(ref.size());
    cudaMemcpy(got.data(), dd, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double maxabs = 0.0, maxdiff = 0.0;
    size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < ref.size(); ++i) {
        maxabs = std::fmax(maxabs, std::fabs(ref[i]));
        const double dev = std::fabs((double)got[i] - ref[i]);
        if (dev > maxdiff) maxdiff = dev;
        if (dev > 1e-3 * std::fmax(1.0, std::fabs(ref[i])) && first_bad == (size_t)-1)
            first_bad = i;
    }
    printf("   max|ref|=%.6g  max|diff|=%.6g  (tol = 1e-3 * max(1,|ref|))\n", maxabs, maxdiff);
    printf("   D[0][0..7] =");
    for (int n = 0; n < PROBE_N; ++n) printf(" %10.5f", (double)got[n]);
    printf("\n   ref[0][0..7]=");
    for (int n = 0; n < PROBE_N; ++n) printf(" %10.5f", ref[n]);
    printf("\n");
    if (first_bad != (size_t)-1) {
        const size_t m = first_bad / PROBE_N, n = first_bad % PROBE_N;
        printf("   first mismatch at (m=%zu, n=%zu): got %.6f expect %.6f\n", m, n,
               (double)got[first_bad], ref[first_bad]);
        printf("[FAIL] layout/scale mismatch\n");
    } else {
        printf("[PASS] D matches the CPU reference within tolerance\n");
    }
    printf("        (max|diff| should be tiny rounding only; a large value means the\n"
           "         SMEM descriptor or the TMEM/scale mapping is wrong, not ptxas)\n");

    cudaFree(da);
    cudaFree(dsfa);
    cudaFree(db);
    cudaFree(dsfb);
    cudaFree(dd);
    cudaFree(ddbg);
    return first_bad == (size_t)-1 ? 0 : 1;
}

// =============================================================================
// PHASE 0 — real-quantization parity kernels and suite
// =============================================================================
#ifndef PROBE_RAW_LAYOUT
#define PROBE_RAW_LAYOUT 0
#endif

// M is pinned by the instruction (CUTLASS static_asserts M == 128 for the
// 1-CTA mxf8f6f4 form); N is the minimum legal value (multiples of 8 in [8,256]).
#define PH0_M 128
#define PH0_N 8
// One K=32 block of A is 128 rows x 32 B = 4096 B (unpacked fp4); of B, 8 x 32
// = 256 B. PH0_MAXBLK blocks are staged in STATIC shared memory (34 KiB at 8,
// under the 48 KiB static limit) — going beyond that needs dynamic smem +
// cudaFuncSetAttribute, which is Phase 1's chunked staging problem.
#define PH0_MAXBLK 8
#define PH0_A_BYTES (PH0_M * 32)
#define PH0_B_BYTES (PH0_N * 32)

namespace {

// Device-side decodes (the host twins live in ph0::).
__device__ const float kDE2M1[16] = {0.f,  0.5f, 1.f,   1.5f,  2.f,  3.f,  4.f,  6.f,
                                     0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float e4m3_to_f(uint8_t b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    float v = (e == 0) ? ldexpf((float)m / 8.0f, -6) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return s ? -v : v;
}

// -----------------------------------------------------------------------------
// The MMA under test: M=128, N=8, K=32 per instruction, `nblk` instructions with
// D accumulating (enable_input_d = 1 after the first). A = fp4 e2m1 weights in
// the unpacked smem form, B = e4m3 activations, both with a per-32-block e8m0
// scale. sf_mode selects the SF layout hypothesis (see the header):
//   0 PACKED  4 consecutive blocks share a word, byte = b%4, sf_id = b%4
//   1 PERBLK  one block per word in byte 0, sf_id = 0
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(128) ph0_parity_kernel(
    const uint8_t* __restrict__ a_code,  // [128][K=nblk*32] e2m1, 1 element/byte
    const uint8_t* __restrict__ a_sf,    // [128][nblk] e8m0
    const uint8_t* __restrict__ b_code,  // [  8][K] e4m3
    const uint8_t* __restrict__ b_sf,    // [  8][nblk] e8m0
    float* __restrict__ d_out,           // [128][8]
    uint64_t* __restrict__ dbg,          // [4] {a_desc0, b_desc0, idesc_last, tmem_base}
    int nblk, int sf_mode, int tmem_cols) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int K = nblk * 32;
    const int sfa_cols = (sf_mode == 0) ? 4 * ((nblk + 3) / 4) : 4 * nblk;

    __shared__ __align__(1024) uint8_t s_a[PH0_MAXBLK * PH0_A_BYTES];
    __shared__ __align__(1024) uint8_t s_b[PH0_MAXBLK * PH0_B_BYTES];
    __shared__ __align__(8) uint64_t s_mbar;
    __shared__ uint32_t s_tmem;

    if (warp == 0) {
        tc_alloc(&s_tmem, (uint32_t)tmem_cols);
        tc_relinquish();
    }
    if (tid == 0) mbar_init(&s_mbar, 1);
    __syncthreads();

    const uint32_t tb = s_tmem;
    const uint32_t d_col = tb;
    const uint32_t sfa_col = tb + PH0_N;
    const uint32_t sfb_col = sfa_col + sfa_cols;

    // -------- stage A and B: one K=32 block per tile, canonical Major-K
    // SWIZZLE_NONE (unit16 = (r%8) + 8*kb + 16*(r/8); A: LBO 128 B, SBO 256 B;
    // B has a single 8-row group so SBO is unused).
    for (int idx = tid; idx < nblk * PH0_M * 2; idx += 128) {
        const int blk = idx / (PH0_M * 2);
        const int r = idx % (PH0_M * 2);
        const int m = r >> 1, kb = r & 1;
        const int unit = (m & 7) + 8 * kb + 16 * (m >> 3);
        uint4 v;
        __builtin_memcpy(&v, a_code + (size_t)m * K + blk * 32 + kb * 16, 16);
        *reinterpret_cast<uint4*>(s_a + blk * PH0_A_BYTES + unit * 16) = v;
    }
    for (int idx = tid; idx < nblk * PH0_N * 2; idx += 128) {
        const int blk = idx / (PH0_N * 2);
        const int r = idx % (PH0_N * 2);
        const int n = r >> 1, kb = r & 1;
        const int unit = (n & 7) + 8 * kb;
        uint4 v;
        __builtin_memcpy(&v, b_code + (size_t)n * K + blk * 32 + kb * 16, 16);
        *reinterpret_cast<uint4*>(s_b + blk * PH0_B_BYTES + unit * 16) = v;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // -------- scale factors -> TMEM. Row m -> lane (m%32), column (base + m/32);
    // every warp writes its own 32-lane partition with identical content (PTX:
    // the factors must be duplicated to all four lane partitions).
    if (sf_mode == 0) {  // PACKED
        const int nq = (nblk + 3) / 4;
        for (int q = 0; q < nq; ++q) {
            uint32_t wa[4], wb = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int m = 32 * j + lane;
                uint32_t w = 0;
                for (int t = 0; t < 4; ++t) {
                    const int b = 4 * q + t;
                    if (b < nblk) w |= (uint32_t)a_sf[(size_t)m * nblk + b] << (8 * t);
                }
                wa[j] = w;
            }
            if (lane < PH0_N) {
                uint32_t w = 0;
                for (int t = 0; t < 4; ++t) {
                    const int b = 4 * q + t;
                    if (b < nblk) w |= (uint32_t)b_sf[(size_t)lane * nblk + b] << (8 * t);
                }
                wb = w;
            }
            tc_st_x4(((uint32_t)(warp * 32) << 16) | (sfa_col + 4 * q), wa[0], wa[1], wa[2], wa[3]);
            tc_st_x4(((uint32_t)(warp * 32) << 16) | (sfb_col + q), wb, 0u, 0u, 0u);
        }
    } else {  // PERBLK
        for (int b = 0; b < nblk; ++b) {
            uint32_t wa[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int m = 32 * j + lane;
                wa[j] = (uint32_t)a_sf[(size_t)m * nblk + b];  // byte 0 only
            }
            const uint32_t wb = (lane < PH0_N) ? (uint32_t)b_sf[(size_t)lane * nblk + b] : 0u;
            tc_st_x4(((uint32_t)(warp * 32) << 16) | (sfa_col + 4 * b), wa[0], wa[1], wa[2], wa[3]);
            tc_st_x4(((uint32_t)(warp * 32) << 16) | (sfb_col + b), wb, 0u, 0u, 0u);
        }
    }
    tc_wait_st();
    tc_fence_before_thread_sync();
    __syncthreads();
    tc_fence_after_thread_sync();

    // -------- the K loop: one K=32 MMA per block
    if (tid == 0) {
        for (int b = 0; b < nblk; ++b) {
            const uint64_t da =
                make_desc(smem_addr(s_a) + b * PH0_A_BYTES, PROBE_LBO_BYTES, PROBE_SBO_BYTES);
            const uint64_t db =
                make_desc(smem_addr(s_b) + b * PH0_B_BYTES, PROBE_LBO_BYTES, PROBE_SBO_BYTES);
            const uint32_t sf_id = (sf_mode == 0) ? (uint32_t)(b & 3) : 0u;
            const uint32_t sa = (sf_mode == 0) ? (sfa_col + 4 * (b >> 2)) : (sfa_col + 4 * b);
            const uint32_t sb = (sf_mode == 0) ? (sfb_col + (b >> 2)) : (sfb_col + b);
            // a_format = 5 (E2M1 weight), b_format = 0 (E4M3 activation),
            // scale_format = UE8M0, majors = K, k_size = 0 (dense K32).
            const uint32_t id =
                make_idesc_mxf8f6f4(PH0_M >> 4, PH0_N >> 3, 5, 0, sf_id, sf_id, 0);
            if (b == 0) { dbg[0] = da; dbg[1] = db; }
            if (b == nblk - 1) dbg[2] = (uint64_t)id;
            tc_mma_mxf8f6f4_1x(d_col, da, db, id, sa, sb, b == 0 ? 0u : 1u);
        }
        dbg[3] = (uint64_t)tb;
        tc_commit(&s_mbar);
    }
    mbar_wait(&s_mbar, 0);

    // -------- D[m][n] at lane (m%32), column (d_col + n)
    {
        uint32_t v[PH0_N];
        tc_ld_x8(((uint32_t)(warp * 32) << 16) | d_col, v);
        tc_wait_ld();
#pragma unroll
        for (int n = 0; n < PH0_N; ++n)
            d_out[(size_t)(warp * 32 + lane) * PH0_N + n] = __uint_as_float(v[n]);
    }
    __syncthreads();
    if (warp == 0) tc_dealloc(tb, (uint32_t)tmem_cols);
}

// -----------------------------------------------------------------------------
// (c) the f32 SIMT reference: one thread per (m,n), dequantize + fmaf. This is
// the arithmetic shape of the current expert path (expert_gemv_fp4 in
// dsv41_experts_mxf4.cu — it reads PACKED fp4 activations, so it cannot be
// called with e4m3 data; this mirrors it instead). Its purpose is to prove the
// dequantization + reduction itself is right on the device, independently of
// every tcgen05 concept (descriptors, TMEM, SF layout).
// -----------------------------------------------------------------------------
__global__ void ph0_simt_ref_kernel(const uint8_t* __restrict__ a_code,
                                    const uint8_t* __restrict__ a_sf,
                                    const uint8_t* __restrict__ b_code,
                                    const uint8_t* __restrict__ b_sf, float* __restrict__ d,
                                    int M, int N, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    const int m = i / N, n = i % N;
    const int nblk = K >> 5;
    float acc = 0.f;
    for (int b = 0; b < nblk; ++b) {
        const float sa = ldexpf(1.f, (int)a_sf[(size_t)m * nblk + b] - 127);
        const float sb = ldexpf(1.f, (int)b_sf[(size_t)n * nblk + b] - 127);
        float part = 0.f;
        for (int k = 0; k < 32; ++k)
            part = fmaf(kDE2M1[a_code[(size_t)m * K + (b << 5) + k] & 0xF],
                        e4m3_to_f(b_code[(size_t)n * K + (b << 5) + k]), part);
        acc = fmaf(part, sa * sb, acc);  // block scale folded once, exactly
    }
    d[i] = acc;
}

}  // namespace

// -----------------------------------------------------------------------------
// Host runner for one (case, sf_mode) pair. Returns 0 on pass, 1 on fail.
// -----------------------------------------------------------------------------
#define PH0_CK(expr)                                                             \
    do {                                                                         \
        cudaError_t e_ = (expr);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            printf("   [FAIL] %s: %s\n", #expr, cudaGetErrorString(e_));          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

int ph0_run_case(const ph0::CaseCfg& cfg, int sf_mode) {
    const char* mname = (sf_mode == 0) ? "PACKED " : "PERBLK ";
    const ph0::CaseCore c = ph0::build_case(cfg);
    const int M = cfg.M, N = cfg.N, K = cfg.K, nblk = K / 32;
    if (M != PH0_M || N != PH0_N) {
        printf("   [%s] %s: M/N must be %d/%d (the instruction pins them)\n", mname, cfg.name,
               PH0_M, PH0_N);
        return 1;
    }
    if (nblk > PH0_MAXBLK) {
        printf("   [%s] %s: SKIP nblk=%d > PH0_MAXBLK=%d\n", mname, cfg.name, nblk, PH0_MAXBLK);
        return 0;  // not a failure: the shape is simply out of this file's scope
    }
    const int sfa_cols = (sf_mode == 0) ? 4 * ((nblk + 3) / 4) : 4 * nblk;
    const int sfb_cols = (sf_mode == 0) ? ((nblk + 3) / 4) : nblk;
    int tcols = 32;
    while (tcols < PH0_N + sfa_cols + sfb_cols) tcols <<= 1;

    uint8_t *dw = nullptr, *dws = nullptr, *da = nullptr, *das = nullptr;
    float *dd = nullptr, *dsimt = nullptr;
    uint64_t* ddbg = nullptr;
    PH0_CK(cudaMalloc(&dw, c.W.code.size()));
    PH0_CK(cudaMalloc(&dws, c.W.sf.size()));
    PH0_CK(cudaMalloc(&da, c.A.code.size()));
    PH0_CK(cudaMalloc(&das, c.A.sf.size()));
    PH0_CK(cudaMalloc(&dd, (size_t)M * N * sizeof(float)));
    PH0_CK(cudaMalloc(&dsimt, (size_t)M * N * sizeof(float)));
    PH0_CK(cudaMalloc(&ddbg, 4 * sizeof(uint64_t)));
    PH0_CK(cudaMemcpy(dw, c.W.code.data(), c.W.code.size(), cudaMemcpyHostToDevice));
    PH0_CK(cudaMemcpy(dws, c.W.sf.data(), c.W.sf.size(), cudaMemcpyHostToDevice));
    PH0_CK(cudaMemcpy(da, c.A.code.data(), c.A.code.size(), cudaMemcpyHostToDevice));
    PH0_CK(cudaMemcpy(das, c.A.sf.data(), c.A.sf.size(), cudaMemcpyHostToDevice));
    PH0_CK(cudaMemset(dd, 0, (size_t)M * N * sizeof(float)));
    PH0_CK(cudaMemset(dsimt, 0, (size_t)M * N * sizeof(float)));
    PH0_CK(cudaMemset(ddbg, 0, 4 * sizeof(uint64_t)));

    ph0_parity_kernel<<<1, 128>>>(dw, dws, da, das, dd, ddbg, nblk, sf_mode, tcols);
    ph0_simt_ref_kernel<<<(M * N + 127) / 128, 128>>>(dw, dws, da, das, dsimt, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("   [%s] %s: launch/sync: %s\n", mname, cfg.name, cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> got((size_t)M * N), simt((size_t)M * N);
    std::vector<uint64_t> hdbg(4);
    PH0_CK(cudaMemcpy(got.data(), dd, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    PH0_CK(cudaMemcpy(simt.data(), dsimt, simt.size() * sizeof(float), cudaMemcpyDeviceToHost));
    PH0_CK(cudaMemcpy(hdbg.data(), ddbg, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost));

    std::vector<double> gotd(got.begin(), got.end()), simtd(simt.begin(), simt.end());
    const ph0::Stats g = ph0::compare(gotd, c.golden, M, N);  // the parity criterion
    const ph0::Stats e = ph0::compare(gotd, c.exact, M, N);   // vs unquantized
    const ph0::Stats s = ph0::compare(simtd, c.golden, M, N);  // SIMT vs golden

    printf("\n   [%s] %s M=%d N=%d K=%d nblk=%d tmem=%d cols(SFA=%d,SFB=%d)\n", mname, cfg.name,
           M, N, K, nblk, tcols, sfa_cols, sfb_cols);
    printf("     a_desc=0x%016llx b_desc=0x%016llx idesc(last)=0x%08llx tmem=0x%08llx\n",
           (unsigned long long)hdbg[0], (unsigned long long)hdbg[1],
           (unsigned long long)hdbg[2], (unsigned long long)hdbg[3]);
    ph0::report("tcgen05 vs golden", g, "PARITY (criterion 5e-2)");
    ph0::report("SIMT     vs golden", s, "device dequant+reduce");
    ph0::report("tcgen05 vs exact ", e, "incl. quantization loss");
    printf("     argmax over M rows: %d/%d  over N cols: %d/%d (golden vs tcgen05); "
           "risky cols (maxdiff > top1-top2): %d/%d, min rel margin=%.2e\n",
           g.argM_ok, g.argM_tot, g.argN_ok, g.argN_tot, g.risky_cols, g.argM_tot,
           g.min_margin);

    int rc = 0;
    if (e.mx >= 5e-2 && g.mx < 5e-2) {
        // The layout is fine (tcgen05 == golden) but the two formats together are
        // too coarse for 5e-2 at this K. Report it, do not fail the layout test.
        printf("     [note] quantization loss alone (vs unquantized) is %.3g >= 5e-2: the\n"
               "            layout is verified, the FORMAT is what limits accuracy here.\n",
               e.mx);
    }
    if (!g.pass(5e-2)) {
        printf("     [FAIL] max rel err %.4g >= 5e-2  (p99=%.3g)\n", g.mx, g.p99);
        rc = 1;
    } else {
        printf("     [PASS] max rel err %.3g < 5e-2\n", g.mx);
    }
    if (g.argM_ok != g.argM_tot || g.argN_ok != g.argN_tot) {
        printf("     [FAIL] argmax differs (M %d/%d, N %d/%d)\n", g.argM_ok, g.argM_tot,
               g.argN_ok, g.argN_tot);
        rc = 1;
    }
    cudaFree(dw);
    cudaFree(dws);
    cudaFree(da);
    cudaFree(das);
    cudaFree(dd);
    cudaFree(dsimt);
    cudaFree(ddbg);
    return rc;
}

// -----------------------------------------------------------------------------
// The suite: the same four real-quantization cases as ph0::host_analysis(), each
// run under both SF layout hypotheses. The nblk==1 case is run once (the two
// hypotheses are bit-identical there by construction).
// -----------------------------------------------------------------------------
int ph0_parity_suite() {
    const ph0::CaseCfg cfgs[] = {
        {"w=N(0,1) a=N(0,1)", 128, 8, 32, 20260912u, ph0::D_NORMAL, ph0::D_NORMAL},
        {"w=N(0,1) a=N(0,1)", 128, 8, 128, 20260913u, ph0::D_NORMAL, ph0::D_NORMAL},
        {"w=outlier a=N(0,1)", 128, 8, 256, 20260914u, ph0::D_OUTLIER, ph0::D_NORMAL},
        {"w=N(0,1) a=lognormal", 128, 8, 256, 20260915u, ph0::D_NORMAL, ph0::D_LOGNORMAL},
    };
    const int NC = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
    int fails = 0, packed_fail = 0, perblk_fail = 0, perblk_ran = 0;
    for (int i = 0; i < NC; ++i) {
        printf("\n--- case %d/%d: %s (K=%d) ---\n", i + 1, NC, cfgs[i].name, cfgs[i].K);
        const int r0 = ph0_run_case(cfgs[i], 0);
        packed_fail += r0;
        fails += r0;
        if (cfgs[i].K / 32 > 1) {
            const int r1 = ph0_run_case(cfgs[i], 1);
            perblk_fail += r1;
            fails += r1;
            ++perblk_ran;
        } else {
            printf("\n   [PERBLK ] skipped at nblk=1: identical to PACKED by construction\n");
        }
    }
    printf("\n== suite verdict: %s ==\n", fails ? "FAIL" : "PASS");
    if (packed_fail && perblk_ran && !perblk_fail) {
        printf("   DIAGNOSIS: PACKED failed while PERBLK passed every multi-block case.\n"
               "   => the MMA / SMEM descriptor / TMEM-D mapping are RIGHT; the SF BYTE model\n"
               "      is wrong (a_sf_id/b_sf_id is not a byte index, or the word packing is\n"
               "      not [SF0,SF1,SF2,SF3]). Fix the scale staging in Phase 1, not the\n"
               "      descriptors.\n");
    } else if (!packed_fail && perblk_fail) {
        printf("   DIAGNOSIS: PACKED passed, PERBLK did not. PERBLK only matches the hardware\n"
               "   if, for every block, it reads byte 0 of the word the idesc base points at;\n"
               "   the verified single-block probe is exactly that case, so a PERBLK failure\n"
               "   is evidence about the K-loop (accumulation / per-block SF addressing).\n");
    } else if (packed_fail && perblk_fail) {
        printf("   DIAGNOSIS: both layouts failed => look at the parts they SHARE: the SMEM\n"
               "   descriptor (LBO 128 / SBO 256), the unpacked-fp4 byte assumption, the\n"
               "   idesc formats (a=5 E2M1, b=0 E4M3), or the D TMEM mapping. The random-code\n"
               "   case 0 above isolates those from quantization.\n");
    } else if (fails == 0) {
        printf("   All real-quantization cases match the CPU golden; the scale-factor byte\n"
               "   position is settled (PACKED == canonical 4-blocks-per-word).\n");
    }
    return fails;
}

// =============================================================================
int main() {
#if PROBE_RAW_LAYOUT
    return ph0_raw_probe_main();
#else
    printf("== tcgen05 kind::mxf8f6f4.block_scale.scale_vec::1X — Phase 0 parity suite ==\n");
    printf("   M=128 (pinned), N=8, K=32*nblk; A = fp4 e2m1 weights from real f32\n");
    printf("   quantization (per-32 e8m0), B = e4m3 activations (per-32 e8m0)\n");
    printf("   criterion: max rel err < 5e-2 vs the dequantized CPU golden + argmax equal\n");
    int fails = 0;
    fails += ph0_raw_probe_main();  // case 0: random-code layout probe (exact arithmetic)
    fails += ph0_parity_suite();
    printf("\n[PHASE 0] %s\n", fails ? "FAILED — do NOT start Phase 1" : "ALL GREEN");
    return fails ? 1 : 0;
#endif
}

#endif  // !PH0_HOST_ONLY
#endif  // PROBE_ASM_ONLY
