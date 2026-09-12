// tests_dsv41_sh_exp_mrows.cu — the M-ROW acceptance test for
// `dsv41_gemm_fp8_sh_exp_fused` (kernels/cuda/dsv41_kernels.cu), the
// `template<M>` form of the shared-expert chain (`dsv41_gemm_fp8_sh_pair`).
//
// WHY THIS FILE EXISTS. The verify block's iron rule is
//
//     row r of an m-row launch  ==  the m=1 decode of row r, BIT FOR BIT
//
// `dsv41_gemm_fp8_sh_exp_fused` runs the whole block's shared expert in ONE
// launch: phase 1 maps the grid over (i-tile, activation-row group) so the block
// count is `ceil(n1/32) * ceil(m/fold_r)` instead of `ceil(n1/32)`, and phase 2
// carries `m` accumulators over one w2 row. Both phases are the M=1 kernel's
// bodies transcribed term for term (its C1-C8), and this suite checks exactly
// that claim:
//
//   1. BIT-IDENTITY of BOTH phase outputs: the m-row launch's `aq` / `aqsc`
//      (phase 1) and `out` (phase 2) are bit-identical (memcmp over the raw
//      bits, `uint32_t` compare — so NaN payloads and ±0 cannot pass by luck) to
//      m single-row `dsv41_gemm_fp8_sh_pair` calls on the same operands. Exact
//      equality, NOT a tolerance.
//   2. THE THREE KNOBS, each an independent arm:
//        * `fold_r in {1, 2, m}` — the phase-1 A/B knob (runtime argument). 1 is
//          the design's §3.3 default (the M=1 kernel's exact `s_af[j]` consume
//          path); >1 folds activation rows into the warp's registers (the mrows
//          form). ALL THREE must produce the same bits — they are one program
//          with a different partition of the same chains.
//        * `epi_add in {0, 1}` — 1 folds `ferrite_add(out, phase-2)` into the
//          store as a read-modify-write. Checked against the host's
//          `pre[i] + ref[i]`, the same operand pair in the same order.
//        * `act == nullptr` vs a real buffer — the by-product f32 rows must be
//          identical when requested and must not be written when null.
//   3. ROW INDEPENDENCE + FULL COVERAGE: the buffers are pre-filled with a NaN
//      sentinel and compared WHOLE, so a kernel that only did row 0 — or that
//      left a phase-1 tile unwritten — cannot pass.
//   4. THE DECLINE PATH: m outside 1..=8, fold_r outside 1..=m, k1/n1 not a
//      multiple of 32, `aq_stride` not 16-byte aligned, `out_stride < n2` and a
//      null operand must return 2 (the caller's "keep the per-row chain"
//      signal), never a real launch error.
//
// NOT checked here: end-to-end model parity (the Rust wiring is a separate
// change) and launch performance.
//
// A NOTE ON THE REFERENCE. `dsv41_gemm_fp8_sh_pair` (the M=1 arm the Rust side
// already wires) is the reference, NOT the five-launch (quant1 -> mx2 -> swiglu
// -> quant1 -> mx_add) chain: the M=1 kernel's own header carries the argument
// that IT is bit-identical to that chain, so this suite composes the two claims
// instead of re-deriving it.
//
// ---------------------------------------------------------------------------
// Build (needs nvcc, NO GPU — dsv41_kernels.cu is the only TU):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math \
//        -std=c++17 -o /tmp/t_sh_exp_mrows kernels/cuda/tests_dsv41_sh_exp_mrows.cu
// Run (needs ONE free GPU; peak allocation is a few dozen MB):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_sh_exp_mrows            # full suite
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_sh_exp_mrows --quick    # small shapes only
//
// ENV THAT CHANGES WHAT IS COVERED (read once per process, before main):
//   DSV41_GEMV_FP8_MODE  must be 4 for any arm to run; any other value makes the
//                        launcher decline and the whole suite reports SKIP.
//   DSV41_GEMV_A32       must be ON (the default); a32-staged also declines.
//   DSV41_NO_GEMV_FP8    set => both launchers decline => the suite reports SKIP.
#include "dsv41_kernels.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// ---------------------------------------------------------------- test utils
int g_fails = 0;
int g_skips = 0;

#define SH_CHECK(expr, fmt, ...)                                                       \
    do {                                                                               \
        if (!(expr)) {                                                                 \
            printf("    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);     \
            ++g_fails;                                                                 \
        }                                                                              \
    } while (0)

uint32_t g_rng = 20260912u;
uint32_t sh_xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
int sh_irand(int n) { return (int)(sh_xr() % (uint32_t)n); }

// e4m3 byte that is NOT a NaN. exp == 15 && mantissa == 7 is NaN (0x7F / 0xFF);
// excluding the code keeps a bit-for-bit failure unambiguous.
uint8_t sh_e4m3_byte() {
    uint8_t b;
    do {
        b = (uint8_t)(sh_xr() & 0xFFu);
    } while ((b & 0x7Fu) == 0x7Fu);
    return b;
}

// A sane ue8m0 exponent (2^-3 .. 2^5): keeps the k-long products inside f32.
uint8_t sh_ue8m0_byte() { return (uint8_t)(124 + sh_irand(9)); }

// A power-of-two f32 activation scale, the shape `quant_fp8(round_scale=true)`
// emits.
float sh_act_scale() { return std::ldexp(1.0f, sh_irand(13) - 6); }

// The NaN sentinel: "never written" must be distinguishable from "written small".
void sh_fill_nan(std::vector<float>& v) {
    const float nan = __builtin_nanf("");
    uint32_t bits;
    std::memcpy(&bits, &nan, 4);
    for (size_t i = 0; i < v.size(); ++i) std::memcpy(&v[i], &bits, 4);
}

bool sh_bits_equal(const std::vector<float>& x, const std::vector<float>& y, size_t* at) {
    if (x.size() != y.size()) { *at = 0; return false; }
    for (size_t i = 0; i < x.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &x[i], 4);
        std::memcpy(&b, &y[i], 4);
        if (a != b) { *at = i; return false; }
    }
    return true;
}

bool sh_bytes_equal(const std::vector<uint8_t>& x, const std::vector<uint8_t>& y, size_t* at) {
    if (x.size() != y.size()) { *at = 0; return false; }
    for (size_t i = 0; i < x.size(); ++i)
        if (x[i] != y[i]) { *at = i; return false; }
    return true;
}

// One f32 add, forced through memory so fast-math cannot fuse it with anything.
// This is the host side of the `epi_add` arm: the kernel's store is
// `out[i] = out[i] + acc`, and IEEE-754 addition is exactly rounded, so the two
// sides agree bit for bit.
float sh_host_add(float x, float y) {
    volatile float vx = x, vy = y;
    return vx + vy;
}

// -------------------------------------------------------------- one shape arm
// `fold_r` is the phase-1 fold knob; `epi_add` folds the trailing add;
// `with_act` decides whether the by-product f32 rows are asked for.
int sh_case(const char* tag, int m, int fold_r, int n1, int k1, int n2, float limit,
            int epi_add, bool with_act) {
    const int nb_k1 = k1 / 32;      // phase-1 k-blocks / a_scale row width
    const int nb_k2 = n1 / 32;      // phase-2 k-blocks / aqsc row width
    SH_CHECK((k1 % 32) == 0 && (n1 % 32) == 0 && (n1 % 16) == 0 && m >= 1 && m <= 8,
             "[%s] bad shape", tag);

    // ---- host operands ----------------------------------------------------
    std::vector<uint8_t> ha((size_t)m * (size_t)k1);
    std::vector<float> hasc((size_t)m * (size_t)nb_k1);
    std::vector<uint8_t> hwg((size_t)n1 * (size_t)k1), hwu((size_t)n1 * (size_t)k1);
    // w_scale is indexed [(row >> 5) * nb_k1 + kb], so one row per 32-row block
    // plus a pad row keeps an `n1 % 32 != 0` shape readable (the launcher
    // declines it, but the buffer is sized so a bug cannot fault).
    std::vector<uint8_t> hwgs((size_t)(n1 / 32 + 2) * (size_t)nb_k1);
    std::vector<uint8_t> hwus((size_t)(n1 / 32 + 2) * (size_t)nb_k1);
    std::vector<uint8_t> hw2((size_t)n2 * (size_t)n1);
    std::vector<uint8_t> hw2s((size_t)(n2 / 32 + 2) * (size_t)nb_k2);
    for (auto& b : ha) b = sh_e4m3_byte();
    for (auto& s : hasc) s = sh_act_scale();
    for (auto& b : hwg) b = sh_e4m3_byte();
    for (auto& b : hwu) b = sh_e4m3_byte();
    for (auto& b : hwgs) b = sh_ue8m0_byte();
    for (auto& b : hwus) b = sh_ue8m0_byte();
    for (auto& b : hw2) b = sh_e4m3_byte();
    for (auto& b : hw2s) b = sh_ue8m0_byte();

    const size_t bytes_a = (size_t)m * (size_t)k1;
    const size_t bytes_asc = (size_t)m * (size_t)nb_k1 * sizeof(float);
    const size_t bytes_wg = (size_t)n1 * (size_t)k1;
    const size_t bytes_wgs = hwgs.size();
    const size_t bytes_w2 = (size_t)n2 * (size_t)n1;
    const size_t bytes_w2s = hw2s.size();
    const size_t bytes_aq = (size_t)m * (size_t)n1;                    // fp8 bytes
    const size_t bytes_aqs = (size_t)m * (size_t)nb_k2 * sizeof(float);
    const size_t bytes_out = (size_t)m * (size_t)n2 * sizeof(float);
    const size_t bytes_act = (size_t)m * (size_t)n1 * sizeof(float);

    uint8_t *da = nullptr, *dwg = nullptr, *dwu = nullptr, *dwgs = nullptr, *dwus = nullptr;
    uint8_t *dw2 = nullptr, *dw2s = nullptr;
    uint8_t *daq_m = nullptr, *daq_r = nullptr;
    float *dasc = nullptr, *daqs_m = nullptr, *daqs_r = nullptr;
    float *dout_m = nullptr, *dout_r = nullptr, *dact_m = nullptr, *dact_r = nullptr;
    bool ok = true;
    ok &= cudaMalloc((void**)&da, bytes_a) == cudaSuccess;
    ok &= cudaMalloc((void**)&dasc, bytes_asc) == cudaSuccess;
    ok &= cudaMalloc((void**)&dwg, bytes_wg) == cudaSuccess;
    ok &= cudaMalloc((void**)&dwu, bytes_wg) == cudaSuccess;
    ok &= cudaMalloc((void**)&dwgs, bytes_wgs) == cudaSuccess;
    ok &= cudaMalloc((void**)&dwus, bytes_wgs) == cudaSuccess;
    ok &= cudaMalloc((void**)&dw2, bytes_w2) == cudaSuccess;
    ok &= cudaMalloc((void**)&dw2s, bytes_w2s) == cudaSuccess;
    ok &= cudaMalloc((void**)&daq_m, bytes_aq) == cudaSuccess;
    ok &= cudaMalloc((void**)&daq_r, bytes_aq) == cudaSuccess;
    ok &= cudaMalloc((void**)&daqs_m, bytes_aqs) == cudaSuccess;
    ok &= cudaMalloc((void**)&daqs_r, bytes_aqs) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_m, bytes_out) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_r, bytes_out) == cudaSuccess;
    ok &= cudaMalloc((void**)&dact_m, bytes_act) == cudaSuccess;
    ok &= cudaMalloc((void**)&dact_r, bytes_act) == cudaSuccess;
    SH_CHECK(ok, "[%s] cudaMalloc failed", tag);
    if (!ok) return 1;

    std::vector<float> sentinel((size_t)m * (size_t)n2);
    sh_fill_nan(sentinel);
    // The epi_add arm's `pre`: what the MoE accumulator holds before the shared
    // expert's contribution lands (a real value, from the same generator).
    std::vector<float> pre((size_t)m * (size_t)n2);
    for (auto& v : pre) v = (float)(sh_irand(2001) - 1000) * 0.01f;

    ok &= cudaMemcpy(da, ha.data(), bytes_a, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dasc, hasc.data(), bytes_asc, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dwg, hwg.data(), bytes_wg, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dwu, hwu.data(), bytes_wg, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dwgs, hwgs.data(), bytes_wgs, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dwus, hwus.data(), bytes_wgs, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dw2, hw2.data(), bytes_w2, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dw2s, hw2s.data(), bytes_w2s, cudaMemcpyHostToDevice) == cudaSuccess;
    const std::vector<uint8_t> sentinel_b(bytes_aq, 0x5Au);   // phase-1 sentinel
    ok &= cudaMemcpy(daq_m, sentinel_b.data(), bytes_aq, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(daq_r, sentinel_b.data(), bytes_aq, cudaMemcpyHostToDevice) == cudaSuccess;
    std::vector<float> sentinel_s((size_t)m * (size_t)nb_k2);
    sh_fill_nan(sentinel_s);
    ok &= cudaMemcpy(daqs_m, sentinel_s.data(), bytes_aqs, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(daqs_r, sentinel_s.data(), bytes_aqs, cudaMemcpyHostToDevice) == cudaSuccess;
    // epi_add = 0: the output starts at the NaN sentinel (coverage check).
    // epi_add = 1: it starts at `pre`, and the kernel's RMW must produce
    //              `pre + acc`.
    const std::vector<float>& out0 = (epi_add != 0) ? pre : sentinel;
    ok &= cudaMemcpy(dout_m, out0.data(), bytes_out, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dout_r, sentinel.data(), bytes_out, cudaMemcpyHostToDevice) == cudaSuccess;
    // `act` is [m, n1] f32 (a row per activation row), so its sentinel is its own
    // buffer -- `sentinel_s` is only [m, n1/32].
    std::vector<float> sentinel_act((size_t)m * (size_t)n1);
    sh_fill_nan(sentinel_act);
    ok &= cudaMemcpy(dact_m, sentinel_act.data(), bytes_act, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dact_r, sentinel_act.data(), bytes_act, cudaMemcpyHostToDevice) == cudaSuccess;
    SH_CHECK(ok, "[%s] cudaMemcpy H2D failed", tag);

    // ---- the m single-row REFERENCE (`dsv41_gemm_fp8_sh_pair`, M = 1) ------
    // Its phase-1 grid maps one block per 32 inter rows of ONE activation row,
    // and its phase-2 loop is grid-strided over n2 -- so launching it m times on
    // the m rows is exactly the chain this arm replaces.
    for (int r = 0; r < m; ++r) {
        const int rc1 = dsv41_gemm_fp8_sh_pair(
            da + (size_t)r * (size_t)k1, dasc + (size_t)r * (size_t)nb_k1, dwg, dwgs, dwu, dwus,
            limit, n1, k1,
            dact_r + (size_t)r * (size_t)n1,                 // act  [n1] (never null here)
            daq_r + (size_t)r * (size_t)n1,                  // aq   [n1]
            daqs_r + (size_t)r * (size_t)nb_k2,              // aqsc [n1/32]
            dw2, dw2s, n2, dout_r + (size_t)r * (size_t)n2, /*stream=*/nullptr);
        if (rc1 == 2) {
            printf("  [%s] SKIP: dsv41_gemm_fp8_sh_pair declined (mode=%d, no_gemv=%d)\n", tag,
                   g_gemv_fp8_mode, (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr));
            ++g_skips;
            return 0;
        }
        SH_CHECK(rc1 == 0, "[%s] reference row %d returned %d (%s)", tag, r, rc1,
                 cudaGetErrorString((cudaError_t)rc1));
    }

    // ---- the m-row launch -------------------------------------------------
    const int rc = dsv41_gemm_fp8_sh_exp_fused(
        da, dasc, dwg, dwgs, dwu, dwus, limit, n1, k1, m, fold_r,
        with_act ? dact_m : nullptr, with_act ? n1 : 0,
        daq_m, daqs_m, /*aq_stride=*/n1, /*aqsc_stride=*/nb_k2,
        dw2, dw2s, n2, /*out_stride=*/n2, epi_add, dout_m, /*stream=*/nullptr);
    if (rc == 2) {
        printf("  [%s] SKIP: dsv41_gemm_fp8_sh_exp_fused declined (mode=%d, a32=%d, no_gemv=%d)\n",
               tag, g_gemv_fp8_mode, (int)g_gemv_a32, (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr));
        ++g_skips;
        return 0;
    }
    SH_CHECK(rc == 0, "[%s] m-row launch returned %d (%s)", tag, rc,
             cudaGetErrorString((cudaError_t)rc));
    const cudaError_t se = cudaDeviceSynchronize();
    SH_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    // ---- compare ----------------------------------------------------------
    std::vector<uint8_t> aqm(bytes_aq), aqr(bytes_aq);
    std::vector<float> aqsm((size_t)m * (size_t)nb_k2), aqsr((size_t)m * (size_t)nb_k2);
    std::vector<float> om((size_t)m * (size_t)n2), orr((size_t)m * (size_t)n2);
    std::vector<float> am((size_t)m * (size_t)n1), arr((size_t)m * (size_t)n1);
    ok &= cudaMemcpy(aqm.data(), daq_m, bytes_aq, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(aqr.data(), daq_r, bytes_aq, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(aqsm.data(), daqs_m, bytes_aqs, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(aqsr.data(), daqs_r, bytes_aqs, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(om.data(), dout_m, bytes_out, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(orr.data(), dout_r, bytes_out, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(am.data(), dact_m, bytes_act, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(arr.data(), dact_r, bytes_act, cudaMemcpyDeviceToHost) == cudaSuccess;
    SH_CHECK(ok, "[%s] cudaMemcpy D2H failed", tag);

    // (a) phase 1: the fp8 pair
    size_t at = 0;
    const bool aq_same = sh_bytes_equal(aqm, aqr, &at);
    if (!aq_same)
        printf("    FAIL [%s] phase-1 aq byte diff at r=%zu c=%d: m-row 0x%02x  m=1 0x%02x\n", tag,
               at / (size_t)n1, (int)(at % (size_t)n1), aqm[at], aqr[at]);
    SH_CHECK(aq_same, "[%s] phase-1 aq differs", tag);

    // (b) phase 1: the per-32-block scales
    const bool aqs_same = sh_bits_equal(aqsm, aqsr, &at);
    if (!aqs_same) {
        uint32_t x, y;
        std::memcpy(&x, &aqsm[at], 4);
        std::memcpy(&y, &aqsr[at], 4);
        printf("    FAIL [%s] phase-1 aqsc diff at r=%zu c=%zu: m-row 0x%08x (%g)  m=1 0x%08x (%g)\n",
               tag, at / (size_t)nb_k2, at % (size_t)nb_k2, x, (double)aqsm[at], y,
               (double)aqsr[at]);
    }
    SH_CHECK(aqs_same, "[%s] phase-1 aqsc differs", tag);

    // (c) phase 2: `out`, with the epi_add expectation applied on the host.
    std::vector<float> expected = orr;
    if (epi_add != 0)
        for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = sh_host_add(pre[i], orr[i]);
    const bool out_same = sh_bits_equal(om, expected, &at);
    if (!out_same) {
        uint32_t x, y;
        std::memcpy(&x, &om[at], 4);
        std::memcpy(&y, &expected[at], 4);
        printf("    FAIL [%s] phase-2 out diff at r=%zu c=%zu: m-row 0x%08x (%g)  expected 0x%08x (%g)\n",
               tag, at / (size_t)n2, at % (size_t)n2, x, (double)om[at], y, (double)expected[at]);
    }
    SH_CHECK(out_same, "[%s] phase-2 out differs", tag);

    // (d) the by-product f32 rows: identical when asked for, untouched when not.
    if (with_act) {
        const bool act_same = sh_bits_equal(am, arr, &at);
        SH_CHECK(act_same, "[%s] by-product act differs (at %zu)", tag, at);
    } else {
        size_t sent = 0;
        for (size_t i = 0; i < am.size(); ++i) {
            uint32_t b;
            std::memcpy(&b, &am[i], 4);
            if ((b & 0x7FFFFFFFu) > 0x7F800000u) ++sent;   // still the NaN sentinel
        }
        SH_CHECK(sent == 0, "[%s] act == nullptr but %zu slot(s) look unwritten", tag, sent);
    }

    // (e) coverage: every phase-1 byte and every phase-2 element of every row
    // must have been written by BOTH sides (NaN sentinel = never written).
    size_t unwritten_out = 0;
    if (epi_add == 0)
        for (size_t i = 0; i < om.size(); ++i) {
            uint32_t bm, br;
            std::memcpy(&bm, &om[i], 4);
            std::memcpy(&br, &expected[i], 4);
            if (bm == br && (bm & 0x7FFFFFFFu) > 0x7F800000u) ++unwritten_out;
        }
    SH_CHECK(unwritten_out == 0, "[%s] %zu phase-2 element(s) left unwritten", tag, unwritten_out);
    size_t unwritten_aq = 0;
    for (size_t i = 0; i < aqm.size(); ++i)
        if (aqm[i] == 0x5Au && aqr[i] == 0x5Au) ++unwritten_aq;
    SH_CHECK(unwritten_aq == 0, "[%s] %zu phase-1 byte(s) left unwritten", tag, unwritten_aq);

    const bool pass = aq_same && aqs_same && out_same && unwritten_out == 0 && unwritten_aq == 0;
    if (pass)
        printf("  [%s] OK  m=%d fold_r=%d n1=%d k1=%d n2=%d limit=%g epi_add=%d act=%s\n", tag, m,
               fold_r, n1, k1, n2, (double)limit, epi_add, with_act ? "buf" : "null");

    cudaFree(da); cudaFree(dasc); cudaFree(dwg); cudaFree(dwu); cudaFree(dwgs); cudaFree(dwus);
    cudaFree(dw2); cudaFree(dw2s); cudaFree(daq_m); cudaFree(daq_r); cudaFree(daqs_m);
    cudaFree(daqs_r); cudaFree(dout_m); cudaFree(dout_r); cudaFree(dact_m); cudaFree(dact_r);
    return pass ? 0 : 1;
}

// ------------------------------------------------------------- the declines
int sh_case_declines() {
    const int k1 = 512, n1 = 64, n2 = 128;
    const int nb_k1 = k1 / 32, nb_k2 = n1 / 32, m = 2;
    std::vector<uint8_t> a((size_t)m * k1, 0x38), w1((size_t)n1 * k1, 0x38);
    std::vector<uint8_t> ws((size_t)(n1 / 32 + 2) * nb_k1, 0x7Bu);
    std::vector<uint8_t> w2((size_t)n2 * n1, 0x38), w2s((size_t)(n2 / 32 + 2) * nb_k2, 0x7Bu);
    std::vector<float> asc((size_t)m * nb_k1, 1.0f);
    std::vector<uint8_t> aq((size_t)m * n1, 0), aq16((size_t)m * (n1 + 16), 0);
    std::vector<float> aqs((size_t)m * nb_k2, 0.f);
    std::vector<float> out((size_t)m * n2, 0.f);

    auto up8 = [](const std::vector<uint8_t>& v) {
        uint8_t* p = nullptr; cudaMalloc((void**)&p, v.size());
        cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice); return p;
    };
    auto upf = [](const std::vector<float>& v) {
        float* p = nullptr; cudaMalloc((void**)&p, v.size() * 4);
        cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice); return p;
    };
    uint8_t *da = up8(a), *dw1 = up8(w1), *dws = up8(ws), *dw2 = up8(w2), *dw2s = up8(w2s);
    uint8_t *daq = up8(aq), *daq16 = up8(aq16);
    float *dasc = upf(asc), *daqs = upf(aqs), *dout = upf(out);

    struct { const char* what; int m, fold, n1, k1, aqs16, os; } bad[] = {
        {"m=0",           0, 1, 64, 512, 0, 128},
        {"m=9",           9, 1, 64, 512, 0, 128},
        {"fold_r=0",      2, 0, 64, 512, 0, 128},
        {"fold_r>m",      2, 3, 64, 512, 0, 128},
        {"k1%32!=0",      2, 1, 64, 500, 0, 128},
        {"n1%32!=0",      2, 1, 60, 512, 0, 128},
        {"aq_stride%16",  2, 1, 64, 512, 1, 128},
        {"out_stride<n2", 2, 1, 64, 512, 0,  64},
    };
    for (const auto& b : bad) {
        const int rc = dsv41_gemm_fp8_sh_exp_fused(
            da, dasc, dw1, dws, dw1, dws, 3.0f, b.n1, b.k1, b.m, b.fold, nullptr, 0,
            b.aqs16 ? daq16 : daq, daqs, b.aqs16 ? (b.n1 + 8) : b.n1, b.n1 / 32,
            dw2, dw2s, n2, b.os, 0, dout, /*stream=*/nullptr);
        SH_CHECK(rc == 2, "[declines] %s: expected 2, got %d", b.what, rc);
    }
    // A null operand must decline, not fault.
    SH_CHECK(dsv41_gemm_fp8_sh_exp_fused(nullptr, dasc, dw1, dws, dw1, dws, 3.0f, 64, 512, 2, 1,
                                         nullptr, 0, daq, daqs, 64, 2, dw2, dw2s, n2, n2, 0, dout,
                                         nullptr) == 2,
             "[declines] null a: expected 2");
    SH_CHECK(dsv41_gemm_fp8_sh_exp_fused(da, dasc, dw1, dws, dw1, dws, 3.0f, 64, 512, 2, 1, nullptr,
                                         0, nullptr, daqs, 64, 2, dw2, dw2s, n2, n2, 0, dout,
                                         nullptr) == 2,
             "[declines] null aq: expected 2");
    SH_CHECK(dsv41_gemm_fp8_sh_exp_fused(da, dasc, dw1, dws, dw1, dws, 3.0f, 64, 512, 2, 1, nullptr,
                                         0, daq, daqs, 64, 2, dw2, dw2s, n2, n2, 0, nullptr,
                                         nullptr) == 2,
             "[declines] null out: expected 2");
    printf("  [declines] OK  (m/fold_r range, k1%%32, n1%%32, aq_stride%%16, out_stride, null operands)\n");
    cudaFree(da); cudaFree(dw1); cudaFree(dws); cudaFree(dw2); cudaFree(dw2s); cudaFree(daq);
    cudaFree(daq16); cudaFree(dasc); cudaFree(daqs); cudaFree(dout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    printf("== dsv41 shared-expert M-row (sh_exp_fused<M>) acceptance ==\n");
    printf("   DSV41_GEMV_FP8_MODE=%d  g_gemv_a32=%d  a32_staged=%d  DSV41_NO_GEMV_FP8=%d\n",
           g_gemv_fp8_mode, (int)g_gemv_a32, (int)g_gemv_a32_staged,
           (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr));
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("   device %d: %s (sm_%d%d, %d SM, %d KB smem/block opt-in)\n", dev, prop.name,
               prop.major, prop.minor, prop.multiProcessorCount,
               (int)(prop.sharedMemPerBlockOptin / 1024));

    // The production shape: n1 = sh_il = 288 (TP8), k1 = dim = 5120, n2 = dim.
    // m = 6 is VERIFY_ROWS (the swallow/verify block); m = 1 is the degenerate
    // case the M=1 kernel must still agree with exactly.
    g_fails += sh_case("prod/m=6/fold1", 6, 1, 288, 5120, 5120, 3.0f, 0, false);
    g_fails += sh_case("prod/m=6/fold2", 6, 2, 288, 5120, 5120, 3.0f, 0, false);
    g_fails += sh_case("prod/m=6/fold6", 6, 6, 288, 5120, 5120, 3.0f, 0, false);
    g_fails += sh_case("prod/m=5/fold1", 5, 1, 288, 5120, 5120, 3.0f, 0, false);
    g_fails += sh_case("prod/m=1/fold1", 1, 1, 288, 5120, 5120, 3.0f, 0, false);
    // epi_add + the by-product `act` arms.
    g_fails += sh_case("prod/m=6/epi_add", 6, 1, 288, 5120, 5120, 3.0f, 1, false);
    g_fails += sh_case("prod/m=3/act", 3, 1, 288, 5120, 5120, 3.0f, 0, true);
    // limit = 0 is the "clamp off" arm of the swiglu epilogue.
    g_fails += sh_case("prod/m=6/nolimit", 6, 1, 288, 5120, 5120, 0.0f, 0, false);
    g_fails += sh_case_declines();

    if (!quick) {
        // Every dispatch case the launcher's switch can take.
        for (int m = 1; m <= 8; ++m)
            g_fails += sh_case("dispatch/m", m, 1, 64, 256, 128, 3.0f, 0, false);
        // Every fold_r in 1..m at one shape (the knob's whole range).
        for (int f = 1; f <= 8; ++f)
            g_fails += sh_case("fold/range", 8, f, 64, 256, 128, 3.0f, 0, false);
        // A shape whose phase-2 row count is NOT a multiple of 32 (the
        // grid-strided tail) and one where fold_r does not divide m.
        g_fails += sh_case("n2%32/m=5", 5, 3, 96, 640, 130, 3.0f, 0, false);
        // The smallest shape that still satisfies both shape gates.
        g_fails += sh_case("tiny/m=2", 2, 2, 32, 64, 32, 3.0f, 1, true);
    }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        printf("  sticky CUDA error after the suite: %s\n", cudaGetErrorString(ce));
        ++g_fails;
    }
    if (g_skips)
        printf("RESULT: %d check(s) FAILED, %d arm(s) SKIPPED\n", g_fails, g_skips);
    else
        printf(g_fails ? "RESULT: %d check(s) FAILED\n" : "RESULT: all checks passed\n", g_fails);
    return g_fails ? 1 : 0;
}
