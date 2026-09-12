// tests_dsv41_gemm_mrows_f32.cu — the acceptance test for
// `dsv41_gemm_fp8_mrows_f32` (kernels/cuda/dsv41_kernels.cu), B6: the
// MULTI-ROW form of the f32-activation GEMV (`dsv41_gemm_fp8_mx_f32`'s M=1
// program). Design: docs/agent/b6-mrows-f32-design.md §6.
//
// WHY THIS FILE EXISTS. The verify's wo_b is `m x quant_fp8 + 1 x proj_mrows`
// per layer; B6 replaces that with ONE launch reading the raw f32 activation.
// The program it must be judgeable against is the M=1 f32 GEMV (EAGER's
// `DSV41_WOB_F32` path), because the f32 domain's "materialisation" is the
// identity (`s_af[i] = a_f32[i]`, a pure copy — the gemv header's own words).
// So the contract this suite checks is
//
//     row r of the m-row f32 launch  ==  dsv41_gemm_fp8_mx_f32(row r), BITWISE
//
// and NOT "equal to the old quant_fp8 + mrows pair" — B6 deliberately SKIPS the
// quantise -> dequantise round trip, so its result is strictly more accurate and
// a byte comparison against the old path would be measuring the wrong thing
// (that difference is arm 6's diagnostic, reported and never asserted).
//
//   ARMS
//   1. BIT-IDENTITY: an m-row call is bit-identical (memcmp over the RAW f32
//      bits — so NaN payloads and ±0 cannot pass by luck) to m single-row
//      `dsv41_gemm_fp8_mx_f32` calls on the same rows of the same operand.
//   2. ROW COVERAGE: every (row, r) element must be written. Buffers are
//      pre-filled with a NaN sentinel and compared WHOLE, so with
//      `out_stride > n` the gaps must stay sentinel on both sides — a kernel
//      that only did row 0 cannot pass.
//   3. THE DECLINE TABLE: m outside 1..=8, k not a multiple of 32,
//      `a_stride < k`, `out_stride < n`, a null operand and the mode-0/1
//      reordering arms must return 2 (the caller's "keep the old pair" signal),
//      never a real launch error.
//   4. THE SHAPE MATRIX: the verify shape (a_stride 8x k — the TP8 case the
//      explicit `a_stride` parameter exists for), the draft shape (k = 8192, the
//      >48KB smem staircase), small shapes, and `a_stride > k` / `out_stride > n`
//      — the two ways a wrong stride silently corrupts.
//   5. GATE INVARIANCE (recorded, one setting per process): `DSV41_GEMV_A32` is
//      a `OnceLock` process gate the launcher does NOT read (on the f32 domain
//      both of its forms read the same word), so the requirement is that the
//      suite be green under BOTH settings — run it twice, 0 and 1.
//   6. DIAGNOSTIC (never pass/fail): the max relative difference against the OLD
//      path (`quant_fp8` + `dsv41_gemm_fp8_mrows`), i.e. the size of the
//      fp8-round-trip error B6 removes. Printed for the record.
//
// NOT checked here: end-to-end model parity (the Rust wiring is a separate
// change) and launch performance (arm 6's `nsys` count is the perf evidence).
//
// ---------------------------------------------------------------------------
// Build (needs nvcc, NO GPU — dsv41_kernels.cu is the only other TU):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math \
//        -std=c++17 -o /tmp/t_gemm_mrows_f32 kernels/cuda/tests_dsv41_gemm_mrows_f32.cu
// Run (needs ONE free GPU; peak allocation is ~50 MB at the draft shape):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_gemm_mrows_f32            # full suite
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_gemm_mrows_f32 --quick    # small shapes only
//   DSV41_GEMV_A32=1 ... /tmp/t_gemm_mrows_f32                   # arm 5, second half
//
// ENV THAT CHANGES WHAT IS COVERED (read once per process, before main):
//   DSV41_GEMV_FP8_MODE  must be >= 3 for any arm to run (0/1 reorder a lane's
//                        elements and the launcher declines them — that decline
//                        is itself checked). The default is 4.
//   DSV41_NO_GEMV_FP8    set => the M=1 reference itself takes the M-tile MMA
//                        path (a different expression), so the launcher declines
//                        and the whole suite reports SKIP.
//   DSV41_GEMV_FP8_WARPS / _ADAPTIVE / _WARPS_BIG change nwarps (the parity is
//                        invariant under them — rows are independent; the
//                        geometry only decides how many rows share a block's
//                        staging). Worth running under a couple of settings.
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

#define M32_CHECK(expr, fmt, ...)                                                      \
    do {                                                                               \
        if (!(expr)) {                                                                 \
            printf("    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);     \
            ++g_fails;                                                                 \
        }                                                                              \
    } while (0)

uint32_t g_rng = 20260912u;
uint32_t m32_xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
int m32_irand(int n) { return (int)(m32_xr() % (uint32_t)n); }

// e4m3 byte that is NOT a NaN. exp == 15 && mantissa == 7 is NaN (0x7F / 0xFF);
// excluding the code keeps a failure unambiguous (a NaN would also defeat the
// sentinel/coverage test, which relies on "NaN means never written").
uint8_t m32_e4m3_byte() {
    uint8_t b;
    do {
        b = (uint8_t)(m32_xr() & 0xFFu);
    } while ((b & 0x7Fu) == 0x7Fu);
    return b;
}

// A sane ue8m0 exponent: 2^(124..132) keeps the products inside f32 range.
uint8_t m32_ue8m0_byte() { return (uint8_t)(124 + m32_irand(9)); }

// The f32 activation on this path is a RAW model tensor (wo_a's product), i.e.
// full-range floats rather than a power-of-two quantised scale. Keep them modest
// so the k-fold sum stays well inside f32 (the M=1 reference and the m-row
// kernel must see bit-identical operands either way — the range only matters for
// the arm-6 diagnostic's relative measure).
float m32_act() { return (float)(m32_irand(2001) - 1000) * 0.001f; }

// Fill a buffer with a NaN sentinel so "never written" is distinguishable from
// "written with a small value" (the coverage half of arm 2).
void m32_fill_sentinel(std::vector<float>& v) {
    const float nan = __builtin_nanf("");
    uint32_t bits;
    std::memcpy(&bits, &nan, 4);
    for (size_t i = 0; i < v.size(); ++i) std::memcpy(&v[i], &bits, 4);
}

// Bit-for-bit compare (the RAW f32 bits, so -0.0 vs +0.0 and NaN payloads are
// differences — "bit-identical" is the contract, not "approximately equal").
bool m32_bits_equal(const std::vector<float>& x, const std::vector<float>& y, size_t* at) {
    if (x.size() != y.size()) { *at = 0; return false; }
    for (size_t i = 0; i < x.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &x[i], 4);
        std::memcpy(&b, &y[i], 4);
        if (a != b) { *at = i; return false; }
    }
    return true;
}

// -------------------------------------------------------------- one shape arm
// `a_stride` is the activation ROW PITCH in f32 elements (the B6 ABI's whole
// point): the verify passes `a_stride = 8 * k` (ol_total vs ol_local under TP8).
// `out_stride >= n` is the verify's shape (it writes `dim` of an `nh*hd` row).
// `diagnose` additionally runs the OLD path (quant_fp8 + mrows) and prints the
// max relative difference — never a pass/fail (arm 6).
int m32_case(const char* tag, int m, int n, int k, int a_stride, int out_stride, bool use_bias,
             bool diagnose) {
    const int nb_k = k / 32;
    M32_CHECK(k % 32 == 0 && n > 0 && a_stride >= k && out_stride >= n, "[%s] bad shape", tag);

    // The activation is [m, a_stride] with only the first k elements of each row
    // meaningful — the tail models the verify's true pitch and must never be read
    // by the kernel (a read of it would show up as a bit difference against the
    // M=1 reference, which sees only the row's first k elements).
    std::vector<float> ha((size_t)m * (size_t)a_stride);
    std::vector<uint8_t> hw((size_t)n * (size_t)k);
    // w_scale is indexed [row >> 5][kb] for every row < n, so the tail row block
    // needs its own row; two padding rows keep an `n % 32 != 0` shape readable.
    std::vector<uint8_t> hws((size_t)(n / 32 + 2) * (size_t)nb_k, 0);
    std::vector<float> hb(use_bias ? (size_t)n : 1);
    for (auto& v : ha) v = m32_act();
    for (auto& b : hw) b = m32_e4m3_byte();
    for (auto& b : hws) b = m32_ue8m0_byte();
    for (int i = 0; i < n && use_bias; ++i) hb[(size_t)i] = (float)(m32_irand(2001) - 1000) * 0.001f;

    float *da = nullptr, *db = nullptr, *dout_m = nullptr, *dout_r = nullptr;
    uint8_t *dw = nullptr, *dws = nullptr;
    const size_t bytes_a = (size_t)m * (size_t)a_stride * sizeof(float);
    const size_t bytes_w = (size_t)n * (size_t)k;
    const size_t bytes_ws = hws.size();
    const size_t bytes_out = (size_t)m * (size_t)out_stride * sizeof(float);
    bool ok = true;
    ok &= cudaMalloc((void**)&da, bytes_a) == cudaSuccess;
    ok &= cudaMalloc((void**)&dw, bytes_w) == cudaSuccess;
    ok &= cudaMalloc((void**)&dws, bytes_ws) == cudaSuccess;
    ok &= cudaMalloc((void**)&db, use_bias ? (size_t)n * sizeof(float) : 4u) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_m, bytes_out) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_r, bytes_out) == cudaSuccess;
    M32_CHECK(ok, "[%s] cudaMalloc failed", tag);
    if (!ok) return 1;

    std::vector<float> sentinel((size_t)m * (size_t)out_stride);
    m32_fill_sentinel(sentinel);
    ok &= cudaMemcpy(da, ha.data(), bytes_a, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dw, hw.data(), bytes_w, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dws, hws.data(), bytes_ws, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(db, hb.data(), use_bias ? (size_t)n * sizeof(float) : 4u,
                     cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dout_m, sentinel.data(), bytes_out, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dout_r, sentinel.data(), bytes_out, cudaMemcpyHostToDevice) == cudaSuccess;
    M32_CHECK(ok, "[%s] cudaMemcpy H2D failed", tag);

    // ---- the m-row launch (B6) --------------------------------------------
    const int rc = dsv41_gemm_fp8_mrows_f32(da, dw, dws, use_bias ? db : nullptr, dout_m, m, n, k,
                                            a_stride, out_stride, /*stream=*/nullptr);
    if (rc == 2) {
        printf("  [%s] SKIP: dsv41_gemm_fp8_mrows_f32 declined (mode=%d, no_gemv=%d)\n", tag,
               g_gemv_fp8_mode, (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr));
        ++g_skips;
        cudaFree(da); cudaFree(dw); cudaFree(dws); cudaFree(db);
        cudaFree(dout_m); cudaFree(dout_r);
        return 0;
    }
    M32_CHECK(rc == 0, "[%s] mrows_f32 launch returned %d (%s)", tag, rc,
              cudaGetErrorString((cudaError_t)rc));

    // ---- the m single-row references (the M=1 f32 GEMV = EAGER's program) --
    for (int r = 0; r < m; ++r) {
        const int rc1 = dsv41_gemm_fp8_mx_f32(da + (size_t)r * (size_t)a_stride, dw, dws,
                                              use_bias ? db : nullptr,
                                              dout_r + (size_t)r * (size_t)out_stride,
                                              n, k, /*stream=*/nullptr);
        M32_CHECK(rc1 == 0, "[%s] reference row %d returned %d (%s)", tag, r, rc1,
                  cudaGetErrorString((cudaError_t)rc1));
    }
    const cudaError_t se = cudaDeviceSynchronize();
    M32_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> om((size_t)m * (size_t)out_stride), orr((size_t)m * (size_t)out_stride);
    ok &= cudaMemcpy(om.data(), dout_m, bytes_out, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(orr.data(), dout_r, bytes_out, cudaMemcpyDeviceToHost) == cudaSuccess;
    M32_CHECK(ok, "[%s] cudaMemcpy D2H failed", tag);

    size_t at = 0;
    const bool same = m32_bits_equal(om, orr, &at);
    if (!same) {
        const int r = out_stride > 0 ? (int)(at / (size_t)out_stride) : 0;
        const int c = out_stride > 0 ? (int)(at % (size_t)out_stride) : 0;
        uint32_t ba, bb;
        std::memcpy(&ba, &om[at], 4);
        std::memcpy(&bb, &orr[at], 4);
        printf("    FAIL [%s] bit diff at r=%d c=%d: mrows_f32 0x%08x (%g)  M=1 0x%08x (%g)\n", tag,
               r, c, ba, (double)om[at], bb, (double)orr[at]);
        ++g_fails;
    }
    // Coverage: every element of [0, n) of every row must have been written by
    // BOTH sides (the sentinel is NaN, so "still sentinel" = never written).
    size_t unwritten = 0;
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < out_stride; ++c) {
            uint32_t bm, br;
            std::memcpy(&bm, &om[(size_t)r * out_stride + c], 4);
            std::memcpy(&br, &orr[(size_t)r * out_stride + c], 4);
            if (bm == br && (bm & 0x7FFFFFFFu) > 0x7F800000u) ++unwritten;   // NaN both sides
        }
    M32_CHECK(unwritten == 0, "[%s] %zu element(s) left unwritten by both arms", tag, unwritten);

    if (same && unwritten == 0)
        printf("  [%s] OK  m=%d n=%d k=%d a_stride=%d out_stride=%d bias=%d  (%zu elems bit-identical)\n",
               tag, m, n, k, a_stride, out_stride, (int)use_bias, om.size());

    // ---- arm 6: the OLD path, DIAGNOSTIC ONLY -----------------------------
    // The old wo_b was `m x quant_fp8(rows=1, cols=k, block=32, round_scale=true)`
    // + one `dsv41_gemm_fp8_mrows`. Reproduced here (packing the rows, because
    // `quant_fp8` derives the SOURCE pitch from `cols` — the very trap B6's
    // explicit `a_stride` removes). Its difference from B6 is the fp8 round-trip
    // error B6 deletes: reported, never asserted.
    if (diagnose) {
        uint8_t* dxq = nullptr;
        float* dxsc = nullptr;
        float* dold = nullptr;
        const size_t bytes_q = (size_t)m * (size_t)k;
        const size_t bytes_sc = (size_t)m * (size_t)nb_k * sizeof(float);
        bool ok2 = cudaMalloc((void**)&dxq, bytes_q) == cudaSuccess;
        ok2 &= cudaMalloc((void**)&dxsc, bytes_sc) == cudaSuccess;
        ok2 &= cudaMalloc((void**)&dold, bytes_out) == cudaSuccess;
        if (ok2) {
            for (int r = 0; r < m; ++r) {
                // NOTE: `dsv41_quant_fp8` returns the cudaError_t as an `int`
                // (0 on success) — it is NOT the launchers' 0/2 convention.
                const int qrc = dsv41_quant_fp8(da + (size_t)r * (size_t)a_stride,
                                                dxq + (size_t)r * (size_t)k,
                                                dxsc + (size_t)r * (size_t)nb_k, /*rows=*/1, k,
                                                /*block=*/32, /*round_scale=*/1, nullptr);
                M32_CHECK(qrc == 0, "[%s] quant_fp8 row %d returned %d (%s)", tag, r, qrc,
                          cudaGetErrorString((cudaError_t)qrc));
            }
            cudaError_t e2 = cudaMemcpy(dold, sentinel.data(), bytes_out, cudaMemcpyHostToDevice);
            M32_CHECK(e2 == cudaSuccess, "[%s] old-path sentinel H2D", tag);
            const int rc_old = dsv41_gemm_fp8_mrows(dxq, dxsc, dw, dws,
                                                    use_bias ? db : nullptr, dold, m, n, k,
                                                    out_stride, /*stream=*/nullptr);
            if (rc_old != 0) {
                printf("  [%s] diag: old path declined (%d) — no comparison\n", tag, rc_old);
            } else {
                std::vector<float> oldv((size_t)m * (size_t)out_stride);
                cudaMemcpy(oldv.data(), dold, bytes_out, cudaMemcpyDeviceToHost);
                double worst = 0.0;
                int wr = 0, wc = 0;
                for (int r = 0; r < m; ++r)
                    for (int c = 0; c < n; ++c) {
                        const float a1 = om[(size_t)r * out_stride + c];
                        const float a0 = oldv[(size_t)r * out_stride + c];
                        const double den = std::fabs((double)a0);
                        const double rel =
                            den > 0.0 ? std::fabs((double)a1 - (double)a0) / den : 0.0;
                        if (rel > worst) { worst = rel; wr = r; wc = c; }
                    }
                printf("  [%s] diag: B6 vs old (quant_fp8+mrows) max rel diff %.3e at r=%d c=%d"
                       "  <- the fp8 round trip B6 removes\n", tag, worst, wr, wc);
            }
        } else {
            printf("  [%s] diag: allocation failed — skipped\n", tag);
        }
        cudaFree(dxq); cudaFree(dxsc); cudaFree(dold);
    }

    cudaFree(da); cudaFree(dw); cudaFree(dws); cudaFree(db);
    cudaFree(dout_m); cudaFree(dout_r);
    return same ? 0 : 1;
}

// ------------------------------------------------------------- the declines
int m32_case_declines() {
    std::vector<float> a(4096, 1.0f);
    std::vector<uint8_t> w(4096, 0x38), ws(4096, 0x7Bu);
    float *da = nullptr, *dout = nullptr;
    uint8_t *dw = nullptr, *dws = nullptr;
    if (cudaMalloc((void**)&da, a.size() * 4) != cudaSuccess ||
        cudaMalloc((void**)&dw, w.size()) != cudaSuccess ||
        cudaMalloc((void**)&dws, ws.size()) != cudaSuccess ||
        cudaMalloc((void**)&dout, 64 * 256 * 4) != cudaSuccess) {
        printf("    FAIL [declines] cudaMalloc\n");
        ++g_fails;
        return 1;
    }
    cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dws, ws.data(), ws.size(), cudaMemcpyHostToDevice);

    // `a_stride` is the 4th shape field — the ABI extension this kernel adds, so
    // `a_stride < k` (the silently-wrong read) has its own row here.
    struct { const char* what; int m, n, k, a_stride, os; } bad[] = {
        {"m=0",              0, 32, 512, 512,  32},
        {"m=9",              9, 32, 512, 512,  32},
        {"k%32!=0",          2, 32, 500, 512,  32},
        {"a_stride<k",       2, 32, 512, 256,  32},
        {"out_stride<n",     2, 32, 512, 512,  16},
        {"n=0",              2,  0, 512, 512,  32},
    };
    for (const auto& b : bad) {
        const int rc = dsv41_gemm_fp8_mrows_f32(da, dw, dws, nullptr, dout, b.m, b.n, b.k,
                                                b.a_stride, b.os, /*stream=*/nullptr);
        M32_CHECK(rc == 2, "[declines] %s: expected 2, got %d", b.what, rc);
    }
    // A null operand must decline, not fault.
    M32_CHECK(dsv41_gemm_fp8_mrows_f32(nullptr, dw, dws, nullptr, dout, 2, 32, 512, 512, 32,
                                       nullptr) == 2,
              "[declines] null a_f32: expected 2");
    M32_CHECK(dsv41_gemm_fp8_mrows_f32(da, nullptr, dws, nullptr, dout, 2, 32, 512, 512, 32,
                                       nullptr) == 2,
              "[declines] null w: expected 2");
    cudaFree(da); cudaFree(dw); cudaFree(dws); cudaFree(dout);
    printf("  [declines] OK  (m out of range / k%%32 / a_stride<k / out_stride<n / null)\n");
    return 0;
}

// ------------------------------------------------- arm 5: the gate invariance
// `DSV41_GEMV_A32` is read once per process, so invariance is checked by RUNNING
// THE WHOLE SUITE twice (0 and 1) rather than by flipping it in-process. What
// this function contributes is the disclosure: on the f32 domain both arms read
// the SAME word (`s_af[j] = a_f32[j]`, a pure copy), which is why the B6
// launcher deliberately does NOT decline `g_gemv_a32` the way the fp8 sibling
// used to. If the two runs disagree, that premise is what to look at first.
void m32_report_gates() {
    printf("   DSV41_GEMV_A32=%d (arm 5: this suite must be GREEN at both 0 and 1)\n",
           (int)g_gemv_a32);
    printf("   DSV41_GEMV_A32_STAGED=%d / DSV41_GEMV_A32_CPASYNC=%d / _ACT_CPASYNC=%d"
           " (staging only — never enters the parity)\n",
           (int)g_gemv_a32_staged, (int)(getenv("DSV41_GEMV_A32_CPASYNC") != nullptr),
           (int)(getenv("DSV41_GEMV_A32_ACT_CPASYNC") != nullptr));
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    printf("== dsv41 gemm fp8 mrows f32 (B6) acceptance ==\n");
    printf("   DSV41_GEMV_FP8_MODE=%d  DSV41_NO_GEMV_FP8=%d  nwarps(n=5120)=%d\n",
           g_gemv_fp8_mode, (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr),
           dsv41_mrows_warps_for(5120));
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("   device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);
    m32_report_gates();

    // The two production shapes. VERIFY: m = VERIFY_ROWS = 6, n = dim = 5120,
    // k = ol_local = 1024, a_stride = ol_total = 8192 (8x k under TP8) — the
    // exact case the explicit `a_stride` parameter exists for, and the one whose
    // absence was the verify-value-hunt F1/F2 root cause. DRAFT: the same n with
    // k = a_stride = ol_total = 8192, i.e. the compact [bs, ol_total] row.
    g_fails += m32_case("verify/m=6", 6, 5120, 1024, 8192, 5120, false, /*diagnose=*/true);
    g_fails += m32_case("verify/m=5", 5, 5120, 1024, 8192, 5120, false, false);
    g_fails += m32_case("draft/m=5", 5, 5120, 8192, 8192, 5120, false, /*diagnose=*/true);
    // out_stride > n: the verify writes `dim` of an `nh*hd` row.
    g_fails += m32_case("short-out-stride", 3, 512, 1024, 2048, 1024, false, false);
    // a_stride > k with a small n (the wrong-pitch signature on a cheap shape).
    g_fails += m32_case("pitch/m=6", 6, 128, 512, 4096, 128, false, false);
    g_fails += m32_case("m=1", 1, 64, 512, 512, 64, false, false);
    g_fails += m32_case("bias/m=5", 5, 256, 640, 640, 256, true, false);
    g_fails += m32_case_declines();
    if (!quick) {
        // Every dispatch case the launcher's switch can take.
        for (int m = 1; m <= 8; ++m)
            g_fails += m32_case("dispatch/m", m, 96, 640, 640, 96, false, false);
        // A shape where n is NOT a multiple of nwarps (the `active` guard).
        g_fails += m32_case("n%nwarps/m=5", 5, 130, 640, 640, 130, false, false);
        // The smallest k / n that still satisfy the shape gate.
        g_fails += m32_case("tiny/m=5", 5, 32, 64, 64, 32, true, false);
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
