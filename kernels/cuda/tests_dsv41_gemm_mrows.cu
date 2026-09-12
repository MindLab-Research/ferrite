// tests_dsv41_gemm_mrows.cu — the MULTI-ROW acceptance test for
// `dsv41_gemm_fp8_mrows` (kernels/cuda/dsv41_kernels.cu), the weight-stationary
// multi-row form of the M=1 fp8 GEMV (`dsv41_gemm_fp8_mx`'s m == 1 path).
//
// WHY THIS FILE EXISTS. The verify block's iron rule is
//
//     row r of an m-row launch  ==  the m=1 decode of row r, BIT FOR BIT
//
// `dsv41_gemm_fp8_mx` cannot deliver that at m > 1: m == 1 runs the SIMT
// `gemm_fp8_gemv_kernel` (one warp per output row, `shfl_xor` reduction tree)
// while m > 1 runs `gemm_fp8_kernel`'s 16-row TILE — two different programs.
// `dsv41_gemm_fp8_mrows` re-implements the m == 1 program with the weight rows
// shared across the row batch, so this suite checks the only property that
// makes its wiring legal:
//
//   1. BIT-IDENTITY: an m-row call is bit-identical (memcmp over the raw f32
//      bits, `uint32_t` compare — so NaN payloads and ±0 cannot pass by luck) to
//      m single-row `dsv41_gemm_fp8_mx` calls on the same operands. This is
//      exact equality, NOT a tolerance: the m == 1 consume expression, its
//      ascending-kb walk and its shuffle tree are reproduced per row.
//   2. ROW INDEPENDENCE + FULL COVERAGE: every (row, r) element must be written.
//      The buffers are pre-filled with a NaN sentinel and compared WHOLE (with
//      `out_stride > n` the gaps must stay sentinel on both sides), so a kernel
//      that only did row 0 — the failure the old m>1 tile path hides — cannot
//      pass.
//   3. THE DECLINE PATH: m outside 1..=8, k not a multiple of 32, out_stride < n
//      and the mode-0/1 reordering arms must return 2 (the caller's "keep the
//      per-row loop" signal), never a real launch error.
//
// NOT checked here: end-to-end model parity (the Rust wiring is a separate
// change) and launch performance.
//
// ---------------------------------------------------------------------------
// Build (needs nvcc, NO GPU — dsv41_kernels.cu is the only TU):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math \
//        -std=c++17 -o /tmp/t_gemm_mrows kernels/cuda/tests_dsv41_gemm_mrows.cu
// Run (needs ONE free GPU; peak allocation is a few MB):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_gemm_mrows            # full suite
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_gemm_mrows --quick    # small shapes only
//
// ENV THAT CHANGES WHAT IS COVERED (read once per process, before main):
//   DSV41_GEMV_FP8_MODE  must be >= 3 for any arm to run (0/1 reorder a lane's
//                        elements and the launcher declines them — that decline
//                        is itself checked). The default is 4.
//   DSV41_NO_GEMV_FP8    set => the m=1 reference itself takes the M-tile MMA
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

#define MR_CHECK(expr, fmt, ...)                                                       \
    do {                                                                               \
        if (!(expr)) {                                                                 \
            printf("    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);     \
            ++g_fails;                                                                 \
        }                                                                              \
    } while (0)

uint32_t g_rng = 20260912u;
uint32_t mr_xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
int mr_irand(int n) { return (int)(mr_xr() % (uint32_t)n); }

// e4m3 byte that is NOT a NaN. exp == 15 && mantissa == 7 is NaN (0x7F / 0xFF),
// and NaN breaks a bit-for-bit compare (NaN != NaN under memcmp of the bits is
// fine, but a kernel that reordered could turn a NaN into a different payload;
// excluding the code keeps the suite's failures unambiguous).
uint8_t mr_e4m3_byte() {
    uint8_t b;
    do {
        b = (uint8_t)(mr_xr() & 0xFFu);
    } while ((b & 0x7Fu) == 0x7Fu);
    return b;
}

// A sane ue8m0 exponent: 2^(124..132) keeps the products inside f32 range.
uint8_t mr_ue8m0_byte() { return (uint8_t)(124 + mr_irand(9)); }

// A power-of-two f32 activation scale, the shape `quant_fp8(round_scale=true)`
// emits (fast_round_scale's output).
float mr_act_scale() { return std::ldexp(1.0f, mr_irand(13) - 6); }

// Fill a buffer with a NaN sentinel so "never written" is distinguishable from
// "written with a small value" (the coverage half of arm 2).
void mr_fill_sentinel(std::vector<float>& v) {
    const float nan = __builtin_nanf("");
    uint32_t bits;
    std::memcpy(&bits, &nan, 4);
    for (size_t i = 0; i < v.size(); ++i) std::memcpy(&v[i], &bits, 4);
}

// Bit-for-bit compare (the RAW f32 bits, so -0.0 vs +0.0 and NaN payloads are
// differences — "bit-identical" is the contract, not "approximately equal").
bool mr_bits_equal(const std::vector<float>& x, const std::vector<float>& y, size_t* at) {
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
// `out_stride >= n` is the wq_b shape (this rank writes nlh*head_dim of an
// nh*head_dim row); `out_stride == n` is every other projection.
int mr_case(const char* tag, int m, int n, int k, int out_stride, bool use_bias) {
    const int nb_k = k / 32;
    MR_CHECK(k % 32 == 0 && n > 0, "[%s] bad shape", tag);

    std::vector<uint8_t> ha((size_t)m * (size_t)k);
    std::vector<float> hasc((size_t)m * (size_t)nb_k);
    std::vector<uint8_t> hw((size_t)n * (size_t)k);
    // w_scale is indexed [row >> 5][kb] for every row < n, so the tail row block
    // needs its own row; two padding rows keep an `n % 32 != 0` shape readable.
    std::vector<uint8_t> hws((size_t)(n / 32 + 2) * (size_t)nb_k, 0);
    std::vector<float> hb(use_bias ? (size_t)n : 1);
    for (auto& b : ha) b = mr_e4m3_byte();
    for (auto& s : hasc) s = mr_act_scale();
    for (auto& b : hw) b = mr_e4m3_byte();
    for (auto& b : hws) b = mr_ue8m0_byte();
    for (int i = 0; i < n && use_bias; ++i) hb[(size_t)i] = (float)(mr_irand(2001) - 1000) * 0.001f;

    uint8_t *da = nullptr, *dw = nullptr, *dws = nullptr;
    float *dasc = nullptr, *db = nullptr, *dout_m = nullptr, *dout_r = nullptr;
    const size_t bytes_a = (size_t)m * (size_t)k;
    const size_t bytes_w = (size_t)n * (size_t)k;
    const size_t bytes_ws = hws.size();
    const size_t bytes_asc = (size_t)m * (size_t)nb_k * sizeof(float);
    const size_t bytes_out = (size_t)m * (size_t)out_stride * sizeof(float);
    bool ok = true;
    ok &= cudaMalloc((void**)&da, bytes_a) == cudaSuccess;
    ok &= cudaMalloc((void**)&dw, bytes_w) == cudaSuccess;
    ok &= cudaMalloc((void**)&dws, bytes_ws) == cudaSuccess;
    ok &= cudaMalloc((void**)&dasc, bytes_asc) == cudaSuccess;
    ok &= cudaMalloc((void**)&db, use_bias ? (size_t)n * sizeof(float) : 4u) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_m, bytes_out) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_r, bytes_out) == cudaSuccess;
    MR_CHECK(ok, "[%s] cudaMalloc failed", tag);
    if (!ok) return 1;

    std::vector<float> sentinel((size_t)m * (size_t)out_stride);
    mr_fill_sentinel(sentinel);
    ok &= cudaMemcpy(da, ha.data(), bytes_a, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dw, hw.data(), bytes_w, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dws, hws.data(), bytes_ws, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dasc, hasc.data(), bytes_asc, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(db, hb.data(), use_bias ? (size_t)n * sizeof(float) : 4u,
                     cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dout_m, sentinel.data(), bytes_out, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dout_r, sentinel.data(), bytes_out, cudaMemcpyHostToDevice) == cudaSuccess;
    MR_CHECK(ok, "[%s] cudaMemcpy H2D failed", tag);

    // ---- the m-row launch -------------------------------------------------
    const int rc = dsv41_gemm_fp8_mrows(da, dasc, dw, dws, use_bias ? db : nullptr, dout_m, m, n,
                                        k, out_stride, /*stream=*/nullptr);
    if (rc == 2) {
        printf("  [%s] SKIP: dsv41_gemm_fp8_mrows declined (mode=%d, no_gemv=%d)\n", tag,
               g_gemv_fp8_mode, (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr));
        ++g_skips;
        cudaFree(da); cudaFree(dw); cudaFree(dws); cudaFree(dasc); cudaFree(db);
        cudaFree(dout_m); cudaFree(dout_r);
        return 0;
    }
    MR_CHECK(rc == 0, "[%s] mrows launch returned %d (%s)", tag, rc, cudaGetErrorString((cudaError_t)rc));

    // ---- the m single-row references -------------------------------------
    for (int r = 0; r < m; ++r) {
        const int rc1 = dsv41_gemm_fp8_mx(da + (size_t)r * k, dasc + (size_t)r * nb_k, dw, dws,
                                          use_bias ? db : nullptr,
                                          dout_r + (size_t)r * out_stride, /*m=*/1, n, k,
                                          /*stream=*/nullptr);
        MR_CHECK(rc1 == 0, "[%s] reference row %d returned %d (%s)", tag, r, rc1,
                 cudaGetErrorString((cudaError_t)rc1));
    }
    const cudaError_t se = cudaDeviceSynchronize();
    MR_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> om((size_t)m * (size_t)out_stride), orr((size_t)m * (size_t)out_stride);
    ok &= cudaMemcpy(om.data(), dout_m, bytes_out, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(orr.data(), dout_r, bytes_out, cudaMemcpyDeviceToHost) == cudaSuccess;
    MR_CHECK(ok, "[%s] cudaMemcpy D2H failed", tag);

    size_t at = 0;
    const bool same = mr_bits_equal(om, orr, &at);
    if (!same) {
        const int r = out_stride > 0 ? (int)(at / (size_t)out_stride) : 0;
        const int c = out_stride > 0 ? (int)(at % (size_t)out_stride) : 0;
        uint32_t ba, bb;
        std::memcpy(&ba, &om[at], 4);
        std::memcpy(&bb, &orr[at], 4);
        printf("    FAIL [%s] bit diff at r=%d c=%d: mrows 0x%08x (%g)  m=1 0x%08x (%g)\n", tag, r, c,
               ba, (double)om[at], bb, (double)orr[at]);
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
    MR_CHECK(unwritten == 0, "[%s] %zu element(s) left unwritten by both arms", tag, unwritten);

    if (same && unwritten == 0)
        printf("  [%s] OK  m=%d n=%d k=%d out_stride=%d bias=%d  (%zu elems bit-identical)\n", tag, m, n,
               k, out_stride, (int)use_bias, om.size());

    cudaFree(da); cudaFree(dw); cudaFree(dws); cudaFree(dasc); cudaFree(db);
    cudaFree(dout_m); cudaFree(dout_r);
    return same ? 0 : 1;
}

// ------------------------------------------------------------- the declines
int mr_case_declines() {
    std::vector<uint8_t> a(4096, 0x38), w(4096, 0x38), ws(4096, 0x7Bu);
    std::vector<float> asc(1024, 1.0f);
    uint8_t *da = nullptr, *dw = nullptr, *dws = nullptr;
    float *dasc = nullptr, *dout = nullptr;
    if (cudaMalloc((void**)&da, a.size()) != cudaSuccess ||
        cudaMalloc((void**)&dw, w.size()) != cudaSuccess ||
        cudaMalloc((void**)&dws, ws.size()) != cudaSuccess ||
        cudaMalloc((void**)&dasc, asc.size() * 4) != cudaSuccess ||
        cudaMalloc((void**)&dout, 64 * 256 * 4) != cudaSuccess) {
        printf("    FAIL [declines] cudaMalloc\n");
        ++g_fails;
        return 1;
    }
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dws, ws.data(), ws.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dasc, asc.data(), asc.size() * 4, cudaMemcpyHostToDevice);

    struct { const char* what; int m, n, k, os; } bad[] = {
        {"m=0",      0, 32, 512, 32},
        {"m=9",      9, 32, 512, 32},
        {"k%32!=0",  2, 32, 500, 32},
        {"stride<n", 2, 32, 512, 16},
        {"n=0",      2,  0, 512, 32},
    };
    for (const auto& b : bad) {
        const int rc = dsv41_gemm_fp8_mrows(da, dasc, dw, dws, nullptr, dout, b.m, b.n, b.k, b.os,
                                            /*stream=*/nullptr);
        MR_CHECK(rc == 2, "[declines] %s: expected 2, got %d", b.what, rc);
    }
    // A null operand must decline, not fault.
    MR_CHECK(dsv41_gemm_fp8_mrows(nullptr, dasc, dw, dws, nullptr, dout, 2, 32, 512, 32, nullptr) == 2,
             "[declines] null a: expected 2");
    cudaFree(da); cudaFree(dw); cudaFree(dws); cudaFree(dasc); cudaFree(dout);
    printf("  [declines] OK  (m out of range / k%%32 / out_stride<n / null operand)\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    printf("== dsv41 gemm fp8 multi-row (mrows) acceptance ==\n");
    printf("   DSV41_GEMV_FP8_MODE=%d  DSV41_NO_GEMV_FP8=%d  nwarps(n=1280)=%d  nwarps(n=4096)=%d\n",
           g_gemv_fp8_mode, (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr),
           dsv41_gemv_warps_for(1280), dsv41_gemv_warps_for(4096));
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("   device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

    // The production verify shapes (m = VERIFY_ROWS = 6, and m = 5 which is the
    // block the plan's launch ledger counts).
    g_fails += mr_case("wq_a/m=5", 5, 1280, 5120, 1280, false);
    g_fails += mr_case("wq_a/m=6+bias", 6, 1280, 5120, 1280, true);
    g_fails += mr_case("wkv/m=5", 5, 512, 5120, 512, false);
    // wq_b: out_stride != n (nlh*head_dim written inside an nh*head_dim row).
    g_fails += mr_case("wq_b/stride/m=6", 6, 2048, 1280, 4096, false);
    g_fails += mr_case("wo_b/m=5", 5, 5120, 1024, 5120, false);
    g_fails += mr_case("m=1", 1, 64, 512, 64, false);
    g_fails += mr_case_declines();
    if (!quick) {
        // Every dispatch case the launcher's switch can take.
        for (int m = 1; m <= 8; ++m)
            g_fails += mr_case("dispatch/m", m, 96, 640, 96, false);
        // A shape where n is NOT a multiple of nwarps (the `active` guard).
        g_fails += mr_case("n%nwarps/m=5", 5, 130, 640, 130, false);
        // The smallest k / n that still satisfy the shape gate.
        g_fails += mr_case("tiny/m=5", 5, 32, 64, 32, true);
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
