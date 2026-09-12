// tests_dsv41_head_mrows.cu — the bit-parity acceptance test for
// `dsv41_head_gemv_bf16_mrows` (kernels/cuda/dsv41_glue.cu), the weight-stationary
// multi-row form of the verify head's GEMV.
//
// WHY THIS FILE EXISTS. The verify block's iron rule is
//
//     row r of the m-row launch  ==  the PRODUCTION single-row head GEMV of row r
//
// and the production single-row program is NOT v1's scalar `gemv_bf16_kernel`:
// it is `gemv_bf16_nt_kernel`'s per-token body at WPR == 1 (ferrite_kernels.cu),
// the body `gemv_bf16_v2_kernel` also runs — `gv2_wpr` and the nt launcher both
// return WPR = 1 for out_f >= 16384, and the head is out_f = 129280. That body
// walks `for (k = k0 + lane*8; k + 7 < k1; k += 32*8)` with a uint4 weight load,
// four `__bfloat1622float2` decodes and two 4-term FMA groups per lane-step, and
// reduces with `__shfl_down_sync(off = 16,8,4,2,1)`. The multi-row kernel that
// the verify folds its rows into must reproduce that program per row; the
// earlier version reproduced v1's instead, and the ~1e-3 rounding gap was enough
// to flip near-tie argmaxes (`verify_out[0] == next` on 33% of rows = the
// repeated text). This suite checks the only property that makes the fold
// legal:
//
//   1. BIT-IDENTITY vs the production single-row launch: an m-row call is
//      bit-identical (raw f32 bits, uint32_t compare — ±0 and NaN payloads count
//      as differences) to m single-row `ferrite_gemv_bf16_v2(..., nrows = 1)`
//      calls, issued with the SAME argument order device.rs:2933 uses.
//   2. BIT-IDENTITY vs the verify chain's batched form: the same m-row call is
//      bit-identical to ONE `ferrite_gemv_bf16_nt(..., nrows = m)` — the kernel
//      the parity is named for (m >= 2; the nt entry has no nrows == 1 arm).
//   3. ROW INDEPENDENCE + FULL COVERAGE: every (row, col) of [m, n) must be
//      written. The output buffers are pre-filled with a qNaN sentinel and
//      compared WHOLE, so a kernel that only did row 0 cannot pass.
//   4. THE DECLINE PATH: m outside 1..=8 and k % 8 != 0 must return
//      cudaErrorInvalidValue and leave `out` untouched — the Rust caller
//      (`device.rs` `head_gemv_bf16_mrows`) declines on the same conditions, so
//      the verify keeps its per-row loop instead of faulting on a misaligned
//      uint4 load or erroring out.
//
// The `--domain-probe` arm is evidence, not a gate: it runs one shape BELOW the
// WPR == 1 boundary (n in [4096, 16384) => WPR = 2) and REPORTS the mismatch
// instead of failing, documenting that the parity domain really is the boundary
// the header claims.
//
// NOT checked here: end-to-end model parity (the Rust wiring is a separate
// change) and launch performance.
//
// ---------------------------------------------------------------------------
// Build (needs nvcc, NO GPU — the test TU #includes dsv41_glue.cu, so
// ferrite_kernels.cu is a SECOND input file, exactly how the production .so is
// linked):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math \
//        -std=c++17 -o /tmp/t_head_mrows kernels/cuda/tests_dsv41_head_mrows.cu \
//        kernels/cuda/ferrite_kernels.cu
// Run (needs ONE free GPU; --quick keeps every arm at the WPR == 1 boundary,
// 16384x5120 bf16 = 160 MB, while the full run also builds the head's own
// 129280x5120 = 1.26 GB):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_head_mrows [--quick] [--domain-probe]
//   bash scripts/verify_mrows.sh --test head [--quick]
#include "dsv41_glue.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The reference entries live in the OTHER TU (ferrite_kernels.cu) — the test
// links it as a second input file rather than #including it, so the two kernel
// TUs see exactly the flags and the split the .so gives them.
extern "C" cudaError_t ferrite_gemv_bf16_v2(const float* x, const void* w, const float* bias,
                                            float* out, int in_f, int out_f, int nrows,
                                            cudaStream_t s);
extern "C" cudaError_t ferrite_gemv_bf16_nt(const float* x, const void* w, const float* bias,
                                            float* out, int in_f, int out_f, int nrows,
                                            cudaStream_t s);

namespace {

// ---------------------------------------------------------------- test utils
int g_fails = 0;
int g_skips = 0;

#define HM_CHECK(expr, fmt, ...)                                                      \
    do {                                                                              \
        if (!(expr)) {                                                                \
            printf("    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);    \
            ++g_fails;                                                                \
        }                                                                             \
    } while (0)

uint32_t g_rng = 20260912u;
uint32_t hm_xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
int hm_irand(int n) { return (int)(hm_xr() % (uint32_t)n); }

// A finite, non-NaN bf16: exponent 122..130 (2^-5 .. 2^3) with a random mantissa.
// NaN/Inf are excluded on purpose — the compare is over raw bits and the
// sentinel is a NaN, so a NaN operand would blur "written by the kernel" into
// "still sentinel" and make a failure ambiguous.
__nv_bfloat16 hm_bf16() {
    const uint16_t e = (uint16_t)(122 + (hm_xr() % 9));
    const uint16_t b = (uint16_t)((hm_xr() & 0x8000u) | (e << 7) | (hm_xr() & 0x7Fu));
    __nv_bfloat16 v;
    std::memcpy(&v, &b, 2);
    return v;
}

const uint32_t HM_NAN = 0x7FC00000u;   // quiet NaN, the "never written" sentinel
void hm_fill_nan(std::vector<float>& v) {
    for (auto& x : v) std::memcpy(&x, &HM_NAN, 4);
}

// Bit-for-bit compare over the RAW f32 bits ("bit-identical" is the contract,
// not "approximately equal"): ±0 and NaN payloads are differences.
bool hm_bits_equal(const std::vector<float>& x, const std::vector<float>& y, size_t* at) {
    if (x.size() != y.size()) { *at = 0; return false; }
    for (size_t i = 0; i < x.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &x[i], 4);
        std::memcpy(&b, &y[i], 4);
        if (a != b) { *at = i; return false; }
    }
    return true;
}

uint32_t hm_bits_of(const std::vector<float>& v, size_t i) {
    uint32_t b;
    std::memcpy(&b, &v[i], 4);
    return b;
}

// THE DOMAIN GATE, mirrored from `gv2_wpr` (ferrite_kernels.cu) and the nt
// launcher's copy of it. The multi-row kernel implements the WPR == 1 program;
// for out_f < 16384 the production GEMV K-splits a row across WPR warps and
// folds the partials in shared memory, which this kernel does not reproduce —
// an arm outside the boundary would be checking a program that is not the
// reference.
int hm_wpr(int out_f) {
    return out_f >= 16384 ? 1 : (out_f >= 4096 ? 2 : (out_f >= 1024 ? 4 : 8));
}

// -------------------------------------------------------------- one shape arm
// `probe` = report the v2/nt divergence below the WPR == 1 boundary as INFO
// (never a failure): that arm is evidence about the domain, not a contract.
int hm_case(const char* tag, int m, int n, int k, bool probe) {
    HM_CHECK(k % 8 == 0 && n > 0 && m >= 1 && m <= 8, "[%s] bad shape m=%d n=%d k=%d", tag, m, n, k);
    const int wpr = hm_wpr(n);
    printf("  [%s] m=%d n=%d k=%d  (launcher WPR for n: %d)\n", tag, m, n, k, wpr);

    const size_t nw = (size_t)n * (size_t)k;             // bf16 weight elements
    const size_t nx = (size_t)m * (size_t)k;             // f32 activation elements
    const size_t no = (size_t)m * (size_t)n;             // f32 logits elements
    const size_t bytes_w = nw * 2;
    const size_t bytes_x = nx * sizeof(float);
    const size_t bytes_o = no * sizeof(float);

    // Do not fight a co-tenant serve for 1.26 GB: report SKIP, not FAIL.
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b < bytes_w + 3 * bytes_o + bytes_x + (64u << 20)) {
        printf("    SKIP [%s] needs ~%.0f MB free, %.0f MB available\n", tag,
               (double)(bytes_w + 3 * bytes_o + bytes_x) / 1048576.0, (double)free_b / 1048576.0);
        ++g_skips;
        return 0;
    }

    std::vector<__nv_bfloat16> hw(nw);
    std::vector<float> hx(nx);
    for (size_t i = 0; i < nw; ++i) hw[i] = hm_bf16();
    for (size_t i = 0; i < nx; ++i) hx[i] = (float)(hm_irand(2001) - 1000) * 0.001f;

    __nv_bfloat16* dw = nullptr;
    float *dx = nullptr, *dout_m = nullptr, *dout_r = nullptr, *dout_nt = nullptr;
    bool ok = cudaMalloc((void**)&dw, bytes_w) == cudaSuccess;
    ok &= cudaMalloc((void**)&dx, bytes_x) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_m, bytes_o) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_r, bytes_o) == cudaSuccess;
    ok &= cudaMalloc((void**)&dout_nt, bytes_o) == cudaSuccess;
    HM_CHECK(ok, "[%s] cudaMalloc failed (%zu B weight)", tag, bytes_w);
    if (!ok) {
        cudaFree(dw); cudaFree(dx); cudaFree(dout_m); cudaFree(dout_r); cudaFree(dout_nt);
        return 1;
    }

    std::vector<float> sentinel(no);
    hm_fill_nan(sentinel);
    ok &= cudaMemcpy(dw, hw.data(), bytes_w, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dx, hx.data(), bytes_x, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dout_m, sentinel.data(), bytes_o, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dout_r, sentinel.data(), bytes_o, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(dout_nt, sentinel.data(), bytes_o, cudaMemcpyHostToDevice) == cudaSuccess;
    HM_CHECK(ok, "[%s] cudaMemcpy H2D failed", tag);

    // ---- the m-row launch under test --------------------------------------
    const int rc = dsv41_head_gemv_bf16_mrows(dw, dx, dout_m, m, n, k, /*stream=*/nullptr);
    HM_CHECK(rc == 0, "[%s] mrows launch returned %d (%s)", tag, rc,
             cudaGetErrorString((cudaError_t)rc));

    // ---- the m single-row production launches (v2, nrows = 1) --------------
    // Argument order = device.rs:2933's exactly: (x, w, bias, out, in_f=k,
    // out_f=n, nrows=1). Row r reads x + r*k and writes out + r*n.
    for (int r = 0; r < m; ++r) {
        const cudaError_t rc1 = ferrite_gemv_bf16_v2(dx + (size_t)r * k, dw, nullptr,
                                                     dout_r + (size_t)r * n, k, n, 1,
                                                     /*stream=*/nullptr);
        HM_CHECK(rc1 == cudaSuccess, "[%s] single-row v2 row %d returned %s", tag, r,
                 cudaGetErrorString(rc1));
    }

    // ---- the verify chain's batched form (nt, nrows = m) ------------------
    // The nt entry has no nrows == 1 arm (its switch starts at 2); m == 1 is
    // covered by the v2 arm above, which is the same body.
    bool nt_arm = (m >= 2 && m <= 8);
    if (nt_arm) {
        const cudaError_t rcn = ferrite_gemv_bf16_nt(dx, dw, nullptr, dout_nt, k, n, m,
                                                     /*stream=*/nullptr);
        HM_CHECK(rcn == cudaSuccess, "[%s] nt(nrows=%d) returned %s", tag, m,
                 cudaGetErrorString(rcn));
        nt_arm = (rcn == cudaSuccess);
    } else {
        printf("    INFO [%s] nt arm skipped (nrows=%d: the nt switch has no such case)\n", tag, m);
    }

    const cudaError_t se = cudaDeviceSynchronize();
    HM_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> om(no), orr(no), ont(no);
    ok &= cudaMemcpy(om.data(), dout_m, bytes_o, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(orr.data(), dout_r, bytes_o, cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(ont.data(), dout_nt, bytes_o, cudaMemcpyDeviceToHost) == cudaSuccess;
    HM_CHECK(ok, "[%s] cudaMemcpy D2H failed", tag);

    size_t at = 0;
    const bool same_v2 = hm_bits_equal(om, orr, &at);
    const bool same_nt = nt_arm ? hm_bits_equal(om, ont, &at) : true;
    const bool same = same_v2 && same_nt;

    // Coverage: every element of [0, n) of every row must have been WRITTEN by
    // the mrows arm (the sentinel is a NaN, so "still sentinel" = never written).
    size_t unwritten = 0;
    for (size_t i = 0; i < no; ++i) {
        if ((hm_bits_of(om, i) & 0x7FFFFFFFu) > 0x7F800000u) ++unwritten;
    }
    HM_CHECK(unwritten == 0, "[%s] %zu element(s) left unwritten by the mrows arm", tag, unwritten);

    if (probe) {
        // INFO, never FAIL: this arm is below the WPR == 1 boundary, so a
        // divergence is the documented domain limit, and agreement would mean
        // the domain note is too conservative.
        size_t pa = 0;
        const bool ps = hm_bits_equal(om, orr, &pa);
        printf("    INFO [%s] WPR=%d domain probe: mrows vs single-row v2 %s\n", tag, wpr,
               ps ? "AGREE (boundary is wider than documented)" : "DIFFER (as documented)");
        if (!ps) {
            const int r = (int)(pa / (size_t)n), c = (int)(pa % (size_t)n);
            printf("         first diff at r=%d c=%d: mrows 0x%08x (%g)  v2 0x%08x (%g)\n", r, c,
                   hm_bits_of(om, pa), (double)om[pa], hm_bits_of(orr, pa), (double)orr[pa]);
        }
        // WHICH single-row program is the mrows kernel actually equal to? The
        // head's own per-row fallback — `dsv41_gemv_bf16` (the FOLD=0 path in
        // chain_dev/dspark_dev) — is v1's scalar kernel: `gemv_bf16_v2_wanted`
        // only diverts n < 2048 and the head's n is 129280, so the fallback
        // stays on v1 while the batched decode / verify chain runs v2/nt at
        // WPR == 1. Measuring all three here is what tells those two apart
        // without a serve run: mrows is built to equal the v2/nt program.
        float* dout_v1 = nullptr;
        if (cudaMalloc((void**)&dout_v1, bytes_o) == cudaSuccess) {
            cudaMemcpy(dout_v1, sentinel.data(), bytes_o, cudaMemcpyHostToDevice);
            for (int r = 0; r < m; ++r)
                (void)dsv41_gemv_bf16(dw, dx + (size_t)r * k, dout_v1 + (size_t)r * n, n, k,
                                      /*stream=*/nullptr);
            (void)cudaDeviceSynchronize();
            std::vector<float> ov1(no);
            cudaMemcpy(ov1.data(), dout_v1, bytes_o, cudaMemcpyDeviceToHost);
            size_t q1 = 0, q2 = 0;
            const bool m_eq_v1 = hm_bits_equal(om, ov1, &q1);
            const bool v2_eq_v1 = hm_bits_equal(orr, ov1, &q2);
            printf("    INFO [%s] cross-check vs the per-row FOLD=0 fallback: "
                   "mrows vs dsv41_gemv_bf16(v1) %s ; v2(nrows=1) vs v1 %s\n", tag,
                   m_eq_v1 ? "AGREE" : "DIFFER", v2_eq_v1 ? "AGREE" : "DIFFER");
            if (!m_eq_v1) {
                const int r1 = (int)(q1 / (size_t)n), c1 = (int)(q1 % (size_t)n);
                printf("         mrows-vs-v1 first diff at r=%d c=%d: 0x%08x (%g) vs 0x%08x (%g)\n",
                       r1, c1, hm_bits_of(om, q1), (double)om[q1], hm_bits_of(ov1, q1),
                       (double)ov1[q1]);
            }
            cudaFree(dout_v1);
        }
        ++g_skips;
    } else {
        if (!same) {
            const int r = (int)(at / (size_t)n), c = (int)(at % (size_t)n);
            printf("    FAIL [%s] bit diff at r=%d c=%d: mrows 0x%08x (%g)  ref 0x%08x (%g)"
                   "  [%s]\n", tag, r, c, hm_bits_of(om, at), (double)om[at],
                   hm_bits_of(same_v2 ? ont : orr, at), (double)(same_v2 ? ont : orr)[at],
                   same_v2 ? "vs nt(nrows=m)" : "vs v2(nrows=1)");
            ++g_fails;
        }
        if (same && unwritten == 0) {
            printf("    OK  [%s] m=%d: %zu logits bit-identical to %d single-row v2 launch(es)%s\n",
                   tag, m, no, m, nt_arm ? " AND to the nt(nrows=m) launch" : "");
        }
    }

    cudaFree(dw); cudaFree(dx); cudaFree(dout_m); cudaFree(dout_r); cudaFree(dout_nt);
    return same ? 0 : 1;
}

// ------------------------------------------------------------- the declines
// m outside 1..=8, k % 8 != 0: the C entry refuses WITHOUT launching, so the
// output keeps its sentinel. (The Rust wrapper declines on the same conditions
// and returns Ok(false); the verify then keeps its per-row loop.)
int hm_declines() {
    const int n = 64, k = 5128;                  // k % 8 == 0, 129280-free
    const int kb = 5121;                         // k % 8 != 0
    const size_t nw = (size_t)n * (size_t)k;
    const size_t nwb = (size_t)n * (size_t)kb;
    const size_t no = 8 * (size_t)n;
    std::vector<__nv_bfloat16> hw(nwb > nw ? nwb : nw);
    for (auto& v : hw) v = hm_bf16();
    std::vector<float> hx((size_t)8 * (size_t)kb, 0.5f);
    std::vector<float> sentinel(no);
    hm_fill_nan(sentinel);

    __nv_bfloat16* dw = nullptr;
    float *dx = nullptr, *dout = nullptr;
    if (cudaMalloc((void**)&dw, hw.size() * 2) != cudaSuccess ||
        cudaMalloc((void**)&dx, hx.size() * 4) != cudaSuccess ||
        cudaMalloc((void**)&dout, no * 4) != cudaSuccess) {
        printf("    FAIL [declines] cudaMalloc\n");
        ++g_fails;
        return 1;
    }
    cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);

    struct { const char* what; int m, n, k; int want; } bad[] = {
        {"m=0 (no-op)", 0,  n,  k,  (int)cudaSuccess},
        {"m=9",         9,  n,  k,  (int)cudaErrorInvalidValue},
        {"k%8!=0",      5,  n,  kb, (int)cudaErrorInvalidValue},
        {"n=0 (no-op)", 5,  0,  k,  (int)cudaSuccess},
    };
    for (const auto& b : bad) {
        cudaMemcpy(dout, sentinel.data(), no * 4, cudaMemcpyHostToDevice);
        const int rc = dsv41_head_gemv_bf16_mrows(dw, dx, dout, b.m, b.n, b.k, /*stream=*/nullptr);
        HM_CHECK(rc == b.want, "[declines] %s: expected %d, got %d (%s)", b.what, b.want, rc,
                 cudaGetErrorString((cudaError_t)rc));
        std::vector<float> after(no);
        cudaMemcpy(after.data(), dout, no * 4, cudaMemcpyDeviceToHost);
        size_t touched = 0;
        for (size_t i = 0; i < no; ++i) {
            if (hm_bits_of(after, i) != HM_NAN) ++touched;
        }
        HM_CHECK(touched == 0, "[declines] %s: wrote %zu element(s)", b.what, touched);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dout);
    printf("  [declines] OK  (m out of range / k%%8 != 0 refused, output untouched)\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false, probe = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        if (std::strcmp(argv[i], "--domain-probe") == 0) probe = true;
    }
    printf("== dsv41 head gemv bf16 multi-row (mrows) bit-parity acceptance ==\n");
    printf("   the reference program: gemv_bf16_nt_kernel / gemv_bf16_v2_kernel at WPR == 1\n");
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("   device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

    // The WPR == 1 boundary is where the parity starts (see hm_wpr): 16384 is
    // the smallest such n, and the head (129280) is the shape that matters.
    g_fails += hm_case("boundary/n=16384/m=1", 1, 16384, 5120, false);
    g_fails += hm_case("boundary/n=16384/m=5", 5, 16384, 5120, false);
    // n % 8 != 0 (the grid's last block has idle warps) and a second k (the
    // vector loop's iteration count), both still inside WPR == 1.
    g_fails += hm_case("odd-n/n=16391/m=6", 6, 16391, 5120, false);
    g_fails += hm_case("k=4096/m=5", 5, 16384, 4096, false);
    g_fails += hm_case("k=8/m=5", 5, 16384, 8, false);
    g_fails += hm_declines();
    if (!quick) {
        // The production verify head, at the verbatim shape (m = 5 is the block
        // the perf plan's launch ledger counts; the verify's VERIFY_ROWS is 6).
        g_fails += hm_case("HEAD/n=129280/m=5", 5, 129280, 5120, false);
        g_fails += hm_case("HEAD/n=129280/m=6", 6, 129280, 5120, false);
        g_fails += hm_case("HEAD/n=129280/m=8", 8, 129280, 5120, false);
    }
    if (probe) {
        // Below the boundary the production GEMV K-splits: report, do not fail.
        // The head-shape probe also cross-checks the FOLD=0 fallback program
        // (`dsv41_gemv_bf16`, v1) against mrows and against v2(nrows=1) — that
        // pair of INFO lines is what separates "mrows matches production" from
        // "mrows matches the slow fallback".
        (void)hm_case("probe/HEAD/n=129280/m=5", 5, 129280, 5120, true);
        (void)hm_case("probe/n=8192/m=5", 5, 8192, 5120, true);
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
