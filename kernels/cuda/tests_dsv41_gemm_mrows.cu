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
//
// THE TWO L4/L5 GATES THIS SUITE NOW COVERS (2026-09-12, plan
// docs/agent/l4l5-next-batch-implementation-plan.md §6 N1 "parity 扩轴"):
//
//   DSV41_MROWS_ACT_CPASYNC (1b) — SWEPT IN-PROCESS. The activation staging
//     loop is a pure copy (`s_a[r*k+i] = ar[i]`), so its cp.async16 form must be
//     BIT-IDENTICAL to the scalar form at every shape — including a shape where
//     the runtime `dsv41_f4_ok` guard declines it and the kernel keeps the scalar
//     loop anyway. `mrows_act_cpasync_host()` reads `getenv` on EVERY launch (NOT
//     at load time, unlike fold_r), so `mr_case_cp16_axis` runs both arms inside
//     one process and checks each against the SAME m single-row references:
//       arm "<tag>/cp16=0"  DSV41_MROWS_ACT_CPASYNC unset (the shipped default)
//       arm "<tag>/cp16=1"  DSV41_MROWS_ACT_CPASYNC=1
//     arm-on == arm-off then follows from both == reference, which is the
//     stronger form.
//
//   DSV41_MROWS_FOLD_R (1a) — ONE PROCESS PER ARM. `g_mrows_fold_r` is a
//     FILE-SCOPE const initialised from `getenv` at LOAD time, so an in-process
//     sweep is impossible by construction. The axis is therefore exercised by
//     re-running the WHOLE suite once per value (the mr_case arms then run on the
//     folded grid — same expected output, since `acc[q]` is an independent chain
//     and nothing is ever combined across activation rows):
//       bash: for fr in 1 2 3 6; do DSV41_MROWS_FOLD_R=$fr /tmp/t_gemm_mrows; done
//     `mr_fold_r_contract` prints the resolved (fold_r, ng) per production shape
//     and pins the 2026-09-12 fix — with the gate unset/`0`/`auto` the resolution
//     MUST be the identity (`fold_r = m`, `ng = 1`), the program that keeps the
//     6x weight-re-staging regression from coming back unnoticed.
//     NOTE FOR THE LAZY PATH: at m = 1 the identity is also the ONLY reachable
//     program (`fold_r` clamps into [1, m] = {1} => ng = 1 => grid = nt), i.e.
//     1a is a PROVABLE no-op on the lazy/per-row verify — see the L4/L5 design's
//     m=1 row in `docs/agent/lazy-l45-next-ab-design.md`.
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
    // BOTH sides (the sentinel is NaN, so "still sentinel" = never written), and
    // every element of [n, out_stride) — the GAP a narrowed projection
    // (`out_stride > n`, the wq_b shape) deliberately leaves alone — must STILL be
    // the sentinel on both sides.
    //
    // The two regions are counted APART. A single `c < out_stride` sweep counts
    // the intentional gap as "unwritten by both arms": m * (out_stride - n) false
    // positives, and the wq_b arm reported exactly that (12288 = 6 * 2048), so the
    // suite exited 1 on a healthy binary — which would have masked a REAL coverage
    // loss behind a known-red arm. The gap's sentinel-ness is asserted HERE rather
    // than left to the whole-buffer `mr_bits_equal` above: that comparison only
    // proves the two arms agree, not that neither wrote into the gap.
    size_t unwritten = 0, gap_written = 0;
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < out_stride; ++c) {
            uint32_t bm, br;
            std::memcpy(&bm, &om[(size_t)r * out_stride + c], 4);
            std::memcpy(&br, &orr[(size_t)r * out_stride + c], 4);
            const bool nan_m = (bm & 0x7FFFFFFFu) > 0x7F800000u;
            const bool nan_r = (br & 0x7FFFFFFFu) > 0x7F800000u;
            if (c < n)
                unwritten += (bm == br && nan_m) ? 1u : 0u;   // NaN both sides, write region
            else
                gap_written += (nan_m && nan_r) ? 0u : 1u;    // the gap must stay sentinel
        }
    MR_CHECK(unwritten == 0, "[%s] %zu element(s) of [0,n) left unwritten by both arms", tag, unwritten);
    MR_CHECK(gap_written == 0, "[%s] %zu element(s) of [n,out_stride) written into the sentinel gap",
             tag, gap_written);

    if (same && unwritten == 0 && gap_written == 0)
        printf("  [%s] OK  m=%d n=%d k=%d out_stride=%d bias=%d  (%zu elems bit-identical)\n", tag, m, n,
               k, out_stride, (int)use_bias, om.size());

    cudaFree(da); cudaFree(dw); cudaFree(dws); cudaFree(dasc); cudaFree(db);
    cudaFree(dout_m); cudaFree(dout_r);
    return same ? 0 : 1;
}

// ------------------------------------------- 1b: the act cp.async16 parity axis
// Both arms IN ONE PROCESS. `mrows_act_cpasync_host()` reads `getenv` on every
// launch, so the switch is live for the second arm without a re-exec — unlike
// `DSV41_MROWS_FOLD_R`, whose file-scope const is frozen at load time.
//
// The two arms are checked against the SAME m single-row references (`mr_case`'s
// contract), so "arm on == arm off" follows from "each == the reference", which
// is the stronger statement (it also catches an arm that is self-consistently
// wrong). Each arm prints its own `[<tag>/cp16=N] OK|FAIL` line, so a run's log
// names the arm that failed.
int mr_case_cp16_axis(const char* tag, int m, int n, int k, int out_stride, bool use_bias) {
    int fails = 0;
    char t[160];
    unsetenv("DSV41_MROWS_ACT_CPASYNC");
    std::snprintf(t, sizeof t, "%s/cp16=0", tag);
    fails += mr_case(t, m, n, k, out_stride, use_bias);
    // `overwrite = 1`: a caller's pre-set value is replaced, never inherited —
    // the axis must be exactly {unset, "1"} or the log lies about the arm.
    setenv("DSV41_MROWS_ACT_CPASYNC", "1", 1);
    std::snprintf(t, sizeof t, "%s/cp16=1", tag);
    fails += mr_case(t, m, n, k, out_stride, use_bias);
    unsetenv("DSV41_MROWS_ACT_CPASYNC");
    return fails;
}

// ---------------------------------------------- 1a: the fold_r contract pin
// The resolution table for the production shapes, printed so a run's log is
// self-describing, plus the two invariants the kernel relies on:
//   * `fold_r` clamps into [1, m] at EVERY setting (never 1-as-decline, never
//     > m), so `ng = ceil(m / fold_r)` is the only grid the launcher ever hands
//     the kernel;
//   * with the gate unset / `0` / `auto` the resolution IS the identity
//     (`fold_r = m`, `ng = 1`) — the 2026-09-12 fix. The first rule of this knob
//     folded `n <= 1024` to 1, which made the batched verify re-stage the same
//     weight row `ng = m` times (measured 63.8 -> 10.3 tok/s); pinning the
//     identity here keeps that regression from returning unseen.
//
// The SWEEP itself is one process per value (see the file header): the mr_case
// arms above then run on the folded grid and their bit-identity claim — which is
// exactly the claim 1a's wiring rests on — is re-checked at that fold_r.
int mr_fold_r_contract() {
    struct { const char* what; int n, m; } sh[] = {
        {"wkv", 512, 5},
        {"wq_a", 1280, 5},
        {"wq_a", 1280, 6},
        {"wq_b", 2048, 6},
        {"wo_b", 5120, 5},
        {"sh_w1/w3", 288, 6},
        // m = 1 is the lazy/per-row verify block: `fold_r` clamps into [1, 1],
        // i.e. the identity is the ONLY reachable program there, so 1a cannot
        // move anything on the lazy path at any setting of the knob.
        {"lazy/m=1", 512, 1},
        {"lazy/m=1/wq_a", 1280, 1},
    };
    const char* e = getenv("DSV41_MROWS_FOLD_R");
    const bool want_off = (e == nullptr) || (e[0] == '0' && e[1] == '\0') || e[0] == 'a' ||
                          e[0] == 'A';
    printf("  [fold_r] DSV41_MROWS_FOLD_R=%s -> resolved (fold_r, ng) per production shape:\n",
           e == nullptr ? "<unset>" : e);
    int fails = 0;
    for (const auto& s : sh) {
        const int fr = dsv41_mrows_fold_r_for(s.n, s.m);
        const int ng = (s.m + fr - 1) / fr;
        printf("           %-14s n=%-5d m=%d -> fold_r=%d ng=%d\n", s.what, s.n, s.m, fr, ng);
        MR_CHECK(fr >= 1 && fr <= s.m, "[fold_r] %s: fold_r=%d outside [1,%d]", s.what, fr, s.m);
        MR_CHECK(ng >= 1 && ng <= s.m, "[fold_r] %s: ng=%d outside [1,%d]", s.what, ng, s.m);
        if (want_off)
            MR_CHECK(fr == s.m && ng == 1,
                     "[fold_r] %s: gate OFF/auto must be the identity (fold_r=m, ng=1), got "
                     "fold_r=%d ng=%d",
                     s.what, fr, ng);
    }
    if (fails == 0)
        printf("  [fold_r] OK  (resolution inside [1,m]; %s)\n",
               want_off ? "identity held for OFF/auto/0" : "positive arm: identity NOT asserted");
    return fails;
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
    // ---- L4/L5 N1 axes (see the file header). The OFF arms are the six
    //      production runs above (the shipped default); these are the ON arms
    //      for 1b and the resolution pin for 1a.
    // 1b: both arms in-process over the shapes the lazy verify actually runs —
    //     the m=1 block included, because that IS the lazy/per-row block and the
    //     mrows kernel is the one it dispatches to at m=1.
    g_fails += mr_case_cp16_axis("wq_a/m=6+bias", 6, 1280, 5120, 1280, true);
    g_fails += mr_case_cp16_axis("wkv/m=5", 5, 512, 5120, 512, false);
    g_fails += mr_case_cp16_axis("wo_b/m=5", 5, 5120, 1024, 5120, false);
    g_fails += mr_case_cp16_axis("m=1", 1, 64, 512, 64, false);
    g_fails += mr_case_cp16_axis("lazy/m=1/wkv", 1, 512, 5120, 512, false);
    // 1a: the resolution table + the [1, m] clamp, plus the identity pin when the
    //     knob is OFF/auto/0. The fold sweep itself is one process per value
    //     (`for fr in 1 2 3 6; do DSV41_MROWS_FOLD_R=$fr ...; done`) — the arms
    //     above then re-check bit-identity on the folded grid.
    g_fails += mr_fold_r_contract();
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
