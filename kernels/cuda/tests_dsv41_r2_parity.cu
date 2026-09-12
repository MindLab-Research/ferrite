// tests_dsv41_r2_parity.cu — P0: the R2 BYTE-PARITY test.
//
// QUESTION THIS FILE ANSWERS, ONCE AND FOR ALL, ON REAL SHAPES:
//
//   Is `lin2` (dsv41_gemm_fp8_mx2) + `lin_rope_norm`
//   (dsv41_gemm_fp8_mx_rope_norm) BIT-FOR-BIT equal to the verify block's
//   separate chain
//
//       quant_rows + proj_mrows x2                       (wq_a + wkv)
//       norm_rows + quant_rows + proj_mrows              (wq_b)
//       apply_rope_mrows                                 (q rope)
//
//   ...or is one of those two kernels a DIFFERENT PROGRAM whose result differs
//   by ~1 ULP? That single ULP, cascaded over 40 layers x 60+ steps, is the
//   standing explanation for R2's (DSV41_ATTN_LIN_FUSE=1) corrupted output --
//   and it has never been measured, only argued ("the mrows kernel is claimed
//   to be gemm_fp8_mx's m == 1 program").
//
// This is NOT a tolerance test. Every comparison is a memcmp over the RAW f32
// bits (uint32_t), so -0.0 vs +0.0, NaN payloads and a single last-bit flip all
// count as differences. The report gives the COUNT of differing elements, the
// FIRST differing index (with head/column), the maximum ULP distance, and a hex
// dump of the neighbourhood of the first difference.
//
// ---------------------------------------------------------------------------
// THE REAL SHAPES (DeepSeek-V4.1-Flash production; crates/ferrite-models/src/
// dsv41/config.rs::production()):
//   dim = 5120, n_heads = 64, head_dim = 512, rope_head_dim = 64,
//   q_lora_rank = 1280, world = 8  =>  nlh = 8 heads per rank
//
//   wq_a / wkv  (lin2):   a[1, 5120] @ W[1280 | 512, 5120]^T
//                         -> n1 = ql = 1280, n2 = head_dim = 512, k = 5120
//   wq_b (lin_rope_norm): a[1, 1280] @ W[4096, 1280]^T + rope
//                         -> n = nlh*hd = 4096, k = ql = 1280,
//                            out_stride = nh*hd = 32768 (the rank writes its
//                            leading nlh*hd of an nh*hd-wide row)
//   rope:                 rd = 64, hd = 512, half = 32, mul = 1, off = 0,
//                         step = 0 (m == 1: every output row shares the
//                         position `*pos_ctr`, exactly as the separate chain's
//                         per-(row, head) apply_rope does)
//
// ---------------------------------------------------------------------------
// THE ARMS (all launchers are the SAME C entries the Rust chain calls; see
// crates/ferrite-models/src/dsv41/chain_dev.rs):
//
//   A  FUSED     (what R2 does)
//      mx2                : quant once -> dsv41_gemm_fp8_mx2 -> qr_r, kv_r
//      mx_rope_norm       : dsv41_gemm_fp8_mx_rope_norm(qr_raw) -> q_r
//                           (norm prologue + GEMV + rope epilogue, ONE launch)
//
//   B  SEPARATE  (what the verify block does with R2 off)
//      quant_fp8          : dsv41_quant_fp8(xn) -> xq/xsc
//      mrows x2           : dsv41_gemm_fp8_mrows -> qr_r, kv_r
//      rmsnorm_rows       : the q norm, in place on qr_r
//      quant_fp8          : dsv41_quant_fp8(qr_r)
//      mrows              : dsv41_gemm_fp8_mrows -> q_r (pre-rope)
//      apply_rope_mrows   : dsv41_apply_rope_mrows -> q_r (roped)
//
//   C  STAGE REFERENCE (decomposes A vs B into "which launch differs")
//      rmsnorm_q          : dsv41_rmsnorm_q -> normalised f32 + fp8 pair
//      mx_rope            : dsv41_gemm_fp8_mx_rope on C's fp8 pair -> q_r
//   A == C proves the fused PROLOGUE + GEMV + EPILOGUE is exactly
//   rmsnorm_q + gemm_fp8_mx_rope.
//   C != B proves the disagreement is inside mrows / apply_rope_mrows, not the
//   fused norm prologue.
//
//   D  CONTROLS (these MUST be bit-equal; if they are not, the harness or the
//      environment is broken, not R2)
//      D1  dsv41_gemm_fp8_mx(m=1)          vs dsv41_gemm_fp8_mrows(m=1)
//      D2  dsv41_gemm_fp8_mx(m=1) family 1 vs dsv41_gemm_fp8_mx2 family 1
//      D3  dsv41_gemm_fp8_mx(m=1) family 2 vs dsv41_gemm_fp8_mx2 family 2
//      D4  rmsnorm_rows output vs rmsnorm_q's elementwise output (the norm the
//          `lin_rope_norm` prologue claims to reproduce), plus the fp8 pair.
//
// Both arms always get the SAME fp8 activation BYTES: the quantiser is called
// twice into two separate buffers and those two buffers are memcmp'd first
// (the "staging determinism" check). Only then do the arms differ in the
// projection launch, so a diff cannot be blamed on the staging.
//
// ---------------------------------------------------------------------------
// INPUT COVERAGE (boundaries, not just random):
//   * all zero              -- 0/0 rms, the 1e-30 scale floor
//   * all 448 (fp8 clamp)    -- every element saturates
//   * all tiny (1e-30)       -- the scale floor and e4m3 subnormals
//   * random [-8, 8)         -- the production-like case
//   * random wide 2^k        -- magnitudes 2^-12..2^12, the block amax varies
//   * fp8 rounding boundary  -- values exactly between two e4m3 codes
//   * alternating signs      -- cancellation in the k-reduction
//   * one huge + rest tiny   -- the amax dominated by one element
// rotated over several deterministic weight seeds, rope positions and eps.
//
// ---------------------------------------------------------------------------
// Build (single TU, NO GPU needed to compile):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
//        -o /tmp/t_r2parity kernels/cuda/tests_dsv41_r2_parity.cu
// On a node whose CUDA toolkit comes from the pip `nvidia-cu13` wheel (this
// workspace's case: nvcc at /tmp/nvccx/nvidia/cu13/bin/nvcc, headers/libs under
// ~/.local/lib/python3.10/site-packages/nvidia/cu13), add the -I/-L:
//   CU=~/.local/lib/python3.10/site-packages/nvidia/cu13
//   /tmp/nvccx/nvidia/cu13/bin/nvcc -I$CU/include -L$CU/lib \
//        -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
//        -o /tmp/t_r2parity kernels/cuda/tests_dsv41_r2_parity.cu
// Run (ONE free GPU; peak allocation is ~16 MB):
//   CUDA_VISIBLE_DEVICES=7 /tmp/t_r2parity              # full suite
//   CUDA_VISIBLE_DEVICES=7 /tmp/t_r2parity --quick      # 2 seeds x 3 patterns
//   CUDA_VISIBLE_DEVICES=7 /tmp/t_r2parity --verbose    # also print every EQUAL
// Exit status: 0 = every measured comparison was bit-identical, 1 = at least
// one differed (or a harness failure).
//
// OPTIONAL EXTRA (one more TU on the command line):
//   nvcc ... -DPARITY_FERRITE_RMSNORM tests_dsv41_r2_parity.cu ferrite_kernels.cu
// additionally cross-checks `ferrite_rmsnorm` (the kernel the verify block
// actually uses with DSV41_NORM_MROWS unset) against `dsv41_rmsnorm_rows`. It is
// off by default so the documented build stays a single TU, exactly like
// tests_dsv41_gemm_mrows.cu.
//
// ENV THAT CHANGES WHAT IS COVERED (read once per process, before main):
//   DSV41_NO_GEMV_FP8    set => the m=1 GEMV is off, so the mrows AND the fused
//                        launchers decline -> every fused-vs-mrows arm reports
//                        SKIP (the harness still runs the controls).
//   DSV41_GEMV_FP8_MODE  must be >= 3 for the mrows arms to run (mode 0/1
//                        reorder a lane's elements and the launcher declines).
//                        The default is 4.
//   DSV41_GEMV_A32 / _STAGED / _WARPS / _WARPS_ADAPTIVE / _WARPS_BIG change
//                        the geometry. Parity is invariant under them by
//                        construction -- run the suite under a couple of
//                        settings if a diff shows up: a diff that MOVES with
//                        them is a geometry bug, one that does not is an
//                        expression bug.
#include "dsv41_kernels.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef PARITY_FERRITE_RMSNORM
// From ferrite_kernels.cu (add that TU to the nvcc command line).
extern "C" cudaError_t ferrite_rmsnorm(const float* x, const float* w, float* out, int n, int dim,
                                       float eps, cudaStream_t s);
#endif

namespace {

// ===========================================================================
// the real shapes
// ===========================================================================
constexpr int kDim = 5120;    // hidden
constexpr int kQl = 1280;     // q_lora_rank
constexpr int kHd = 512;      // head_dim
constexpr int kRd = 64;       // rope_head_dim
constexpr int kHalf = kRd / 2;
constexpr int kNh = 64;       // n_heads
constexpr int kWorld = 8;
constexpr int kNlh = kNh / kWorld;   // 8 heads per rank
constexpr int kNQ = kNlh * kHd;      // 4096: this rank's wq_b output width
constexpr int kOs = kNh * kHd;       // 32768: the (nh*hd) row pitch wq_b writes into
constexpr int kNbDim = kDim / 32;    // 160
constexpr int kNbQl = kQl / 32;      // 40
constexpr int kPosRows = 2048;       // rope table rows (positions 0..2047)

static_assert(kDim % 32 == 0 && kQl % 32 == 0 && kRd % 2 == 0 && kRd % 32 == 0,
              "shape precondition broken");
static_assert(kOs % 32 == 0 && kRd <= kHd && kHd % 32 == 0, "rope precondition broken");
static_assert(kNQ % 32 == 0 && kQl % 32 == 0, "wq_b precondition broken");

// ===========================================================================
// reporting
// ===========================================================================
int g_fails = 0;   // a FAIL is an EXPECTED-equal comparison that differed
int g_diffs = 0;   // an ARM comparison that differed (the measurement itself)
int g_skips = 0;

#define P_CHECK(expr, fmt, ...)                                                    \
    do {                                                                           \
        if (!(expr)) {                                                             \
            printf("    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            ++g_fails;                                                             \
        }                                                                          \
    } while (0)

struct Acc {
    char tag[96];
    size_t cases = 0;
    size_t elems = 0;
    size_t bad_elems = 0;
    size_t bad_cases = 0;
    size_t first_bad_case = 0;
    uint64_t max_ulp = 0;
    bool skipped = false;
    size_t skips = 0;
};
std::vector<Acc> g_acc;
bool g_verbose = false;   // print every EQUAL case (default: only DIFF + a count)

Acc& acc(const char* tag) {
    for (auto& a : g_acc)
        if (std::strcmp(a.tag, tag) == 0) return a;
    Acc a{};
    std::snprintf(a.tag, sizeof(a.tag), "%s", tag);
    g_acc.push_back(a);
    return g_acc.back();
}

// Monotone f32 -> uint32 ordering key, so ULP distance is a plain subtract.
inline uint32_t f2o(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
inline uint32_t bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}
inline bool is_nan_bits(uint32_t b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
inline bool is_inf_bits(uint32_t b) { return (b & 0x7FFFFFFFu) == 0x7F800000u; }

// Which lanes of a (nlh x kHd = kNQ) q row participate.
enum Region { R_ALL, R_PREROPE, R_ROPE };

struct Diff {
    size_t consider = 0;
    size_t bad = 0;
    size_t first = (size_t)-1;
    uint64_t max_ulp = 0;
    size_t nan_got = 0, nan_exp = 0, inf_got = 0, inf_exp = 0;
};

Diff diff_bits(const std::vector<float>& got, const std::vector<float>& exp, size_t n,
               Region reg = R_ALL) {
    Diff d;
    if (got.size() < n || exp.size() < n) {
        printf("    FAIL diff_bits: buffer too small (%zu/%zu < %zu)\n", got.size(), exp.size(), n);
        ++g_fails;
        return d;
    }
    for (size_t i = 0; i < n; ++i) {
        if (reg != R_ALL) {
            const size_t c = i % (size_t)kHd;
            const bool is_rope = c >= (size_t)(kHd - kRd);
            if ((reg == R_ROPE) != is_rope) continue;
        }
        ++d.consider;
        const uint32_t a = bits(got[i]), b = bits(exp[i]);
        if (is_nan_bits(a)) ++d.nan_got;
        if (is_nan_bits(b)) ++d.nan_exp;
        if (is_inf_bits(a)) ++d.inf_got;
        if (is_inf_bits(b)) ++d.inf_exp;
        if (a == b) continue;
        if (d.bad == 0) d.first = i;
        ++d.bad;
        const uint32_t oa = f2o(got[i]), ob = f2o(exp[i]);
        const uint64_t u = oa > ob ? (uint64_t)(oa - ob) : (uint64_t)(ob - oa);
        if (u > d.max_ulp) d.max_ulp = u;
    }
    return d;
}

void dump_at(const std::vector<float>& got, const std::vector<float>& exp, size_t at, size_t n,
             bool with_pos) {
    for (size_t i = at; i < at + n; ++i) {
        if (i >= got.size() || i >= exp.size()) break;
        if (with_pos) {
            printf("      [%6zu] head=%zu incol=%zu  A=0x%08x %+15.8e   B=0x%08x %+15.8e %s\n", i,
                   i / (size_t)kHd, i % (size_t)kHd, bits(got[i]), (double)got[i], bits(exp[i]),
                   (double)exp[i], bits(got[i]) == bits(exp[i]) ? "" : "<-- DIFF");
        } else {
            printf("      [%6zu]                    A=0x%08x %+15.8e   B=0x%08x %+15.8e %s\n", i,
                   bits(got[i]), (double)got[i], bits(exp[i]), (double)exp[i],
                   bits(got[i]) == bits(exp[i]) ? "" : "<-- DIFF");
        }
    }
}

// One arm comparison. A = `got` (fused), B = `exp` (separate).
void compare(const char* tag, const char* what, const std::vector<float>& got,
             const std::vector<float>& exp, size_t n, Region reg = R_ALL, bool with_pos = false) {
    Acc& a = acc(tag);
    const Diff d = diff_bits(got, exp, n, reg);
    ++a.cases;
    a.elems += d.consider;
    a.bad_elems += d.bad;
    if (d.max_ulp > a.max_ulp) a.max_ulp = d.max_ulp;
    if (d.bad == 0) {
        if (g_verbose)
            printf("  [%s] EQUAL  %s (%zu elems considered, max_ulp=0)\n", tag, what, d.consider);
        return;
    }
    ++a.bad_cases;
    if (a.first_bad_case == 0) a.first_bad_case = a.cases;
    ++g_diffs;
    printf("  [%s] *** DIFF ***  %s\n", tag, what);
    printf("        %zu/%zu elems differ (%.4f%%), first at idx=%zu", d.bad, d.consider,
           100.0 * (double)d.bad / (double)d.consider, d.first);
    if (with_pos) printf(" (head=%zu incol=%zu)", d.first / (size_t)kHd, d.first % (size_t)kHd);
    printf(", max_ulp=%llu\n", (unsigned long long)d.max_ulp);
    printf("        A(fused)=0x%08x %+15.8e   B(separate)=0x%08x %+15.8e\n", bits(got[d.first]),
           (double)got[d.first], bits(exp[d.first]), (double)exp[d.first]);
    if (d.nan_got || d.nan_exp || d.inf_got || d.inf_exp)
        printf("        NaN: A=%zu B=%zu   Inf: A=%zu B=%zu\n", d.nan_got, d.nan_exp, d.inf_got,
               d.inf_exp);
    const size_t lo = d.first > 3 ? d.first - 3 : 0;
    dump_at(got, exp, lo, 8, with_pos);
}

// Compare two raw byte buffers (the fp8 staging).
void compare_bytes(const char* tag, const char* what, const std::vector<uint8_t>& got,
                   const std::vector<uint8_t>& exp) {
    Acc& a = acc(tag);
    ++a.cases;
    a.elems += exp.size();
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < exp.size(); ++i) {
        if (got[i] == exp[i]) continue;
        if (bad == 0) first = i;
        ++bad;
    }
    a.bad_elems += bad;
    if (bad == 0) {
        if (g_verbose) printf("  [%s] EQUAL  %s (%zu bytes)\n", tag, what, exp.size());
        return;
    }
    ++a.bad_cases;
    if (a.first_bad_case == 0) a.first_bad_case = a.cases;
    ++g_diffs;
    printf("  [%s] *** DIFF ***  %s: %zu/%zu bytes differ, first at %zu (A=0x%02x B=0x%02x)\n", tag,
           what, bad, exp.size(), first, got[first], exp[first]);
}

void skip(const char* tag, const char* why) {
    Acc& a = acc(tag);
    ++a.skips;
    if (!a.skipped) printf("  [%s] SKIP: %s\n", tag, why);
    a.skipped = true;
    ++g_skips;
}

// ===========================================================================
// deterministic input generation
// ===========================================================================
uint32_t g_rng = 0x9E3779B9u;
uint32_t xr() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
float xf() { return (float)(xr() & 0xFFFFFFu) / (float)0x1000000u; }   // [0, 1)
int xirand(int n) { return (int)(xr() % (uint32_t)n); }

// e4m3 byte that is NOT a NaN (0x7F / 0xFF are NaN and would poison a bit
// compare: a reordered kernel could turn one NaN payload into another and the
// failure would be ambiguous).
uint8_t e4m3_byte() {
    uint8_t b;
    do {
        b = (uint8_t)(xr() & 0xFFu);
    } while ((b & 0x7Fu) == 0x7Fu);
    return b;
}
// ue8m0: 2^(e-127); e in 120..134 keeps the products inside f32 range.
uint8_t ue8m0_byte() { return (uint8_t)(120 + xirand(15)); }

enum Pattern {
    P_ZERO,     // all 0
    P_CLAMP,    // all 448 (the fp8 clamp), mixed signs
    P_TINY,     // all 1e-30 (the scale floor)
    P_RANDOM,   // uniform [-8, 8)
    P_WIDE,     // +-2^k, k in -12..12
    P_EDGE,     // exactly between two e4m3 codes -> a rounding tie everywhere
    P_SIGNS,    // alternating +-1, one huge element
    P_SPIKE,    // one huge + the rest denormal-tiny
    P_COUNT
};

const char* pattern_name(Pattern p) {
    switch (p) {
        case P_ZERO: return "zero";
        case P_CLAMP: return "clamp448";
        case P_TINY: return "tiny1e-30";
        case P_RANDOM: return "random";
        case P_WIDE: return "wide2^k";
        case P_EDGE: return "fp8-edge";
        case P_SIGNS: return "signs";
        case P_SPIKE: return "spike";
        default: return "?";
    }
}

void fill_activation(std::vector<float>& v, Pattern p) {
    const size_t n = v.size();
    switch (p) {
        case P_ZERO:
            for (size_t i = 0; i < n; ++i) v[i] = 0.0f;
            break;
        case P_CLAMP:
            for (size_t i = 0; i < n; ++i) v[i] = (i & 1) ? -448.0f : 448.0f;
            break;
        case P_TINY:
            for (size_t i = 0; i < n; ++i) v[i] = (i & 1) ? -1e-30f : 1e-30f;
            break;
        case P_RANDOM:
            for (size_t i = 0; i < n; ++i) v[i] = (xf() * 16.0f) - 8.0f;
            break;
        case P_WIDE:
            for (size_t i = 0; i < n; ++i) {
                const int k = xirand(25) - 12;
                v[i] = std::ldexp((xf() * 2.0f) - 1.0f, k);
            }
            break;
        case P_EDGE:
            // A tie in the e4m3 round: (m + 0.5) * 2^e. One in three gets a
            // 1-ULP nudge so the tie is broken both ways.
            for (size_t i = 0; i < n; ++i) {
                const int e = xirand(10) - 4;
                const int m = 1 + xirand(126);
                float f = std::ldexp((float)m + 0.5f, e);
                const int r = xirand(3);
                if (r == 1) f = std::nextafterf(f, 1e30f);
                if (r == 2) f = std::nextafterf(f, -1e30f);
                v[i] = (i & 1) ? -f : f;
            }
            break;
        case P_SIGNS:
            for (size_t i = 0; i < n; ++i) v[i] = (i & 1) ? -1.0f : 1.0f;
            if (n) v[n / 3] = 3.0e4f;
            break;
        case P_SPIKE:
            for (size_t i = 0; i < n; ++i) v[i] = (i & 1) ? -1e-32f : 1e-32f;
            if (n) v[n / 2] = 1.0e5f;
            break;
        default:
            break;
    }
}

// ===========================================================================
// device buffers
// ===========================================================================
constexpr uint32_t kSentinelBits = 0x7FC0DEADu;   // a quiet NaN, so "untouched"
                                                  // is never a plausible value
void fill_sentinel(std::vector<float>& v) {
    const uint32_t s = kSentinelBits;
    for (size_t i = 0; i < v.size(); ++i) std::memcpy(&v[i], &s, 4);
}
size_t still_sentinel(const std::vector<float>& v, size_t from, size_t to) {
    size_t c = 0;
    for (size_t i = from; i < to && i < v.size(); ++i)
        if (bits(v[i]) == kSentinelBits) ++c;
    return c;
}

struct Dev {
    float* x = nullptr;                       // [kDim] activation
    uint8_t* xqA = nullptr;                   // [kDim] fp8 activation, staging A
    float* xscA = nullptr;                    // [kDim/32]
    uint8_t* xqB = nullptr;                   // [kDim] fp8 activation, staging B
    float* xscB = nullptr;
    float *qrA = nullptr, *qrB = nullptr, *qrC = nullptr;   // [kQl]
    float *kvA = nullptr, *kvB = nullptr, *kvC = nullptr;   // [kHd]
    uint8_t *wqa = nullptr, *wsqa = nullptr;                // wq_a  [kQl, kDim]
    uint8_t *wkv = nullptr, *wskv = nullptr;                // wkv   [kHd, kDim]
    uint8_t *wqb = nullptr, *wsqb = nullptr;                // wq_b  [kNQ, kQl]
    float *qA = nullptr, *qB = nullptr, *qC = nullptr, *qD = nullptr;   // [kOs]
    float* qrRawA = nullptr;                  // [kQl] raw wq_b input (arm A)
    float* qrRawB = nullptr;                  // [kQl] an identical copy (arm B)
    uint8_t* xqQ = nullptr;                   // [kQl] fp8 of rmsnorm_q's output
    float* xscQ = nullptr;                    // [kQl/32]
    uint8_t* xqR = nullptr;                   // [kQl] fp8 of arm B's qr
    float* xscR = nullptr;
    float* normOut = nullptr;                 // [kQl] rmsnorm_q's f32 output
    float* qnorm = nullptr;                   // [kQl] the q_norm weight
    float* cos = nullptr, *sin = nullptr;     // [kPosRows * kHalf]
    int* pos_ctr = nullptr;                   // the device position counter
    int* pos_rows = nullptr;                  // [kNlh] per-row positions (m == 1)
};

#define ALLOC(p, nbytes, what)                                                        \
    do {                                                                              \
        if (cudaMalloc((void**)&(p), (nbytes)) != cudaSuccess) {                       \
            printf("FATAL cudaMalloc %s (%zu bytes): %s\n", what, (size_t)(nbytes),     \
                   cudaGetErrorString(cudaGetLastError()));                          \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

int h2d(void* dst, const void* src, size_t nbytes, const char* what) {
    const cudaError_t e = cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        printf("    FAIL H2D %s: %s\n", what, cudaGetErrorString(e));
        ++g_fails;
        return 1;
    }
    return 0;
}
int d2h(void* dst, const void* src, size_t nbytes, const char* what) {
    const cudaError_t e = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        printf("    FAIL D2H %s: %s\n", what, cudaGetErrorString(e));
        ++g_fails;
        return 1;
    }
    return 0;
}

// The ue8m0 (row-block, k-block) scale plane of an [n, k] fp8 weight:
// `scale[row >> 5][kb]`, so the tail row block needs its own row.
size_t ws_bytes(int n, int k) { return (size_t)(n / 32 + 2) * (size_t)(k / 32); }

// Read a device buffer back into a host vector (sized on first use).
std::vector<float> read_f32(const float* p, size_t n) {
    std::vector<float> v(n);
    d2h(v.data(), p, n * 4, "read_f32");
    return v;
}
std::vector<uint8_t> read_u8(const uint8_t* p, size_t n) {
    std::vector<uint8_t> v(n);
    d2h(v.data(), p, n, "read_u8");
    return v;
}

// Report a launcher's return code. 0 = ran, 2 = declined (the caller's
// "keep the old path" signal), anything else = a real failure.
bool ok_rc(const char* tag, const char* what, int rc) {
    if (rc == 2) {
        skip(tag, what);
        return false;
    }
    if (rc != 0) {
        printf("    FAIL %s: %s returned %d (%s)\n", tag, what, rc,
               cudaGetErrorString((cudaError_t)rc));
        ++g_fails;
        return false;
    }
    return true;
}

}  // namespace

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        else if (std::strcmp(argv[i], "--verbose") == 0) g_verbose = true;
        else if (std::strcmp(argv[i], "--help") == 0) {
            printf("usage: %s [--quick] [--verbose]\n", argv[0]);
            return 0;
        }
    }

    const int n_wseed = quick ? 2 : 6;
    const int pos_bases[5] = {0, 1, 37, 600, 1023};
    const int n_pos = quick ? 2 : 5;
    const float eps_list[2] = {1e-20f, 1e-6f};   // production is 1e-20
    const int n_eps = quick ? 1 : 2;

    printf("== dsv41 R2 byte-parity: fused lin2 + lin_rope_norm  vs  the separate chain ==\n");
    printf("   shapes: dim=%d ql=%d hd=%d rd=%d nh=%d world=%d -> nlh=%d n_q=%d out_stride=%d\n",
           kDim, kQl, kHd, kRd, kNh, kWorld, kNlh, kNQ, kOs);
    printf("   gates : mode=%d a32=%d a32_staged=%d warps=%d adaptive=%d warps_big=%d "
           "NO_GEMV_FP8=%d\n",
           g_gemv_fp8_mode, (int)g_gemv_a32, (int)g_gemv_a32_staged, g_gemv_warps,
           (int)g_gemv_warps_adaptive, g_gemv_warps_big,
           (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr));
    printf("   nwarps: mrows(1280)=%d mrows(512)=%d mrows(4096)=%d mx2(1792)=%d  "
           "rope-family=32 (forced)\n",
           dsv41_mrows_warps_for(kQl), dsv41_mrows_warps_for(kHd), dsv41_mrows_warps_for(kNQ),
           dsv41_gemv_warps_for(kQl + kHd));
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("   device %d: %s (sm_%d%d, %d SMs)\n", dev, prop.name, prop.major, prop.minor,
               prop.multiProcessorCount);
    printf("   sweep : wseeds=%d patterns=%d rope positions=%d eps=%d%s\n\n", n_wseed, (int)P_COUNT,
           n_pos, n_eps, quick ? "  [--quick]" : "");

    const bool can_any = (getenv("DSV41_NO_GEMV_FP8") == nullptr);
    if (!can_any)
        printf("!! DSV41_NO_GEMV_FP8 is set: the m=1 GEMV family is disabled, the fused and mrows "
               "launchers will decline -> the measurement arms report SKIP.\n");
    if (g_gemv_fp8_mode < 3)
        printf("!! DSV41_GEMV_FP8_MODE=%d (< 3): the vectorised/scalar arms reorder a lane's "
               "elements, so dsv41_gemm_fp8_mrows declines -> SKIP.\n",
               g_gemv_fp8_mode);

    // ---- allocation -------------------------------------------------------
    Dev d;
    ALLOC(d.x, kDim * 4, "x");
    ALLOC(d.xqA, kDim, "xqA");
    ALLOC(d.xscA, kNbDim * 4, "xscA");
    ALLOC(d.xqB, kDim, "xqB");
    ALLOC(d.xscB, kNbDim * 4, "xscB");
    ALLOC(d.qrA, kQl * 4, "qrA");
    ALLOC(d.qrB, kQl * 4, "qrB");
    ALLOC(d.qrC, kQl * 4, "qrC");
    ALLOC(d.kvA, kHd * 4, "kvA");
    ALLOC(d.kvB, kHd * 4, "kvB");
    ALLOC(d.kvC, kHd * 4, "kvC");
    ALLOC(d.wqa, (size_t)kQl * kDim, "wqa");
    ALLOC(d.wsqa, ws_bytes(kQl, kDim), "wsqa");
    ALLOC(d.wkv, (size_t)kHd * kDim, "wkv");
    ALLOC(d.wskv, ws_bytes(kHd, kDim), "wskv");
    ALLOC(d.wqb, (size_t)kNQ * kQl, "wqb");
    ALLOC(d.wsqb, ws_bytes(kNQ, kQl), "wsqb");
    ALLOC(d.qA, (size_t)kOs * 4, "qA");
    ALLOC(d.qB, (size_t)kOs * 4, "qB");
    ALLOC(d.qC, (size_t)kOs * 4, "qC");
    ALLOC(d.qD, (size_t)kOs * 4, "qD");
    ALLOC(d.qrRawA, kQl * 4, "qrRawA");
    ALLOC(d.qrRawB, kQl * 4, "qrRawB");
    ALLOC(d.xqQ, kQl, "xqQ");
    ALLOC(d.xscQ, kNbQl * 4, "xscQ");
    ALLOC(d.xqR, kQl, "xqR");
    ALLOC(d.xscR, kNbQl * 4, "xscR");
    ALLOC(d.normOut, kQl * 4, "normOut");
    ALLOC(d.qnorm, kQl * 4, "qnorm");
    ALLOC(d.cos, (size_t)kPosRows * kHalf * 4, "cos");
    ALLOC(d.sin, (size_t)kPosRows * kHalf * 4, "sin");
    ALLOC(d.pos_ctr, 4, "pos_ctr");
    ALLOC(d.pos_rows, kNlh * 4, "pos_rows");

    // ---- the rope tables and the q_norm weight ---------------------------
    {
        std::vector<float> hcos((size_t)kPosRows * kHalf), hsin((size_t)kPosRows * kHalf);
        // theta_i = i / half (a descending-frequency table, the usual shape);
        // the absolute values only have to be the SAME for both arms, which
        // they are by construction (one table, one device buffer).
        for (int t = 0; t < kPosRows; ++t)
            for (int i = 0; i < kHalf; ++i) {
                const double theta = (double(t) * 0.017) + (double(i) / double(kHalf)) * 3.7;
                hcos[(size_t)t * kHalf + i] = (float)std::cos(theta);
                hsin[(size_t)t * kHalf + i] = (float)std::sin(theta);
            }
        h2d(d.cos, hcos.data(), hcos.size() * 4, "cos");
        h2d(d.sin, hsin.data(), hsin.size() * 4, "sin");

        std::vector<float> hqn(kQl);
        for (int i = 0; i < kQl; ++i) hqn[i] = 0.5f + xf() * 1.5f;   // positive, o(1)
        h2d(d.qnorm, hqn.data(), kQl * 4, "qnorm");
    }

    // ---- the sweep --------------------------------------------------------
    std::vector<float> hx(kDim);          // the wq_a/wkv activation
    std::vector<float> hqr(kQl);          // the wq_b activation (raw)
    std::vector<uint8_t> hqa((size_t)kQl * kDim), hwka((size_t)kHd * kDim),
        hwba((size_t)kNQ * kQl);
    std::vector<uint8_t> hqas(ws_bytes(kQl, kDim)), hwkas(ws_bytes(kHd, kDim)),
        hwbas(ws_bytes(kNQ, kQl));

    for (int ws = 0; ws < n_wseed; ++ws) {
        printf("---- weight seed %d/%d ----\n", ws + 1, n_wseed);
        for (auto& b : hqa) b = e4m3_byte();
        for (auto& b : hwka) b = e4m3_byte();
        for (auto& b : hwba) b = e4m3_byte();
        for (auto& b : hqas) b = ue8m0_byte();
        for (auto& b : hwkas) b = ue8m0_byte();
        for (auto& b : hwbas) b = ue8m0_byte();
        h2d(d.wqa, hqa.data(), hqa.size(), "wqa");
        h2d(d.wsqa, hqas.data(), hqas.size(), "wsqa");
        h2d(d.wkv, hwka.data(), hwka.size(), "wkv");
        h2d(d.wskv, hwkas.data(), hwkas.size(), "wskv");
        h2d(d.wqb, hwba.data(), hwba.size(), "wqb");
        h2d(d.wsqb, hwbas.data(), hwbas.size(), "wsqb");

        for (int pi = 0; pi < (int)P_COUNT; ++pi) {
            const Pattern pat = (Pattern)pi;
            fill_activation(hx, pat);
            h2d(d.x, hx.data(), hx.size() * 4, "x");

            // =============================================================
            // staging determinism: the quantiser must emit the same bytes
            // twice, otherwise no downstream comparison means anything.
            // =============================================================
            if (ok_rc("T0.staging", "quant_fp8(A)", dsv41_quant_fp8(d.x, d.xqA, d.xscA, 1, kDim, 32, 1, nullptr)) &
                ok_rc("T0.staging", "quant_fp8(B)", dsv41_quant_fp8(d.x, d.xqB, d.xscB, 1, kDim, 32, 1, nullptr))) {
                compare_bytes("T0.staging", "xq bytes", read_u8(d.xqA, kDim), read_u8(d.xqB, kDim));
                compare("T0.staging", "xsc floats", read_f32(d.xscA, kNbDim), read_f32(d.xscB, kNbDim), kNbDim);
            }

            // =============================================================
            // T1/T2/T3 + controls: the wq_a + wkv half
            // =============================================================
            const bool ran_mx2 = ok_rc(
                "T1.lin2-vs-mrows", "gemm_fp8_mx2",
                dsv41_gemm_fp8_mx2(d.xqA, d.xscA, d.wqa, d.wsqa, nullptr, d.qrA, kQl, d.wkv, d.wskv,
                                   nullptr, d.kvA, kHd, kDim, nullptr));
            const bool ran_mrows = ok_rc(
                "T1.lin2-vs-mrows", "gemm_fp8_mrows(wq_a)",
                dsv41_gemm_fp8_mrows(d.xqB, d.xscB, d.wqa, d.wsqa, nullptr, d.qrB, 1, kQl, kDim, kQl,
                                     nullptr));
            const bool ran_mrows2 = ok_rc(
                "T1.lin2-vs-mrows", "gemm_fp8_mrows(wkv)",
                dsv41_gemm_fp8_mrows(d.xqB, d.xscB, d.wkv, d.wskv, nullptr, d.kvB, 1, kHd, kDim, kHd,
                                     nullptr));
            const bool ran_mx = ok_rc(
                "D1.mx-vs-mrows", "gemm_fp8_mx(wq_a, m=1)",
                dsv41_gemm_fp8_mx(d.xqB, d.xscB, d.wqa, d.wsqa, nullptr, d.qrC, 1, kQl, kDim,
                                  nullptr));
            const bool ran_mx_kv = ok_rc(
                "D1.mx-vs-mrows", "gemm_fp8_mx(wkv, m=1)",
                dsv41_gemm_fp8_mx(d.xqB, d.xscB, d.wkv, d.wskv, nullptr, d.kvC, 1, kHd, kDim,
                                  nullptr));
            if (ran_mx2 && ran_mrows && ran_mrows2) {
                printf("  wseed=%d pattern=%s: T1 (lin2 vs quant+mrows)\n", ws, pattern_name(pat));
                compare("T1.lin2-vs-mrows", "qr_r  (wq_a: fused mx2 vs mx+mrows)", read_f32(d.qrA, kQl),
                        read_f32(d.qrB, kQl), kQl);
                compare("T1.lin2-vs-mrows", "kv_r  (wkv : fused mx2 vs mx+mrows)", read_f32(d.kvA, kHd),
                        read_f32(d.kvB, kHd), kHd);
            }
            if (ran_mx && ran_mrows) {
                // D1: the two M=1 programs on the SAME staging bytes.
                compare("D1.mx-vs-mrows", "gemm_fp8_mx(m=1) vs gemm_fp8_mrows(m=1), wq_a",
                        read_f32(d.qrC, kQl), read_f32(d.qrB, kQl), kQl);
            }
            if (ran_mx_kv && ran_mrows2) {
                compare("D1.mx-vs-mrows", "gemm_fp8_mx(m=1) vs gemm_fp8_mrows(m=1), wkv",
                        read_f32(d.kvC, kHd), read_f32(d.kvB, kHd), kHd);
            }
            if (ran_mx2 && ran_mx) {
                // D2/D3: mx2's two families vs the two single-family GEMVs.
                compare("D2.mx-vs-mx2", "family 1 (wq_a): mx2 vs mx(m=1)", read_f32(d.qrA, kQl),
                        read_f32(d.qrC, kQl), kQl);
            }
            if (ran_mx2 && ran_mx_kv) {
                compare("D3.mx-vs-mx2", "family 2 (wkv): mx2 vs mx(m=1)", read_f32(d.kvA, kHd),
                        read_f32(d.kvC, kHd), kHd);
            }

            // =============================================================
            // T2/T3/T3b/T4: the wq_b half (norm + GEMV + rope), 3-way + controls
            // =============================================================
            fill_activation(hqr, pat);
            for (int e = 0; e < n_eps; ++e) {
                const float eps = eps_list[e];
                for (int pb = 0; pb < n_pos; ++pb) {
                    const int pos = pos_bases[pb];
                    // arm A and arm B must start from the SAME raw bytes.
                    h2d(d.qrRawA, hqr.data(), hqr.size() * 4, "qrRawA");
                    h2d(d.qrRawB, hqr.data(), hqr.size() * 4, "qrRawB");
                    h2d(d.pos_ctr, &pos, 4, "pos_ctr");
                    {
                        std::vector<int> pr(kNlh, pos);
                        h2d(d.pos_rows, pr.data(), kNlh * 4, "pos_rows");
                    }

                    // ---- C: rmsnorm_q + gemm_fp8_mx_rope (the stage reference)
                    const bool ran_rq = ok_rc(
                        "T3.fused-vs-rmsnorm_q+mxrope", "rmsnorm_q",
                        dsv41_rmsnorm_q(d.qrRawA, d.qnorm, d.normOut, 1, kQl, eps, d.xqQ, d.xscQ,
                                        nullptr));
                    const bool ran_mxrope = ok_rc(
                        "T3b.mxrope-vs-mrows", "gemm_fp8_mx_rope",
                        dsv41_gemm_fp8_mx_rope(d.xqQ, d.xscQ, d.wqb, d.wsqb, nullptr, d.qC, kNQ, kQl,
                                               d.cos, d.sin, d.pos_ctr, 1, 0, 0, 0, kRd, kHd,
                                               nullptr));
                    // ---- A: the fused launch (what R2 runs)
                    const bool ran_fused = ok_rc(
                        "T2.fused-vs-sep", "gemm_fp8_mx_rope_norm",
                        dsv41_gemm_fp8_mx_rope_norm(d.qrRawA, d.qnorm, eps, d.wqb, d.wsqb, nullptr,
                                                    d.qA, kNQ, kQl, d.cos, d.sin, d.pos_ctr, 1, 0, 0,
                                                    0, kRd, kHd, nullptr));
                    // ---- B: the separate chain (what the verify block runs)
                    const bool ran_norm = ok_rc(
                        "T4.norm", "rmsnorm_rows (in place)",
                        dsv41_rmsnorm_rows(d.qrRawB, d.qnorm, d.qrRawB, 1, kQl, eps, nullptr));
                    const bool ran_qb = ok_rc(
                        "T2.fused-vs-sep", "quant_fp8(qr_r)",
                        dsv41_quant_fp8(d.qrRawB, d.xqR, d.xscR, 1, kQl, 32, 1, nullptr));
                    const bool ran_mb = ok_rc(
                        "T2.fused-vs-sep", "gemm_fp8_mrows(wq_b)",
                        dsv41_gemm_fp8_mrows(d.xqR, d.xscR, d.wqb, d.wsqb, nullptr, d.qB, 1, kNQ,
                                             kQl, kOs, nullptr));
                    const bool ran_rope = ok_rc(
                        "T2.fused-vs-sep", "apply_rope_mrows",
                        dsv41_apply_rope_mrows(d.qB, d.cos, d.sin, 1, kNlh, kOs, kHd, kRd, kHalf,
                                               d.pos_rows, 0, nullptr));

                    if (ran_rq && ran_fused && ran_norm) {
                        // T4: the normalisation itself. rmsnorm_q writes the
                        // same elementwise output the `lin_rope_norm` prologue
                        // claims to reproduce in shared memory.
                        compare("T4.norm", "rmsnorm_q out vs rmsnorm_rows out", read_f32(d.normOut, kQl),
                                read_f32(d.qrRawB, kQl), kQl);
#ifdef PARITY_FERRITE_RMSNORM
                        // and the kernel the verify block uses when
                        // DSV41_NORM_MROWS is unset.
                        {
                            static float* scratch = nullptr;
                            if (!scratch) cudaMalloc((void**)&scratch, kQl * 4);
                            h2d(d.qrRawA, hqr.data(), hqr.size() * 4, "qrRawA(reload)");
                            const cudaError_t fe =
                                ferrite_rmsnorm(d.qrRawA, d.qnorm, scratch, 1, kQl, eps, nullptr);
                            P_CHECK(fe == cudaSuccess, "ferrite_rmsnorm: %s", cudaGetErrorString(fe));
                            compare("T4.norm", "ferrite_rmsnorm vs rmsnorm_rows", read_f32(scratch, kQl),
                                    read_f32(d.qrRawB, kQl), kQl);
                        }
#endif
                    }
                    if (ran_rq && ran_qb) {
                        compare_bytes("T4.norm", "fp8 bytes: rmsnorm_q vs rmsnorm_rows+quant",
                                      read_u8(d.xqQ, kQl), read_u8(d.xqR, kQl));
                        compare("T4.norm", "fp8 scales: rmsnorm_q vs rmsnorm_rows+quant",
                                read_f32(d.xscQ, kNbQl), read_f32(d.xscR, kNbQl), kNbQl);
                    }
                    if (ran_fused && ran_mxrope && ran_rq) {
                        // T3: this is the isolation that matters -- if the fused
                        // launch is exactly rmsnorm_q + gemm_fp8_mx_rope, then
                        // any diff below is the mrows/rope family's, not the
                        // fused prologue's.
                        printf("  wseed=%d pattern=%s eps=%.0e pos=%d: wq_b half\n", ws,
                               pattern_name(pat), (double)eps, pos);
                        compare("T3.fused-vs-rmsnorm_q+mxrope", "q_r (fused vs rmsnorm_q+mx_rope)",
                                read_f32(d.qA, kNQ), read_f32(d.qC, kNQ), kNQ, R_ALL, true);
                    }
                    if (ran_fused && ran_rope && ran_mb) {
                        printf("  wseed=%d pattern=%s eps=%.0e pos=%d: wq_b half\n", ws,
                               pattern_name(pat), (double)eps, pos);
                        // T2: fused vs the FULL separate chain.
                        compare("T2.fused-vs-sep", "q_r (fused vs norm+quant+mrows+rope)",
                                read_f32(d.qA, kNQ), read_f32(d.qB, kNQ), kNQ, R_ALL, true);
                        // region split: where exactly does it differ?
                        compare("T2.fused-vs-sep.pre-rope", "q_r pre-rope lanes (per head < 448)",
                                read_f32(d.qA, kNQ), read_f32(d.qB, kNQ), kNQ, R_PREROPE, true);
                        compare("T2.fused-vs-sep.rope-lane", "q_r rope lanes (per head >= 448)",
                                read_f32(d.qA, kNQ), read_f32(d.qB, kNQ), kNQ, R_ROPE, true);
                    }
                    if (ran_mxrope && ran_rq && ran_rope && ran_mb) {
                        // T3b: mx_rope chain vs the mrows chain -- the mrows /
                        // apply_rope_mrows question on its own, with the norm
                        // removed from the comparison (both sides consume the
                        // SAME rmsnorm_q fp8 output? no: B consumes its own
                        // rmsnorm_rows+quant pair, which T4 has just checked is
                        // identical to rmsnorm_q's). So this compares the two
                        // GEMM+rope families.
                        compare("T3b.mxrope-vs-mrows", "q_r (mx_rope vs mrows+apply_rope_mrows)",
                                read_f32(d.qC, kNQ), read_f32(d.qB, kNQ), kNQ, R_ALL, true);
                    }
                }   // pos
            }       // eps

            // =============================================================
            // T5: the REAL end-to-end chain (the headline number)
            //     A: lin2 -> lin_rope_norm(qr_r raw)
            //     B: quant -> mrows x2 -> rmsnorm_rows -> quant -> mrows ->
            //        apply_rope_mrows
            // =============================================================
            if (ran_mx2 && ran_mrows) {
                const float eps = eps_list[0];
                {
                    std::vector<int> pr(kNlh, pos_bases[0]);
                    int p0 = pos_bases[0];
                    h2d(d.pos_ctr, &p0, 4, "pos_ctr");
                    h2d(d.pos_rows, pr.data(), kNlh * 4, "pos_rows");
                }
                // Keep the raw qr_r from BOTH arms (T1 wrote qrA from the fused
                // launch and qrB from mrows), so each chain is fed its OWN
                // projection's bytes -- exactly the wiring R2 vs the verify
                // block differs in.
                const std::vector<float> hqr_a = read_f32(d.qrA, kQl);
                const std::vector<float> hqr_b = read_f32(d.qrB, kQl);
                h2d(d.qrRawA, hqr_a.data(), kQl * 4, "qrRawA<-qrA");
                h2d(d.qrRawB, hqr_b.data(), kQl * 4, "qrRawB<-qrB");
                const bool f1 = ok_rc("T5.end-to-end", "fused lin_rope_norm(qr_r_A)",
                                      dsv41_gemm_fp8_mx_rope_norm(
                                          d.qrRawA, d.qnorm, eps, d.wqb, d.wsqb, nullptr, d.qD,
                                          kNQ, kQl, d.cos, d.sin, d.pos_ctr, 1, 0, 0, 0, kRd, kHd,
                                          nullptr));
                const bool n1 = ok_rc("T5.end-to-end", "verify norm_rows(qr_r_B)",
                                      dsv41_rmsnorm_rows(d.qrRawB, d.qnorm, d.qrRawB, 1, kQl, eps,
                                                         nullptr));
                const bool q1 = ok_rc("T5.end-to-end", "verify quant_rows(qr_r_B)",
                                      dsv41_quant_fp8(d.qrRawB, d.xqR, d.xscR, 1, kQl, 32, 1,
                                                      nullptr));
                const bool m1 = ok_rc("T5.end-to-end", "verify proj_mrows(wq_b)",
                                      dsv41_gemm_fp8_mrows(d.xqR, d.xscR, d.wqb, d.wsqb, nullptr,
                                                           d.qB, 1, kNQ, kQl, kOs, nullptr));
                const bool r1 = ok_rc("T5.end-to-end", "verify apply_rope_mrows",
                                      dsv41_apply_rope_mrows(d.qB, d.cos, d.sin, 1, kNlh, kOs, kHd,
                                                             kRd, kHalf, d.pos_rows, 0, nullptr));
                if (f1 && n1 && q1 && m1 && r1) {
                    printf("  wseed=%d pattern=%s: THE END-TO-END CHAIN (fused vs verify)\n", ws,
                           pattern_name(pat));
                    compare("T5.end-to-end", "q_r (lin2+lin_rope_norm vs the verify chain)",
                            read_f32(d.qD, kNQ), read_f32(d.qB, kNQ), kNQ, R_ALL, true);
                    compare("T5.end-to-end.pre-rope", "q_r pre-rope lanes",
                            read_f32(d.qD, kNQ), read_f32(d.qB, kNQ), kNQ, R_PREROPE, true);
                    compare("T5.end-to-end.rope-lane", "q_r rope lanes",
                            read_f32(d.qD, kNQ), read_f32(d.qB, kNQ), kNQ, R_ROPE, true);
                }
                // coverage: neither arm may have written past n (the mrows form
                // has an explicit out_stride; the fused form does not).
                {
                    std::vector<float> sent(kOs);
                    fill_sentinel(sent);
                    h2d(d.qB, sent.data(), sent.size() * 4, "qB sentinel");
                    h2d(d.qD, sent.data(), sent.size() * 4, "qD sentinel");
                    ok_rc("T5.coverage", "verify proj_mrows(wq_b) re-run",
                          dsv41_gemm_fp8_mrows(d.xqR, d.xscR, d.wqb, d.wsqb, nullptr, d.qB, 1, kNQ,
                                               kQl, kOs, nullptr));
                    ok_rc("T5.coverage", "fused lin_rope_norm re-run",
                          dsv41_gemm_fp8_mx_rope_norm(d.qrRawA, d.qnorm, eps_list[0], d.wqb, d.wsqb,
                                                      nullptr, d.qD, kNQ, kQl, d.cos, d.sin,
                                                      d.pos_ctr, 1, 0, 0, 0, kRd, kHd, nullptr));
                    const auto hb = read_f32(d.qB, kOs);
                    const auto hd = read_f32(d.qD, kOs);
                    const size_t ub = still_sentinel(hb, kNQ, kOs);
                    const size_t ud = still_sentinel(hd, kNQ, kOs);
                    P_CHECK(ub == (size_t)(kOs - kNQ),
                            "T5.coverage: mrows left %zu of the %d tail lanes unwritten", ub,
                            kOs - kNQ);
                    P_CHECK(ud == (size_t)(kOs - kNQ),
                            "T5.coverage: fused left %zu of the %d tail lanes unwritten", ud,
                            kOs - kNQ);
                    if (ub == (size_t)(kOs - kNQ) && ud == (size_t)(kOs - kNQ))
                        printf("  [T5.coverage] OK: both arms wrote exactly [0,%d) and left the "
                               "%d-lane tail untouched\n",
                               kNQ, kOs - kNQ);
                }
            }
        }   // pattern
    }       // wseed

    // ---- summary ---------------------------------------------------------
    printf("\n================= SUMMARY =================\n");
    printf("%-34s %7s %8s %10s %10s %10s %8s\n", "comparison", "cases", "skips", "elems",
           "differing", "bad_cases", "max_ulp");
    for (const auto& a : g_acc) {
        if (a.cases == 0 && a.skips == 0) continue;
        printf("%-34s %7zu %8zu %10zu %10zu %10zu %8llu%s\n", a.tag, a.cases, a.skips, a.elems,
               a.bad_elems, a.bad_cases, (unsigned long long)a.max_ulp,
               a.bad_cases ? "   <<< NOT EQUIVALENT" : "");
    }
    printf("\n");
    // The headline verdict, one line per question.
    auto verdict = [](const char* tag, const char* question) {
        for (const auto& a : g_acc) {
            if (std::strcmp(a.tag, tag) != 0) continue;
            if (a.cases == 0) {
                printf("  ?  %-42s NOT MEASURED (all cases declined)\n", question);
                return;
            }
            if (a.bad_cases == 0)
                printf("  OK %-42s BIT-IDENTICAL over %zu cases / %zu elements\n", question,
                       a.cases, a.elems);
            else
                printf("  !! %-42s NOT EQUIVALENT: %zu/%zu cases, %zu/%zu elements differ "
                       "(max %llu ULP), first bad case #%zu\n",
                       question, a.bad_cases, a.cases, a.bad_elems, a.elems,
                       (unsigned long long)a.max_ulp, a.first_bad_case);
            return;
        }
        printf("  ?  %-42s NOT MEASURED\n", question);
    };
    printf("VERDICT\n");
    verdict("D1.mx-vs-mrows", "CONTROL gemm_fp8_mx(m=1) == mrows(m=1)");
    verdict("D2.mx-vs-mx2", "CONTROL mx2 family1 == mx(m=1)");
    verdict("D3.mx-vs-mx2", "CONTROL mx2 family2 == mx(m=1)");
    verdict("T1.lin2-vs-mrows", "lin2 (mx2) == quant+mrows x2  [wq_a/kv half]");
    verdict("T4.norm", "rmsnorm_q == rmsnorm_rows (+quant)");
    verdict("T3.fused-vs-rmsnorm_q+mxrope", "fused == rmsnorm_q+mx_rope (prologue/epilogue)");
    verdict("T3b.mxrope-vs-mrows", "mx_rope == mrows+apply_rope_mrows");
    verdict("T2.fused-vs-sep", "R2 FUSED wq_b == verify separate chain");
    verdict("T5.end-to-end", "R2 FULL CHAIN == verify FULL CHAIN");

    if (g_skips) printf("\n%u SKIP event(s) -- see the per-tag 'skips' column.\n", (unsigned)g_skips);
    if (g_fails) printf("\n%d HARNESS FAILURE(S).\n", g_fails);
    printf("\n%s\n", (g_diffs == 0 && g_fails == 0)
                          ? "RESULT: every measured fusion is BIT-IDENTICAL to the separate chain."
                          : "RESULT: at least one fusion is NOT bit-identical -- see the DIFF lines "
                            "above.");
    return (g_diffs == 0 && g_fails == 0) ? 0 : 1;
}
