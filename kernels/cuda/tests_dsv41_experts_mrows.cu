// tests_dsv41_experts_mrows.cu — the MULTI-ROW (grid.z) acceptance test for the
// batched expert launchers (kernels/cuda/dsv41_experts_mxf4.cu) and the batched
// SwiGLU (kernels/cuda/dsv41_glue.cu).
//
// WHY THIS FILE EXISTS. `rows` used to be validation-only in
//   dsv41_expert_gate_up_fp4_batched / dsv41_expert_down_fp4_batched /
//   dsv41_expert_down_reduce_fp4_batched
// (the grid was output-width based), so every call computed ONE activation row
// and the callers issued one launch per row (the draft/verify MoE loops). `rows`
// is now the launcher's THIRD grid dimension (blockIdx.z), the buffers are
// [rows][slot][...], and this file checks the three things that has to mean:
//
//   1. ROW INDEPENDENCE: row r of a rows=m call is BIT-IDENTICAL to a rows=1 call
//      on the same row. Rows share no output, no accumulator and no smem staging
//      - only base pointers move - so this is exact equality, not a tolerance.
//      Checked over the whole chain gate/up -> (swiglu) -> down(+reduce), in both
//      the FUSED (gate_up+swiglu in the epilogue) and the UNFUSED layout, and for
//      the PLAIN and the INTERLEAVED (DSV41_EXPERT_ILV) weight pools.
//   2. rows=1 IS STILL THE OLD PATH: a rows=1 call (grid.z == 1, every row offset
//      folds to zero) must reproduce an INDEPENDENT CPU golden computed in double
//      over the SAME quantised operands - the check a no-op or decode-broken
//      kernel cannot pass. rows=1 is also run at a production-like slot count.
//   3. the output is actually WRITTEN: the multi-row buffers are pre-filled with a
//      NaN sentinel and every element must be overwritten. The OLD launcher
//      ignored `rows`, i.e. rows 1..m-1 were never touched - the sentinel is what
//      distinguishes "multi-row" from "row 0 only".
//
// NOT checked here: end-to-end model parity (the Rust wiring is a separate
// change) and launch performance.
//
// ---------------------------------------------------------------------------
// Build (needs nvcc, NO GPU). The expert TU is INCLUDED (same style as
// tests_tcgen05_mxf4_gateup.cu); the SwiGLU launcher lives in its own TU and is
// declared extern "C" below, so dsv41_glue.cu is a second input file, and
// dsv41_kernels.cu a third because dsv41_glue.cu's entry points reference the fp8
// quantise kernel (no extern "C" name is defined twice across the three):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//        -o /tmp/t_mrows kernels/cuda/tests_dsv41_experts_mrows.cu \
//        kernels/cuda/dsv41_glue.cu kernels/cuda/dsv41_kernels.cu
// Run (needs ONE free GPU; peak allocation is a few MB):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_mrows            # full suite
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_mrows --quick    # fused arms only
//
// ENV THAT CHANGES WHAT IS COVERED (all read once per process, before main):
//   DSV41_GATEUP_FUSE=0  forces the UNFUSED layout at every shape; the "fused"
//                        arms are then skipped and reported as SKIP (the
//                        interleaved arms cannot run at all: the launcher
//                        rejects ilv+unfused, which is itself checked);
//   DSV41_EXPERT_ILV     is a LOADER decision on the Rust side; this test forms
//                        the interleaved pool itself with the real
//                        dsv41_interleave_gateup_fp4 launcher, so both layouts
//                        are covered regardless;
//   DSV41_GATEUP_ROWS / _KSPLIT / _CPASYNC / _PIPELINE, DSV41_EXPERT_FP4_MODE
//                        and DSV41_DOWN_VEC4 change the kernels' internals. Arm
//                        (1) is invariant under all of them by construction; arm
//                        (2) re-validates whatever they are set to, so the suite
//                        is worth running under a couple of settings.
#include "dsv41_experts_mxf4.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Cross-TU: the batched SwiGLU launcher (kernels/cuda/dsv41_glue.cu). Its kernel
// is the elementwise [row][slot][2*inter] clamp+silu, i.e. exactly the pass the
// UNFUSED expert path runs between gate/up and down.
extern "C" int dsv41_swiglu_limit_batched(float* gate_up, int rows, int inter, float limit,
                                          long slot_stride, int slots, cudaStream_t s);

namespace {

// ---------------------------------------------------------------- test utils
int g_fails = 0;

#define MR_CHECK(expr, fmt, ...)                                                          \
    do {                                                                                  \
        if (!(expr)) {                                                                    \
            printf("    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);        \
            ++g_fails;                                                                    \
        }                                                                                 \
    } while (0)
#define MR_TRY(cond)                       \
    do {                                   \
        if (!(cond)) return 1;             \
    } while (0)

uint32_t g_rng = 20260912u;
uint32_t mr_xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
int mr_irand(int n) { return (int)(mr_xr() % (uint32_t)n); }

// Host mirror of dsv41_e2m1_to_f (device-only in the TU) and of the e8m0 byte
// the kernels turn into a float with `__uint_as_float(((uint32_t)b) << 23)`.
float mr_e2m1_host(uint8_t n) {
    static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float m = mag[n & 7u];
    return (n & 8u) ? -m : m;
}
double mr_e8m0_host(uint8_t b) { return std::ldexp(1.0, (int)b - 127); }

// fp4 nibbles (one per element) -> the kernels' packed bytes: LOW nibble = even
// element, HIGH nibble = odd (dsv41_quant_fp4's packing and the staging decode).
void mr_pack(const std::vector<uint8_t>& nib, std::vector<uint8_t>& packed) {
    packed.assign(nib.size() / 2, 0);
    for (size_t j = 0; j < nib.size(); ++j) {
        if (j & 1u)
            packed[j >> 1] |= (uint8_t)(nib[j] << 4);
        else
            packed[j >> 1] |= (uint8_t)(nib[j] & 0x0Fu);
    }
}

// One expert plane: [rows][cols] fp4 values -> packed [rows][cols/2] bytes plus
// the e8m0 scale plane [rows][cols/32] bytes. All 16 nibble codes are used: the
// joint DECODE (not the quantiser) is what is under test.
void mr_fill_plane(std::vector<uint8_t>& packed, std::vector<uint8_t>& sc, int rows, int cols) {
    std::vector<uint8_t> nib((size_t)rows * (size_t)cols);
    for (size_t j = 0; j < nib.size(); ++j) nib[j] = (uint8_t)(mr_xr() & 0x0Fu);
    mr_pack(nib, packed);
    sc.assign((size_t)rows * (size_t)(cols / 32), 0);
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = (uint8_t)(124 + mr_irand(9));  // 2^-3 .. 2^5
}

// The packed ACTIVATION (the quantiser's native output: [rows][dim/2] bytes and
// [rows][dim/32] f32 scales) - the row-major layout the multi-row pointers assume.
void mr_fill_act(std::vector<uint8_t>& packed, std::vector<float>& asc, int rows, int dim) {
    std::vector<uint8_t> nib((size_t)rows * (size_t)dim);
    for (size_t j = 0; j < nib.size(); ++j) nib[j] = (uint8_t)(mr_xr() & 0x0Fu);
    mr_pack(nib, packed);
    asc.assign((size_t)rows * (size_t)(dim / 32), 0.f);
    for (size_t i = 0; i < asc.size(); ++i) asc[i] = std::ldexp(1.0f, -1 + mr_irand(4));  // 2^-1..2^2
}

// Bitwise comparison of two float buffers (NaN == NaN, -0 != +0: the point is
// that the two paths differ in NOTHING, so any diff is a bug).
bool mr_biteq(const float* a, const float* b, size_t n, const char* tag) {
    uint32_t x, y;
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(&x, a + i, 4);
        std::memcpy(&y, b + i, 4);
        if (x != y) {
            printf("    FAIL [%s] bit diff at %zu: rows=m 0x%08x (%g)  rows=1 0x%08x (%g)\n", tag, i,
                   x, (double)a[i], y, (double)b[i]);
            return false;
        }
    }
    return true;
}

bool mr_any_nan(const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != p[i]) return true;
    return false;
}

// Golden magnitude check. `sum` is the double-accumulated reference and `l1` a
// magnitude proxy for the same element (the sum of |terms| of the dot) - the
// only honest tolerance scale when a random-sign dot cancels.
bool mr_close(const std::vector<double>& sum, const std::vector<double>& l1, const float* got,
              size_t n, const char* tag) {
    size_t shown = 0;
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)got[i] - sum[i]);
        const double tol = 1e-4 * l1[i] + 1e-5;
        if (d > tol) {
            ok = false;
            if (shown++ < 8)
                printf("    FAIL [%s] golden mismatch at %zu: got %g want %g (|d|=%g tol=%g l1=%g)\n",
                       tag, i, (double)got[i], sum[i], d, tol, l1[i]);
        }
    }
    if (!ok) printf("    (%zu element(s) off in the [%s] goldens)\n", shown, tag);
    return ok;
}

// ---------------------------------------------------------------------------
// The launcher's own fuse decision (mirrors dsv41_expert_gate_up_fp4_batched's
// local `g_fuse`): the batched gate/up epilogue writes the swiglu'd [inter] slice
// exactly when this is true, otherwise the full [2*inter] gate|up block.
bool mr_gateup_fused(int dim) {
    const char* e = getenv("DSV41_GATEUP_FUSE");
    const int g_fuse = (e == nullptr) ? 1 : (atoi(e) != 0 ? 1 : 0);
    return g_fuse != 0 && g_expert_fp4_mode == 2 && (dim % 512) == 0;
}

// fp4 dot of plane row `row` against the packed activation `ap`/`asc`, in double:
// the SAME quantised operands the kernel decodes, so the only difference left is
// the fp32 summation order (which the tolerance covers).
double mr_dot(const uint8_t* packed, const uint8_t* sc, int row, int cols, const uint8_t* ap,
              const float* asc, double* l1) {
    const int nb = cols / 32;
    const uint8_t* prow = packed + (size_t)row * (size_t)(cols / 2);
    const uint8_t* srow = sc + (size_t)row * (size_t)nb;
    double s = 0.0, l = 0.0;
    for (int j = 0; j < cols; ++j) {
        const uint8_t ab = ap[j >> 1];
        const double a = (double)mr_e2m1_host((j & 1) ? (uint8_t)(ab >> 4) : (uint8_t)(ab & 0x0Fu)) *
                         (double)asc[j >> 5];
        const uint8_t wb = prow[j >> 1];
        const double w = (double)mr_e2m1_host((j & 1) ? (uint8_t)(wb >> 4) : (uint8_t)(wb & 0x0Fu)) *
                         mr_e8m0_host(srow[j >> 5]);
        s += a * w;
        l += std::fabs(a * w);
    }
    if (l1) *l1 = l;
    return s;
}

// ---------------------------------------------------------------------------
// THE CHAIN CASE: gate/up -> (swiglu) -> down+reduce, rows=m in ONE launch per
// stage, compared bit-for-bit against m runs of the rows=1 shape and checked
// against the CPU golden. Every stage consumes the buffer the previous stage
// wrote, so this also pins the LAYOUT AGREEMENT between the launchers (gate/up's
// out row pitch slots*out_slot_stride == the swiglu/down act row pitch
// slots*act_stride); a disagreement in either pitch breaks the bit-equality arm.
// ---------------------------------------------------------------------------
int mr_case_chain(const char* tag, int dim, int inter, int slots, int rows, int ne, float limit,
                  bool ilv) {
    const bool fused = mr_gateup_fused(dim);
    const int act_slot = fused ? inter : 2 * inter;
    printf("  [%s] dim=%d inter=%d slots=%d rows=%d ne=%d limit=%g ilv=%d fused=%d\n", tag, dim,
           inter, slots, rows, ne, (double)limit, (int)ilv, (int)fused);
    if (ilv && !fused) {
        printf("    SKIP: ilv needs the fused body (DSV41_GATEUP_FUSE=0)\n");
        return 0;
    }

    const int gup_bytes = inter * (dim / 2);   // one expert's w1 (== w3) fp4 plane
    const int gsc_bytes = inter * (dim / 32);  // one expert's w1 e8m0 scale plane
    const int w2_bytes = dim * (inter / 2);
    const int w2s_bytes = dim * (inter / 32);
    const size_t gu_n = (size_t)rows * slots * act_slot;   // [rows][slot][act_slot]
    const size_t dn_n = (size_t)rows * dim;                // [rows][dim]

    // ---- host data (ALL vector declarations stay above the `goto`s below) ----
    std::vector<uint8_t> w1p((size_t)ne * gup_bytes), w1s((size_t)ne * gsc_bytes);
    std::vector<uint8_t> w3p((size_t)ne * gup_bytes), w3s((size_t)ne * gsc_bytes);
    std::vector<uint8_t> w2p((size_t)ne * w2_bytes), w2s((size_t)ne * w2s_bytes);
    std::vector<uint8_t> ap;
    std::vector<float> asc;
    std::vector<int> ids((size_t)rows * slots);
    std::vector<float> rw((size_t)rows * slots);
    std::vector<float> gu_mr(gu_n), gu_1r(gu_n), dn_mr(dn_n), dn_1r(dn_n);
    std::vector<double> gsum(gu_n, 0.0), gl1(gu_n, 0.0), dsum(dn_n, 0.0), dl1(dn_n, 0.0);
    for (int e = 0; e < ne; ++e) {
        std::vector<uint8_t> p, s;
        mr_fill_plane(p, s, inter, dim);
        std::memcpy(w1p.data() + (size_t)e * gup_bytes, p.data(), gup_bytes);
        std::memcpy(w1s.data() + (size_t)e * gsc_bytes, s.data(), gsc_bytes);
        mr_fill_plane(p, s, inter, dim);
        std::memcpy(w3p.data() + (size_t)e * gup_bytes, p.data(), gup_bytes);
        std::memcpy(w3s.data() + (size_t)e * gsc_bytes, s.data(), gsc_bytes);
        mr_fill_plane(p, s, dim, inter);
        std::memcpy(w2p.data() + (size_t)e * w2_bytes, p.data(), w2_bytes);
        std::memcpy(w2s.data() + (size_t)e * w2s_bytes, s.data(), w2s_bytes);
    }
    mr_fill_act(ap, asc, rows, dim);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = mr_irand(ne);
    for (size_t i = 0; i < rw.size(); ++i) rw[i] = (float)(1 + mr_irand(3)) * 0.25f;

    // ---- device ----
    uint8_t* d_w1p = nullptr;
    uint8_t* d_w1s = nullptr;
    uint8_t* d_w3p = nullptr;
    uint8_t* d_w3s = nullptr;
    uint8_t* d_w2p = nullptr;
    uint8_t* d_w2s = nullptr;
    uint8_t* d_ap = nullptr;
    uint8_t* d_ilv = nullptr;
    float* d_asc = nullptr;
    float* d_gu = nullptr;
    float* d_gu1 = nullptr;
    float* d_dn = nullptr;
    float* d_dn1 = nullptr;
    float* d_rw = nullptr;
    int* d_ids = nullptr;
    int rc = 0;
#define MR_MALLOC(p, n)                                             \
    do {                                                            \
        if (cudaMalloc(&(p), (size_t)(n)) != cudaSuccess) {         \
            printf("    FAIL cudaMalloc %s\n", #p);                 \
            rc = 1;                                                 \
            goto done;                                              \
        }                                                           \
    } while (0)
    MR_MALLOC(d_w1p, w1p.size());
    MR_MALLOC(d_w1s, w1s.size());
    MR_MALLOC(d_w3p, w3p.size());
    MR_MALLOC(d_w3s, w3s.size());
    MR_MALLOC(d_w2p, w2p.size());
    MR_MALLOC(d_w2s, w2s.size());
    MR_MALLOC(d_ap, ap.size());
    MR_MALLOC(d_asc, asc.size() * 4);
    MR_MALLOC(d_ids, ids.size() * 4);
    MR_MALLOC(d_rw, rw.size() * 4);
    MR_MALLOC(d_gu, gu_n * 4);
    MR_MALLOC(d_gu1, gu_n * 4);
    MR_MALLOC(d_dn, dn_n * 4);
    MR_MALLOC(d_dn1, dn_n * 4);
    if (ilv) MR_MALLOC(d_ilv, (size_t)ne * 2 * gup_bytes);
#undef MR_MALLOC
    cudaMemcpy(d_w1p, w1p.data(), w1p.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1s, w1s.data(), w1s.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3p, w3p.data(), w3p.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3s, w3s.data(), w3s.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2p, w2p.data(), w2p.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2s, w2s.data(), w2s.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ap, ap.data(), ap.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_asc, asc.data(), asc.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice);
    // Sentinel: every multi-row output element must be overwritten (the old
    // launcher ignored `rows` -> rows 1..m-1 came back untouched).
    cudaMemset(d_gu, 0xFF, gu_n * 4);   // NaN
    cudaMemset(d_dn, 0xFF, dn_n * 4);

    // ILV: the loader builds ONE 8-byte-granule interleaved region per expert with
    // the real launcher; the fused reader derives the up bytes from the gate
    // pointer, so w3's fp4 view is never read. Both arms below see the same pool,
    // so the bit comparison stays meaningful (and the golden decodes the ORIGINAL
    // w1/w3 planes - ILV is a pure permutation).
    const uint8_t* w1_base = d_w1p;
    long w1_stride = gup_bytes;
    const uint8_t* w3_base = d_w3p;
    long w3_stride = gup_bytes;
    if (ilv) {
        for (int e = 0; e < ne; ++e) {
            const int irc = dsv41_interleave_gateup_fp4(
                d_w1p + (size_t)e * gup_bytes, d_w3p + (size_t)e * gup_bytes,
                d_ilv + (size_t)e * 2 * gup_bytes, gup_bytes, 0);
            MR_CHECK(irc == 0, "interleave e=%d rc=%d", e, irc);
        }
        w1_base = d_ilv;
        w1_stride = 2 * (long)gup_bytes;
        w3_base = d_ilv;
        w3_stride = 2 * (long)gup_bytes;
    }

    // ---- arm A: ONE multi-row launch per stage ----
    {
        const int rc1 = dsv41_expert_gate_up_fp4_batched(
            d_ap, d_asc, d_gu, act_slot, rows, dim, inter, limit, slots, w1_base, w1_stride, d_w1s,
            gsc_bytes, w3_base, w3_stride, d_w3s, gsc_bytes, d_ids, ilv ? 1 : 0, 0);
        MR_CHECK(rc1 == 0, "gate_up(rows=%d) rc=%d", rows, rc1);
        if (!fused) {
            const int rc2 =
                dsv41_swiglu_limit_batched(d_gu, rows, inter, limit, (long)act_slot, slots, 0);
            MR_CHECK(rc2 == 0, "swiglu(rows=%d) rc=%d", rows, rc2);
        }
        const int rc3 = dsv41_expert_down_reduce_fp4_batched(d_gu, act_slot, d_dn, rows, dim, inter,
                                                             d_rw, 1, slots, d_w2p, w2_bytes,
                                                             d_w2s, w2s_bytes, d_ids, 0);
        MR_CHECK(rc3 == 0, "down_reduce(rows=%d) rc=%d", rows, rc3);
    }
    // ---- arm B: the same chain, one rows=1 launch per row ----
    for (int r = 0; r < rows; ++r) {
        const uint8_t* a_r = d_ap + (size_t)r * (dim / 2);
        const float* asc_r = d_asc + (size_t)r * (dim / 32);
        float* gu_r = d_gu1 + (size_t)r * slots * act_slot;
        const int* ids_r = d_ids + (size_t)r * slots;
        const float* rw_r = d_rw + (size_t)r * slots;
        float* dn_r = d_dn1 + (size_t)r * dim;
        const int rc1 = dsv41_expert_gate_up_fp4_batched(
            a_r, asc_r, gu_r, act_slot, 1, dim, inter, limit, slots, w1_base, w1_stride, d_w1s,
            gsc_bytes, w3_base, w3_stride, d_w3s, gsc_bytes, ids_r, ilv ? 1 : 0, 0);
        MR_CHECK(rc1 == 0, "gate_up(rows=1,r=%d) rc=%d", r, rc1);
        if (!fused) {
            const int rc2 =
                dsv41_swiglu_limit_batched(gu_r, 1, inter, limit, (long)act_slot, slots, 0);
            MR_CHECK(rc2 == 0, "swiglu(rows=1,r=%d) rc=%d", r, rc2);
        }
        const int rc3 = dsv41_expert_down_reduce_fp4_batched(
            gu_r, act_slot, dn_r, 1, dim, inter, rw_r, 1, slots, d_w2p, w2_bytes, d_w2s, w2s_bytes,
            ids_r, 0);
        MR_CHECK(rc3 == 0, "down_reduce(rows=1,r=%d) rc=%d", r, rc3);
    }
    {
        const cudaError_t ce = cudaDeviceSynchronize();
        MR_CHECK(ce == cudaSuccess, "sync: %s", cudaGetErrorString(ce));
    }
    cudaMemcpy(gu_mr.data(), d_gu, gu_n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gu_1r.data(), d_gu1, gu_n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dn_mr.data(), d_dn, dn_n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dn_1r.data(), d_dn1, dn_n * 4, cudaMemcpyDeviceToHost);

    // (3) the multi-row buffers were really written
    MR_CHECK(!mr_any_nan(gu_mr.data(), gu_n), "gate_up multi-row left the sentinel (rows ignored?)");
    MR_CHECK(!mr_any_nan(dn_mr.data(), dn_n), "down multi-row left the sentinel (rows ignored?)");
    // (1) row r of rows=m == the rows=1 call on row r, bit for bit
    MR_TRY(mr_biteq(gu_mr.data(), gu_1r.data(), gu_n, "gate_up rows=m vs rows=1"));
    MR_TRY(mr_biteq(dn_mr.data(), dn_1r.data(), dn_n, "down rows=m vs rows=1"));

    // (2) CPU golden over the same quantised operands, on the rows=m output -
    //     this is what makes the rows=1 (== old) path independently verified.
    for (int r = 0; r < rows; ++r) {
        const uint8_t* a_r = ap.data() + (size_t)r * (dim / 2);
        const float* asc_r = asc.data() + (size_t)r * (dim / 32);
        for (int s = 0; s < slots; ++s) {
            const int e = ids[(size_t)r * slots + s];
            const uint8_t* w1 = w1p.data() + (size_t)e * gup_bytes;
            const uint8_t* w1sc = w1s.data() + (size_t)e * gsc_bytes;
            const uint8_t* w3 = w3p.data() + (size_t)e * gup_bytes;
            const uint8_t* w3sc = w3s.data() + (size_t)e * gsc_bytes;
            const size_t blk = (size_t)(r * slots + s) * act_slot;
            if (fused) {
                for (int row = 0; row < inter; ++row) {
                    double lg = 0.0, lu = 0.0;
                    const double g = mr_dot(w1, w1sc, row, dim, a_r, asc_r, &lg);
                    const double u = mr_dot(w3, w3sc, row, dim, a_r, asc_r, &lu);
                    const double gc = (limit > 0.f) ? std::fmin(g, (double)limit) : g;
                    const double uc = (limit > 0.f)
                                          ? std::fmin(std::fmax(u, -(double)limit), (double)limit)
                                          : u;
                    gsum[blk + row] = (gc / (1.0 + std::exp(-gc))) * uc;
                    gl1[blk + row] = std::fabs(uc) * (lg + 1.0);   // error bound proxy
                }
            } else {
                for (int row = 0; row < inter; ++row) {
                    double lg = 0.0;
                    const double g = mr_dot(w1, w1sc, row, dim, a_r, asc_r, &lg);
                    gsum[blk + row] =
                        (limit > 0.f) ? std::fmin(g, (double)limit) : g;
                    gl1[blk + row] = lg;
                }
                for (int row = 0; row < inter; ++row) {
                    double lu = 0.0;
                    const double u = mr_dot(w3, w3sc, row, dim, a_r, asc_r, &lu);
                    gsum[blk + inter + row] =
                        (limit > 0.f) ? std::fmin(std::fmax(u, -(double)limit), (double)limit) : u;
                    gl1[blk + inter + row] = lu;
                }
            }
        }
    }
    MR_TRY(mr_close(gsum, gl1, gu_mr.data(), gu_n, "gate_up golden"));

    // down golden: the act is the swiglu'd first `inter` floats of each (row,slot)
    // block (the kernel reads exactly those) and the slot sum is ASCENDING, with
    // the per-slot product rounded before the add - the contract the kernel pins.
    for (int r = 0; r < rows; ++r) {
        for (int s = 0; s < slots; ++s) {
            const int e = ids[(size_t)r * slots + s];
            const float* act_s = gu_mr.data() + (size_t)(r * slots + s) * act_slot;
            const double rwv = (double)rw[(size_t)r * slots + s];
            for (int row = 0; row < dim; ++row) {
                const uint8_t* prow =
                    w2p.data() + (size_t)e * w2_bytes + (size_t)row * (inter / 2);
                const uint8_t* srow =
                    w2s.data() + (size_t)e * w2s_bytes + (size_t)row * (inter / 32);
                double acc = 0.0, l = 0.0;
                for (int j = 0; j < inter; ++j) {
                    const uint8_t wb = prow[j >> 1];
                    const double w = (double)mr_e2m1_host(
                                         (j & 1) ? (uint8_t)(wb >> 4) : (uint8_t)(wb & 0x0Fu)) *
                                     mr_e8m0_host(srow[j >> 5]);
                    acc += (double)act_s[j] * w;
                    l += std::fabs((double)act_s[j] * w);
                }
                dsum[(size_t)r * dim + row] += acc * rwv;
                dl1[(size_t)r * dim + row] += l * std::fabs(rwv);
            }
        }
    }
    MR_TRY(mr_close(dsum, dl1, dn_mr.data(), dn_n, "down golden"));

done:
    if (d_w1p) cudaFree(d_w1p);
    if (d_w1s) cudaFree(d_w1s);
    if (d_w3p) cudaFree(d_w3p);
    if (d_w3s) cudaFree(d_w3s);
    if (d_w2p) cudaFree(d_w2p);
    if (d_w2s) cudaFree(d_w2s);
    if (d_ap) cudaFree(d_ap);
    if (d_asc) cudaFree(d_asc);
    if (d_ids) cudaFree(d_ids);
    if (d_rw) cudaFree(d_rw);
    if (d_gu) cudaFree(d_gu);
    if (d_gu1) cudaFree(d_gu1);
    if (d_dn) cudaFree(d_dn);
    if (d_dn1) cudaFree(d_dn1);
    if (d_ilv) cudaFree(d_ilv);
    return rc;
}

// ---------------------------------------------------------------------------
// The DSV41_DOWN_FUSE=0 fallback: `dsv41_expert_down_fp4_batched` WRITES a
// [rows][slot][dim] scratch (epi_mode 2, per-slot routing weight). Same
// bit-equality contract. (`dsv41_moe_down_reduce` has no `rows` argument at all -
// an un-fused caller keeps one reduce launch per row, which is a wiring fact, not
// a kernel property, so it is out of scope here.)
// ---------------------------------------------------------------------------
int mr_case_down_scratch(const char* tag, int dim, int inter, int slots, int rows, int ne) {
    printf("  [%s] scratch down: dim=%d inter=%d slots=%d rows=%d ne=%d\n", tag, dim, inter, slots,
           rows, ne);
    const int w2_bytes = dim * (inter / 2);
    const int w2s_bytes = dim * (inter / 32);
    const size_t n = (size_t)rows * slots * dim;
    std::vector<uint8_t> w2p((size_t)ne * w2_bytes), w2s((size_t)ne * w2s_bytes);
    std::vector<int> ids((size_t)rows * slots);
    std::vector<float> rw((size_t)rows * slots);
    std::vector<float> act((size_t)rows * slots * inter);
    std::vector<float> o(n), o1(n);
    for (int e = 0; e < ne; ++e) {
        std::vector<uint8_t> p, s;
        mr_fill_plane(p, s, dim, inter);
        std::memcpy(w2p.data() + (size_t)e * w2_bytes, p.data(), w2_bytes);
        std::memcpy(w2s.data() + (size_t)e * w2s_bytes, s.data(), w2s_bytes);
    }
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = mr_irand(ne);
    for (size_t i = 0; i < rw.size(); ++i) rw[i] = (float)(1 + mr_irand(3)) * 0.25f;
    for (size_t i = 0; i < act.size(); ++i) act[i] = (float)(mr_irand(2001) - 1000) / 500.f;

    uint8_t* d_w2p = nullptr;
    uint8_t* d_w2s = nullptr;
    float* d_act = nullptr;
    float* d_rw = nullptr;
    float* d_o = nullptr;
    float* d_o1 = nullptr;
    int* d_ids = nullptr;
    int rc = 0;
    if (cudaMalloc(&d_w2p, w2p.size()) != cudaSuccess || cudaMalloc(&d_w2s, w2s.size()) !=
            cudaSuccess ||
        cudaMalloc(&d_act, act.size() * 4) != cudaSuccess ||
        cudaMalloc(&d_rw, rw.size() * 4) != cudaSuccess ||
        cudaMalloc(&d_ids, ids.size() * 4) != cudaSuccess ||
        cudaMalloc(&d_o, n * 4) != cudaSuccess || cudaMalloc(&d_o1, n * 4) != cudaSuccess) {
        printf("    FAIL cudaMalloc\n");
        rc = 1;
        goto done;
    }
    cudaMemcpy(d_w2p, w2p.data(), w2p.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2s, w2s.data(), w2s.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act, act.data(), act.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0xFF, n * 4);
    {
        const int rc1 = dsv41_expert_down_fp4_batched(d_act, inter, d_o, dim, rows, dim, inter,
                                                      d_rw, 1, slots, d_w2p, w2_bytes, d_w2s,
                                                      w2s_bytes, d_ids, 0);
        MR_CHECK(rc1 == 0, "down_batched(rows=%d) rc=%d", rows, rc1);
        for (int r = 0; r < rows; ++r) {
            const int rc2 = dsv41_expert_down_fp4_batched(
                d_act + (size_t)r * slots * inter, inter, d_o1 + (size_t)r * slots * dim, dim, 1,
                dim, inter, d_rw + (size_t)r * slots, 1, slots, d_w2p, w2_bytes, d_w2s, w2s_bytes,
                d_ids + (size_t)r * slots, 0);
            MR_CHECK(rc2 == 0, "down_batched(rows=1,r=%d) rc=%d", r, rc2);
        }
    }
    {
        const cudaError_t ce = cudaDeviceSynchronize();
        MR_CHECK(ce == cudaSuccess, "sync: %s", cudaGetErrorString(ce));
        cudaMemcpy(o.data(), d_o, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o1.data(), d_o1, n * 4, cudaMemcpyDeviceToHost);
        MR_CHECK(!mr_any_nan(o.data(), n), "down scratch left the sentinel (rows ignored?)");
        MR_TRY(mr_biteq(o.data(), o1.data(), n, "down scratch rows=m vs rows=1"));
    }
done:
    if (d_w2p) cudaFree(d_w2p);
    if (d_w2s) cudaFree(d_w2s);
    if (d_act) cudaFree(d_act);
    if (d_rw) cudaFree(d_rw);
    if (d_ids) cudaFree(d_ids);
    if (d_o) cudaFree(d_o);
    if (d_o1) cudaFree(d_o1);
    return rc;
}

// ---------------------------------------------------------------------------
// The batched SwiGLU alone (grid.z = the activation row, [rows][slot][2*inter]):
// same bit-equality contract plus its own golden, so a failure points at the
// swiglu launcher rather than at the chain above.
// ---------------------------------------------------------------------------
int mr_case_swiglu(const char* tag, int inter, int slots, int rows, float limit) {
    printf("  [%s] swiglu: inter=%d slots=%d rows=%d limit=%g\n", tag, inter, slots, rows,
           (double)limit);
    const size_t n = (size_t)rows * slots * 2 * inter;
    std::vector<float> h(n), a(n), b(n);
    std::vector<double> s(n, 0.0), l(n, 0.0);
    for (size_t i = 0; i < n; ++i) h[i] = (float)(mr_irand(4001) - 2000) / 250.f;  // ±8: clamps fire
    float* d_a = nullptr;
    float* d_b = nullptr;
    int rc = 0;
    if (cudaMalloc(&d_a, n * 4) != cudaSuccess || cudaMalloc(&d_b, n * 4) != cudaSuccess) {
        printf("    FAIL cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(d_a, h.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h.data(), n * 4, cudaMemcpyHostToDevice);
    {
        const int rc1 = dsv41_swiglu_limit_batched(d_a, rows, inter, limit, (long)(2 * inter), slots, 0);
        MR_CHECK(rc1 == 0, "swiglu(rows=%d) rc=%d", rows, rc1);
        for (int r = 0; r < rows; ++r) {
            const int rc2 = dsv41_swiglu_limit_batched(d_b + (size_t)r * slots * 2 * inter, 1, inter,
                                                       limit, (long)(2 * inter), slots, 0);
            MR_CHECK(rc2 == 0, "swiglu(rows=1,r=%d) rc=%d", r, rc2);
        }
        const cudaError_t ce = cudaDeviceSynchronize();
        MR_CHECK(ce == cudaSuccess, "sync: %s", cudaGetErrorString(ce));
        cudaMemcpy(a.data(), d_a, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, n * 4, cudaMemcpyDeviceToHost);
        MR_TRY(mr_biteq(a.data(), b.data(), n, "swiglu rows=m vs rows=1"));
        for (int r = 0; r < rows; ++r) {
            for (int sl = 0; sl < slots; ++sl) {
                const float* base = h.data() + (size_t)(r * slots + sl) * 2 * inter;
                const size_t blk = (size_t)(r * slots + sl) * 2 * inter;
                for (int i = 0; i < inter; ++i) {
                    double g = base[i], u = base[inter + i];
                    if (limit > 0.f) {
                        g = std::fmin(g, (double)limit);
                        u = std::fmin(std::fmax(u, -(double)limit), (double)limit);
                    }
                    s[blk + i] = (g / (1.0 + std::exp(-g))) * u;
                    l[blk + i] = std::fabs(u);   // magnitude proxy
                }
            }
        }
        MR_TRY(mr_close(s, l, a.data(), n, "swiglu golden"));
    }
    cudaFree(d_a);
    cudaFree(d_b);
    return rc;
}

// ---------------------------------------------------------------------------
// Negative control: the ilv + UNFUSED combination must still be REFUSED (the
// unfused body walks the w3 pool separately, so an interleaved pool would be read
// as the wrong bytes - the launcher's loud `cudaErrorInvalidValue`).
// ---------------------------------------------------------------------------
int mr_case_ilv_refused(void) {
    printf("  [ilv-refused] dim=256 (unfused) + ilv=1 must return cudaErrorInvalidValue\n");
    const int dim = 256, inter = 32, slots = 1, rows = 1, ne = 1;
    std::vector<uint8_t> w1p((size_t)inter * (dim / 2)), w1s((size_t)inter * (dim / 32));
    std::vector<uint8_t> w3p((size_t)inter * (dim / 2)), w3s((size_t)inter * (dim / 32));
    std::vector<uint8_t> ap((size_t)dim / 2);
    std::vector<float> asc((size_t)dim / 32, 1.f), out((size_t)inter);
    std::vector<int> ids(slots, 0);
    (void)ne;
    uint8_t* d_w1p = nullptr;
    uint8_t* d_w1s = nullptr;
    uint8_t* d_w3p = nullptr;
    uint8_t* d_w3s = nullptr;
    uint8_t* d_ap = nullptr;
    float* d_asc = nullptr;
    float* d_out = nullptr;
    int* d_ids = nullptr;
    int rc = 0;
    if (cudaMalloc(&d_w1p, w1p.size()) != cudaSuccess ||
        cudaMalloc(&d_w1s, w1s.size()) != cudaSuccess ||
        cudaMalloc(&d_w3p, w3p.size()) != cudaSuccess ||
        cudaMalloc(&d_w3s, w3s.size()) != cudaSuccess ||
        cudaMalloc(&d_ap, ap.size()) != cudaSuccess ||
        cudaMalloc(&d_asc, asc.size() * 4) != cudaSuccess ||
        cudaMalloc(&d_out, out.size() * 4) != cudaSuccess ||
        cudaMalloc(&d_ids, ids.size() * 4) != cudaSuccess) {
        printf("    FAIL cudaMalloc\n");
        return 1;
    }
    const int rc1 = dsv41_expert_gate_up_fp4_batched(d_ap, d_asc, d_out, inter, rows, dim, inter,
                                                     10.f, slots, d_w1p, inter * (dim / 2), d_w1s,
                                                     inter * (dim / 32), d_w3p, inter * (dim / 2),
                                                     d_w3s, inter * (dim / 32), d_ids, 1, 0);
    MR_CHECK(rc1 == (int)cudaErrorInvalidValue, "ilv+unfused rc=%d (want %d)", rc1,
             (int)cudaErrorInvalidValue);
    (void)cudaGetLastError();   // clear the sticky error the refusal may have left
    cudaFree(d_w1p);
    cudaFree(d_w1s);
    cudaFree(d_w3p);
    cudaFree(d_w3s);
    cudaFree(d_ap);
    cudaFree(d_asc);
    cudaFree(d_out);
    cudaFree(d_ids);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    printf("== dsv41 expert multi-row (grid.z) acceptance ==\n");
    printf("   DSV41_EXPERT_FP4_MODE=%d  DSV41_DOWN_VEC4=%d  DSV41_GATEUP_ROWS=%d "
           "DSV41_GATEUP_KSPLIT=%d  fuse(dim=512)=%d\n",
           g_expert_fp4_mode, g_down_fp4_mode, dsv41_gateup_rows(), dsv41_gateup_ksplit(),
           (int)mr_gateup_fused(512));
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("   device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

    // dim % 512 == 0: the FUSED gate/up (the production shape), plain pool.
    g_fails += mr_case_chain("fused/plain", 512, 128, 3, 5, 4, 10.f, false);
    // The INTERLEAVED pool (DSV41_EXPERT_ILV, default ON on the Rust side).
    g_fails += mr_case_chain("fused/ilv", 512, 128, 3, 5, 4, 10.f, true);
    // limit <= 0: the clamp branch is skipped in the kernel AND in the golden.
    g_fails += mr_case_chain("fused/nolimit", 512, 128, 2, 3, 3, 0.f, false);
    if (!quick) {
        // dim % 512 != 0 => the UNFUSED body, with the batched swiglu between
        // gate/up and down (the shape that exercises the separate swiglu pass).
        g_fails += mr_case_chain("unfused/plain", 256, 128, 3, 5, 4, 10.f, false);
        g_fails += mr_case_chain("unfused/small", 256, 64, 2, 2, 2, 10.f, false);
        // rows == 1 IS THE OLD CALL: grid.z == 1, every row offset folds to zero,
        // and the golden must still hold at a production-like slot count.
        g_fails += mr_case_chain("rows=1/old-path", 512, 128, 8, 1, 4, 10.f, false);
        g_fails += mr_case_down_scratch("fallback", 512, 128, 3, 5, 4);
        g_fails += mr_case_swiglu("standalone", 128, 3, 5, 10.f);
        g_fails += mr_case_swiglu("standalone/nolimit", 96, 2, 4, 0.f);
        g_fails += mr_case_ilv_refused();
    }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        printf("  sticky CUDA error after the suite: %s\n", cudaGetErrorString(ce));
        ++g_fails;
    }
    printf(g_fails ? "RESULT: %d check(s) FAILED\n" : "RESULT: all checks passed\n", g_fails);
    return g_fails ? 1 : 0;
}
