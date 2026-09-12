// tests_mrows_route.cu — the bit-parity acceptance test for B5
// (`ferrite_gemv_bf16_v2_mrows_route`, DSV41_GATE_MROWS_ROUTE, default OFF).
//
// WHY THIS FILE EXISTS. `moe_rows` (chain_dev.rs) computes the block's routing
// with TWO launches: the multi-row gate GEMV (`ferrite_gemv_bf16_v2_mrows`, one
// launch over the m activation rows) and `dsv41_route_topk` (one launch over the
// m score rows). B5 folds them into ONE launch by running the route election in
// the GEMV's LAST block (the "last block" pattern the M=1
// `ferrite_gemv_bf16_v2_route` already uses). The parity claim is
//
//     ONE fused launch (m rows)  ==  mrows-GEMV + route_topk, BIT FOR BIT,
//
// for BOTH halves: every gate score AND every (weight, index) pair. The design
// (`docs/agent/mrows-swallow-batched-implementation-design.md` §3 B5) claims
// this is bit-identical by construction, because the fused entry instantiates
// the SAME `gemv_bf16_nt_kernel<NT, WPR>` program and copies `route_topk_kernel`'s
// body row for row. This suite pins that claim:
//
//  1. SCORES: the fused launch's `out` is bit-identical (raw f32 bits, uint32
//     compare, whole buffer) to the standalone `ferrite_gemv_bf16_v2_mrows`
//     launch of the same arguments. The outputs are pre-filled with a qNaN
//     sentinel, so a kernel that wrote only some rows cannot tie by accident.
//  2. ROUTE: `weights`/`indices` are bit-identical to `dsv41_route_topk`'s, on
//     the whole [m, topk] buffer (indices compared as i32 bits, weights as
//     f32 bits). Also sentinel-filled.
//  3. THE EPILOGUE'S `nrows` DIMENSION: m = 2, 5, 6 are swept. A fused entry
//     that routed only row 0 (the `nrows == 1` shape it forwards on) would pass
//     at m = 1 and fail here.
//  4. BOTH score functions the call site can use: 2 (sqrtsoftplus, production)
//     and 0 (sigmoid), plus a non-null route bias so the `a + bias[e]`
//     selection term is exercised, and topk < n_experts.
//  5. THE TWO WPR DOMAINS below `GEMV_V2_MAX_N` (n < 1024 -> WPR 8,
//     1024 <= n < 4096 -> WPR 4) and a k that is NOT a multiple of 32*8*WPR,
//     so both programs' `kper` rounding is exercised.
//
// Build (needs nvcc, NO GPU; the entries under test live in OTHER TUs, linked as
// extra input files — exactly how tests_gate_mrows.cu builds):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
//        -o /tmp/t_mrows_route kernels/cuda/tests_mrows_route.cu \
//        kernels/cuda/ferrite_kernels.cu kernels/cuda/dsv41_route.cu
// Run (needs ONE free GPU):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_mrows_route
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The two-launch reference pair.
extern "C" cudaError_t ferrite_gemv_bf16_v2_mrows(const float* x, const void* w,
                                                  const float* bias, float* out, int in_f,
                                                  int out_f, int nrows, cudaStream_t s);
extern "C" int dsv41_route_topk(const float* scores, const float* bias, float* weights,
                                int32_t* indices, int32_t* hist, int rows, int n_experts, int topk,
                                int norm_topk_prob, float route_scale, int score_func,
                                cudaStream_t s);
// B5 under test: the SAME GEMV with the route election folded into its last block.
extern "C" cudaError_t ferrite_gemv_bf16_v2_mrows_route(
    const float* x, const void* w, const float* bias, float* out, int in_f, int out_f, int nrows,
    float* weights, int32_t* indices, const float* route_bias, int topk, int norm_topk_prob,
    float route_scale, int score_func, unsigned* ctr, cudaStream_t s);

namespace {

int g_fails = 0;

uint32_t g_rng = 0x9E3779B9u;
uint32_t xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
float frand() { return (float)((int)(xr() % 4001) - 2000) * 1.0e-3f; }

// `n` poison words that no kernel output can equal by accident.
std::vector<float> poison_f32(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
        memcpy(&v[i], &bits, 4);
    }
    return v;
}
std::vector<int32_t> poison_i32(size_t n) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
        memcpy(&v[i], &bits, 4);
    }
    return v;
}
// True when any element still holds its poison word — a vacuous "match".
bool untouched_f32(const std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
        uint32_t got;
        memcpy(&got, &v[i], 4);
        if (got == bits) return true;
    }
    return false;
}
bool untouched_i32(const std::vector<int32_t>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
        uint32_t got;
        memcpy(&got, &v[i], 4);
        if (got == bits) return true;
    }
    return false;
}

void case_fused(int m, int n, int k, int topk, int score_func) {
    const size_t xn = (size_t)m * (size_t)k;
    const size_t sn = (size_t)m * (size_t)n;
    const size_t tn = (size_t)m * (size_t)topk;

    std::vector<__nv_bfloat16> hw((size_t)n * (size_t)k);
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = __float2bfloat16(frand());
    std::vector<float> hx(xn), hb(n);
    for (size_t i = 0; i < xn; ++i) hx[i] = frand();
    for (int i = 0; i < n; ++i) hb[i] = frand();   // route bias (selection term)
    const std::vector<float> poison_s = poison_f32(sn);
    const std::vector<float> poison_w = poison_f32(tn);
    const std::vector<int32_t> poison_i = poison_i32(tn);

    __nv_bfloat16* dw = nullptr;
    float *dx = nullptr, *db = nullptr;
    float *scores_a = nullptr, *scores_b = nullptr;
    float *w_a = nullptr, *w_b = nullptr;
    int32_t *i_a = nullptr, *i_b = nullptr;
    unsigned* ctr = nullptr;
    cudaMalloc(&dw, hw.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&dx, xn * sizeof(float));
    cudaMalloc(&db, (size_t)n * sizeof(float));
    cudaMalloc(&scores_a, sn * sizeof(float));
    cudaMalloc(&scores_b, sn * sizeof(float));
    cudaMalloc(&w_a, tn * sizeof(float));
    cudaMalloc(&w_b, tn * sizeof(float));
    cudaMalloc(&i_a, tn * sizeof(int32_t));
    cudaMalloc(&i_b, tn * sizeof(int32_t));
    cudaMalloc(&ctr, sizeof(unsigned));
    cudaMemcpy(dw, hw.data(), hw.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx.data(), xn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(scores_a, poison_s.data(), sn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(scores_b, poison_s.data(), sn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_a, poison_w.data(), tn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_b, poison_w.data(), tn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(i_a, poison_i.data(), tn * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(i_b, poison_i.data(), tn * sizeof(int32_t), cudaMemcpyHostToDevice);
    // The counter is zeroed ONCE (the kernel self-resets it, which is what makes
    // a captured graph replay clean). It is shared by every launch below.
    cudaMemset(ctr, 0, sizeof(unsigned));

    // ---- arm A: the two-launch reference (`gemv_bf16_v2_mrows` + `route_topk`) ----
    const cudaError_t rc_g =
        ferrite_gemv_bf16_v2_mrows(dx, dw, /*bias=*/nullptr, scores_a, k, n, m, /*stream=*/0);
    if (rc_g != cudaSuccess) {
        printf("    FAIL gemv_bf16_v2_mrows rc=%d\n", (int)rc_g);
        ++g_fails;
    }
    const int rc_r = dsv41_route_topk(scores_a, db, w_a, i_a, /*hist=*/nullptr, m, n, topk,
                                      /*norm_topk_prob=*/1, /*route_scale=*/2.5f, score_func,
                                      /*stream=*/0);
    if (rc_r != 0) {
        printf("    FAIL route_topk rc=%d\n", rc_r);
        ++g_fails;
    }

    // ---- arm B: ONE fused launch (B5) ----
    const cudaError_t rc_f = ferrite_gemv_bf16_v2_mrows_route(
        dx, dw, /*bias=*/nullptr, scores_b, k, n, m, w_b, i_b, db, topk, /*norm_topk_prob=*/1,
        /*route_scale=*/2.5f, score_func, ctr, /*stream=*/0);
    if (rc_f != cudaSuccess) {
        printf("    FAIL mrows_route rc=%d\n", (int)rc_f);
        ++g_fails;
    }

    std::vector<float> sa(sn), sb(sn), wa(tn), wb(tn);
    std::vector<int32_t> ia(tn), ib(tn);
    cudaMemcpy(sa.data(), scores_a, sn * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sb.data(), scores_b, sn * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wa.data(), w_a, tn * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wb.data(), w_b, tn * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ia.data(), i_a, tn * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ib.data(), i_b, tn * sizeof(int32_t), cudaMemcpyDeviceToHost);

    const bool scores_eq = memcmp(sa.data(), sb.data(), sn * sizeof(float)) == 0;
    const bool w_eq = memcmp(wa.data(), wb.data(), tn * sizeof(float)) == 0;
    const bool i_eq = memcmp(ia.data(), ib.data(), tn * sizeof(int32_t)) == 0;
    // Coverage: neither buffer may still hold its sentinel (a "pass" on
    // untouched memory would be vacuous).
    const bool covered = !untouched_f32(sb) && !untouched_f32(wb) && !untouched_i32(ib);

    if (scores_eq && w_eq && i_eq && covered) {
        printf("  ok   m=%d n=%d k=%d topk=%d sf=%d (WPR=%d)\n", m, n, k, topk, score_func,
               n >= 1024 ? 4 : 8);
    } else {
        ++g_fails;
        printf("  FAIL m=%d n=%d k=%d topk=%d sf=%d (WPR=%d) scores=%d w=%d i=%d covered=%d\n", m,
               n, k, topk, score_func, n >= 1024 ? 4 : 8, (int)scores_eq, (int)w_eq, (int)i_eq,
               (int)covered);
        int shown = 0;
        for (size_t x = 0; x < sn && shown < 4; ++x) {
            uint32_t ua, ub;
            memcpy(&ua, &sa[x], 4);
            memcpy(&ub, &sb[x], 4);
            if (ua != ub) {
                printf("       scores i=%zu (row %zu col %zu) A=%g/%08x B=%g/%08x\n", x, x / n,
                       x % n, sa[x], ua, sb[x], ub);
                ++shown;
            }
        }
        for (size_t x = 0; x < tn && shown < 8; ++x) {
            if (ia[x] != ib[x] || memcmp(&wa[x], &wb[x], 4) != 0) {
                printf("       route i=%zu (row %zu slot %zu) A=(%d,%g) B=(%d,%g)\n", x, x / topk,
                       x % topk, ia[x], wa[x], ib[x], wb[x]);
                ++shown;
            }
        }
    }

    cudaFree(dw);
    cudaFree(dx);
    cudaFree(db);
    cudaFree(scores_a);
    cudaFree(scores_b);
    cudaFree(w_a);
    cudaFree(w_b);
    cudaFree(i_a);
    cudaFree(i_b);
    cudaFree(ctr);
}

}  // namespace

int main() {
    printf("B5 mrows-route parity (fused(nrows=m) == mrows-GEMV + route_topk, bit for bit)\n");
    // The production gate: n = the routed expert count (WPR = 8), k = dim.
    case_fused(6, 384, 5120, 6, 2);
    case_fused(5, 384, 5120, 6, 2);
    case_fused(2, 384, 5120, 6, 2);
    // sigmoid selection, and a topk that is not the production 6
    case_fused(6, 384, 5120, 6, 0);
    case_fused(6, 384, 5120, 8, 2);
    // WPR = 4: still below the Rust fold bound (n < 2048)
    case_fused(6, 1024, 5120, 6, 2);
    // k NOT a multiple of 32*8*WPR: each warp takes a partial K-slice
    case_fused(6, 384, 520, 6, 2);
    case_fused(6, 384, 64, 6, 2);
    case_fused(6, 1024, 520, 6, 2);

    if (g_fails == 0) {
        printf("all cases passed\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fails);
    return 1;
}
