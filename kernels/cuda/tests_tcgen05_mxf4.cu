// tests_tcgen05_mxf4.cu — minimal numerical self-test for the MXFP4 tcgen05
// expert GEMM (kernels/cuda/dsv41_experts_mxf4.cu).
//
// Small matrices only (a few hundred KB), one kernel launch per case, no model
// data, no serve. The CPU reference uses the quant.rs semantics: e2m1 code
// table + ue8m0 scales, exact arithmetic in double.
//
// Build (on a machine with nvcc, no GPU run needed to build):
//   nvcc -arch=sm_103a -O2 -std=c++17 -o /tmp/t_mxf4 tests_tcgen05_mxf4.cu
// Run (GPU): small allocations only.
//
// Expected result: every case exact (fp4 x fp4 -> fp32 with power-of-two
// block scales is exact arithmetic).
#include "dsv41_experts_mxf4.cu"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

// quant.rs FP4_TABLE, indexed by the 4-bit code (low nibble of a packed byte).
// Host-side table (the kernels read their own copies; nothing here runs on device).
static const float kTab[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                               0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

uint32_t rng_state;
uint32_t xrand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

// e8m0 byte for a power of two 2^e.
uint8_t e8m0_of_exp(int e) { return (uint8_t)(e + 127); }

struct Case {
    int M, N, K;
    std::vector<uint8_t> a_packed;
    std::vector<float> a_scale;
    std::vector<uint8_t> b_packed;
    std::vector<uint8_t> b_scale;
    std::vector<double> expect;
};

Case build_case(int M, int N, int K, uint32_t seed) {
    Case c;
    c.M = M; c.N = N; c.K = K;
    rng_state = seed;
    const int nblk = K / 32;
    c.a_packed.assign((size_t)M * (K / 2), 0);
    c.a_scale.assign((size_t)M * nblk, 1.f);
    c.b_packed.assign((size_t)N * (K / 2), 0);
    c.b_scale.assign((size_t)N * nblk, e8m0_of_exp(0));
    std::vector<double> av((size_t)M * K), bv((size_t)N * K);
    // one scale per 32-element block, codes drawn per element
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            const int blk = k / 32;
            if (k % 32 == 0) c.a_scale[(size_t)m * nblk + blk] = (float)std::ldexp(1.0, (int)(xrand() % 5) - 2);
            const double s = c.a_scale[(size_t)m * nblk + blk];
            const uint8_t code = (uint8_t)(xrand() & 0xF);
            av[(size_t)m * K + k] = kTab[code & 0xF] * s;
            c.a_packed[(size_t)m * (K / 2) + k / 2] |= (uint8_t)((code & 0xF) << (4 * (k & 1)));
        }
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const int blk = k / 32;
            if (k % 32 == 0) {
                const int e = (int)(xrand() % 5) - 2;
                c.b_scale[(size_t)n * nblk + blk] = e8m0_of_exp(e);
            }
            const double s = std::ldexp(1.0, (int)c.b_scale[(size_t)n * nblk + blk] - 127);
            const uint8_t code = (uint8_t)(xrand() & 0xF);
            bv[(size_t)n * K + k] = kTab[code & 0xF] * s;
            c.b_packed[(size_t)n * (K / 2) + k / 2] |= (uint8_t)((code & 0xF) << (4 * (k & 1)));
        }
    c.expect.assign((size_t)M * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += av[(size_t)m * K + k] * bv[(size_t)n * K + k];
            c.expect[(size_t)m * N + n] = acc;
        }
    return c;
}

int run_case(int M, int N, int K, uint32_t seed) {
    const Case c = build_case(M, N, K, seed);
    uint8_t *da = nullptr, *db = nullptr, *dbs = nullptr;
    float *das = nullptr, *dout = nullptr;
    cudaMalloc(&da, c.a_packed.size());
    cudaMalloc(&das, c.a_scale.size() * sizeof(float));
    cudaMalloc(&db, c.b_packed.size());
    cudaMalloc(&dbs, c.b_scale.size());
    cudaMalloc(&dout, c.expect.size() * sizeof(float));
    cudaMemcpy(da, c.a_packed.data(), c.a_packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(das, c.a_scale.data(), c.a_scale.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, c.b_packed.data(), c.b_packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dbs, c.b_scale.data(), c.b_scale.size(), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, c.expect.size() * sizeof(float));

    const int rc = dsv41_mxf4_test_gemm(da, das, db, dbs, dout, M, N, K, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (rc != 0 || err != cudaSuccess) {
        printf("  [M=%3d N=%3d K=%4d] LAUNCH/RUN ERROR rc=%d err=%s\n", M, N, K, rc,
               cudaGetErrorString(err));
        cudaFree(da); cudaFree(das); cudaFree(db); cudaFree(dbs); cudaFree(dout);
        return 1;
    }
    std::vector<float> got(c.expect.size());
    cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double maxdiff = 0.0;
    size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs((double)got[i] - c.expect[i]);
        if (d > maxdiff) maxdiff = d;
        if (d != 0.0 && first_bad == (size_t)-1) first_bad = i;
    }
    printf("  [M=%3d N=%3d K=%4d] maxdiff=%.3e %s", M, N, K, maxdiff,
           first_bad == (size_t)-1 ? "EXACT\n" : "MISMATCH\n");
    if (first_bad != (size_t)-1) {
        const size_t m = first_bad / N, n = first_bad % N;
        printf("      first mismatch at (m=%zu, n=%zu): got %.6f expect %.6f\n", m, n,
               (double)got[first_bad], c.expect[first_bad]);
    }
    cudaFree(da); cudaFree(das); cudaFree(db); cudaFree(dbs); cudaFree(dout);
    return first_bad == (size_t)-1 ? 0 : 1;
}

}  // namespace


// ---------------------------------------------------------------------------
// End-to-end checks of the two production entry points (small shapes).
// ---------------------------------------------------------------------------
uint8_t e2m1_enc_h(float v) {
    const float a = fminf(fabsf(v), 6.0f);
    uint8_t c;
    if (a <= 0.25f) c = 0; else if (a <= 0.75f) c = 1; else if (a <= 1.25f) c = 2;
    else if (a <= 1.75f) c = 3; else if (a <= 2.5f) c = 4; else if (a <= 3.5f) c = 5;
    else if (a <= 5.0f) c = 6; else c = 7;
    return (uint8_t)(c | (v < 0.f ? 8u : 0u));
}
float fast_round_scale6_h(float amax) {
    if (!(amax > 0.f)) return ldexpf(1.f, -126);
    const float r = amax / 6.0f;
    int e; frexpf(r, &e);
    // ceil(log2(r)) = e if r is an exact power of two else e (frexp gives r in [0.5,1))
    // use the same bit trick as the kernel: exponent + (mantissa != 0)
    uint32_t bits; memcpy(&bits, &r, 4);
    int ex = (int)((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) ? 1 : 0);
    if (ex < -126) ex = -126;
    if (ex > 127) ex = 127;
    return ldexpf(1.f, ex);
}
uint8_t f_pow2_ue8m0_h(float s) {
    if (!(s > 0.f)) return 0;
    uint32_t bits; memcpy(&bits, &s, 4);
    int e = (int)((bits >> 23) & 0xFF) - 127;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (uint8_t)(e + 127);
}

int run_gate_up_case() {
    const int R = 128, D = 64, I = 32;  // rows, dim, inter
    std::vector<uint8_t> a((size_t)R * D / 2), w1((size_t)I * D / 2), w3((size_t)I * D / 2);
    std::vector<float> as((size_t)R * D / 32);
    std::vector<uint8_t> w1s((size_t)I * D / 32), w3s((size_t)I * D / 32);
    std::vector<double> av((size_t)R * D), w1v((size_t)I * D), w3v((size_t)I * D);
    rng_state = 11;
    for (int m = 0; m < R; ++m)
        for (int k = 0; k < D; ++k) {
            if (k % 32 == 0) as[(size_t)m * (D / 32) + k / 32] = (float)ldexp(1.0, (int)(xrand() % 5) - 2);
            const double sc = as[(size_t)m * (D / 32) + k / 32];
            const uint8_t code = (uint8_t)(xrand() & 0xF);
            av[(size_t)m * D + k] = kTab[code] * sc;
            a[(size_t)m * (D / 2) + k / 2] |= (uint8_t)((code & 0xF) << (4 * (k & 1)));
        }
    for (int i = 0; i < I; ++i)
        for (int k = 0; k < D; ++k) {
            if (k % 32 == 0) {
                w1s[(size_t)i * (D / 32) + k / 32] = e8m0_of_exp((int)(xrand() % 5) - 2);
                w3s[(size_t)i * (D / 32) + k / 32] = e8m0_of_exp((int)(xrand() % 5) - 2);
            }
            const uint8_t c1 = (uint8_t)(xrand() & 0xF), c3 = (uint8_t)(xrand() & 0xF);
            w1v[(size_t)i * D + k] = kTab[c1] * ldexpf(1.f, (int)w1s[(size_t)i * (D / 32) + k / 32] - 127);
            w3v[(size_t)i * D + k] = kTab[c3] * ldexpf(1.f, (int)w3s[(size_t)i * (D / 32) + k / 32] - 127);
            w1[(size_t)i * (D / 2) + k / 2] |= (uint8_t)((c1 & 0xF) << (4 * (k & 1)));
            w3[(size_t)i * (D / 2) + k / 2] |= (uint8_t)((c3 & 0xF) << (4 * (k & 1)));
        }
    const float limit = 10.f;
    std::vector<double> expect((size_t)R * 2 * I, 0.0);
    for (int m = 0; m < R; ++m)
        for (int i = 0; i < I; ++i) {
            double g = 0, u = 0;
            for (int k = 0; k < D; ++k) { g += av[(size_t)m * D + k] * w1v[(size_t)i * D + k]; u += av[(size_t)m * D + k] * w3v[(size_t)i * D + k]; }
            if (limit > 0) { u = std::min(std::max(u, -(double)limit), (double)limit); g = std::min(g, (double)limit); }
            expect[(size_t)m * 2 * I + i] = g;
            expect[(size_t)m * 2 * I + I + i] = u;
        }
    uint8_t *da, *dw1, *dw3, *dw1s, *dw3s; float *das, *dout;
    cudaMalloc(&da, a.size()); cudaMalloc(&das, as.size() * 4); cudaMalloc(&dw1, w1.size());
    cudaMalloc(&dw3, w3.size()); cudaMalloc(&dw1s, w1s.size()); cudaMalloc(&dw3s, w3s.size());
    cudaMalloc(&dout, expect.size() * 4);
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(das, as.data(), as.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw1, w1.data(), w1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw3, w3.data(), w3.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw1s, w1s.data(), w1s.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw3s, w3s.data(), w3s.size(), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, expect.size() * 4);
    int rc = dsv41_expert_gate_up_fp4(da, das, dw1, dw1s, dw3, dw3s, dout, R, D, I, limit, 0);
    cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> got(expect.size());
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    double md = 0;
    for (size_t i = 0; i < got.size(); ++i) md = std::max(md, std::fabs((double)got[i] - expect[i]));
    printf("  [gate_up R=%d D=%d I=%d] rc=%d err=%s maxdiff=%.3e %s\n", R, D, I, rc,
           cudaGetErrorString(e), md, md == 0 ? "EXACT" : "MISMATCH");
    cudaFree(da); cudaFree(das); cudaFree(dw1); cudaFree(dw3); cudaFree(dw1s); cudaFree(dw3s); cudaFree(dout);
    return md == 0 ? 0 : 1;
}

int run_down_case() {
    const int R = 128, D = 64, I = 64;
    std::vector<float> act((size_t)R * I);
    std::vector<float> weight(R);
    rng_state = 13;
    for (auto& v : act) v = (float)((int)(xrand() % 2001) - 1000) / 100.f;
    for (auto& v : weight) v = (float)(xrand() % 3) * 0.5f;
    std::vector<uint8_t> w2((size_t)D * I / 2);
    std::vector<uint8_t> w2s((size_t)D * I / 32);
    std::vector<double> w2v((size_t)D * I);
    for (int o = 0; o < D; ++o)
        for (int k = 0; k < I; ++k) {
            if (k % 32 == 0) w2s[(size_t)o * (I / 32) + k / 32] = e8m0_of_exp((int)(xrand() % 5) - 2);
            const uint8_t code = (uint8_t)(xrand() & 0xF);
            w2v[(size_t)o * I + k] = kTab[code] * ldexpf(1.f, (int)w2s[(size_t)o * (I / 32) + k / 32] - 127);
            w2[(size_t)o * (I / 2) + k / 2] |= (uint8_t)((code & 0xF) << (4 * (k & 1)));
        }
    // reference: quantise act exactly like the kernel does (fp4, per-32 block,
    // power-of-two scale = fast_round_scale6)
    std::vector<double> actq((size_t)R * I);
    for (int m = 0; m < R; ++m)
        for (int b = 0; b < I / 32; ++b) {
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(act[(size_t)m * I + b * 32 + i]));
            const float sc = fast_round_scale6_h(amax);
            for (int i = 0; i < 32; ++i) {
                const float q = act[(size_t)m * I + b * 32 + i] / sc;
                actq[(size_t)m * I + b * 32 + i] = kTab[e2m1_enc_h(q) & 0xF] * (double)ldexpf(1.f, (int)f_pow2_ue8m0_h(sc) - 127);
            }
        }
    std::vector<double> expect((size_t)R * D, 0.0);
    for (int m = 0; m < R; ++m)
        for (int o = 0; o < D; ++o) {
            double acc = 0;
            for (int k = 0; k < I; ++k) acc += actq[(size_t)m * I + k] * w2v[(size_t)o * I + k];
            expect[(size_t)m * D + o] = acc * weight[m];
        }
    float *dact, *dw, *dout; uint8_t* dw2; uint8_t* dw2s;
    cudaMalloc(&dact, act.size() * 4); cudaMalloc(&dw, R * 4); cudaMalloc(&dout, expect.size() * 4);
    cudaMalloc(&dw2, w2.size()); cudaMalloc(&dw2s, w2s.size());
    cudaMemcpy(dact, act.data(), act.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, weight.data(), R * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw2, w2.data(), w2.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw2s, w2s.data(), w2s.size(), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, expect.size() * 4);
    int rc = dsv41_expert_down_fp4(dact, dw2, dw2s, dw, dout, R, D, I, 0);
    cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> got(expect.size());
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    double md = 0;
    for (size_t i = 0; i < got.size(); ++i) md = std::max(md, std::fabs((double)got[i] - expect[i]));
    printf("  [down R=%d D=%d I=%d] rc=%d err=%s maxdiff=%.3e %s\n", R, D, I, rc,
           cudaGetErrorString(e), md, md < 1e-3 ? "OK" : "MISMATCH");
    cudaFree(dact); cudaFree(dw); cudaFree(dout); cudaFree(dw2); cudaFree(dw2s);
    return md < 1e-3 ? 0 : 1;
}

int main() {
    printf("== dsv41 MXFP4 tcgen05 self-test ==\n");
    int fails = 0;
    fails += run_case(128, 64, 64, 1);    // one atom
    fails += run_case(128, 64, 128, 2);   // two atoms (SF atom placement)
    fails += run_case(64, 32, 64, 3);     // masked M (64 of 128 rows), N < kNTile
    fails += run_case(128, 64, 512, 4);   // 8 atoms, two K stages
    fails += run_case(256, 64, 128, 5);   // two M tiles
    fails += run_gate_up_case();
    fails += run_down_case();
    printf(fails ? "RESULT: %d case(s) FAILED\n" : "RESULT: all cases EXACT\n", fails);
    return fails ? 1 : 0;
}
