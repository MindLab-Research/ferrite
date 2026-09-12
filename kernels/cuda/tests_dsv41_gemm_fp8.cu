// GPU numerics self-test for the dense fp8 GEMM (dsv41_gemm_fp8_mx) and the
// activation quantiser it consumes.
//
// This is the most-used kernel in the model (every attention projection and the
// shared expert go through it), and it had no numerics test until a scale-index
// bug was found incidentally. The reference below is the crate's own semantics:
//   A: [m, k] fp8 e4m3, per-(row, k-block-32) power-of-two scales
//   W: [n, k] fp8 e4m3, per-(n-block-32, k-block-32) ue8m0 scales
//   out[m, n] = sum_k A * W * scale_a[row, kb] * scale_w[n/32, kb]
// (fp32 accumulation per k-block, exactly the reference fp8_gemm_kernel scheme).
//
// Build & run (compile-only + a few hundred KB of GPU memory; no model, no server):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//        -o t_gemm tests_dsv41_gemm_fp8.cu && ./t_gemm
#include "dsv41_kernels.cu"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>

// ---- host-side equivalents of the device helpers (bit-identical) ----
static inline uint32_t h_f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float h_u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline float h_ue8m0_to_f(uint8_t b) {
    return (b == 0xFF) ? NAN : h_u2f((uint32_t)b << 23);
}
static inline float h_fast_round_scale(float amax, float max_inv) {
    const uint32_t bits = h_f2u(amax * max_inv);
    const int e = (int)((bits >> 23) & 0xFFu) - 127 + (((bits & 0x7FFFFFu) != 0) ? 1 : 0);
    return h_u2f((uint32_t)(e + 127) << 23);
}
static inline uint8_t h_ue8m0_byte(float sc) {
    int e = (int)((h_f2u(sc) >> 23) & 0xFFu) - 127;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (uint8_t)(e + 127);
}

static uint8_t e4m3_of(float v) {
    // round-to-nearest-even e4m3 with saturation, mirroring quant.rs
    const float a = fminf(fabsf(v), 448.0f);
    const uint8_t sign = (v < 0.0f) ? 0x80u : 0u;
    if (a == 0.0f) return sign;
    uint8_t e;
    float m;
    if (a < exp2f(-6.0f)) {
        m = roundf(a * 512.0f);
        return sign | (uint8_t)m;
    }
    int ei = (int)floorf(log2f(a));
    float man = (a / exp2f((float)ei) - 1.0f) * 8.0f;
    int mi = (int)lrintf(man);
    if (mi >= 8) { mi = 0; ei += 1; }
    if (ei > 8) return sign | 0x7Eu;
    e = (uint8_t)(ei + 7);
    return sign | (uint8_t)((e << 3) | (mi & 7));
}

static float e4m3_to_f_ref(uint8_t b) {
    const float sgn = (b & 0x80) ? -1.0f : 1.0f;
    const int e = (b >> 3) & 0x0F;
    const int m = b & 7;
    if (e == 0) return sgn * (float)m / 512.0f;
    return sgn * (1.0f + (float)m * 0.125f) * exp2f((float)(e - 7));
}

// CPU reference: block-32 accumulation with the reference's scale scheme.
static void ref_gemm(const std::vector<uint8_t>& a, const std::vector<float>& asc,
                     const std::vector<uint8_t>& w, const std::vector<uint8_t>& wsc,
                     int m, int n, int k, std::vector<float>& out) {
    out.assign((size_t)m * n, 0.0f);
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            float acc = 0.0f;
            for (int kb = 0; kb < k / 32; kb++) {
                float part = 0.0f;
                for (int kk = 0; kk < 32; kk++) {
                    part += e4m3_to_f_ref(a[(size_t)r * k + kb * 32 + kk]) *
                            e4m3_to_f_ref(w[(size_t)c * k + kb * 32 + kk]);
                }
                acc += part * asc[(size_t)r * (k / 32) + kb] *
                       h_ue8m0_to_f(wsc[(size_t)(c / 32) * (k / 32) + kb]);
            }
            out[(size_t)r * n + c] = acc;
        }
    }
}

static int run_case(int m, int n, int k, bool with_bias) {
    std::mt19937 rng(1234u + m * 7 + n * 13 + k);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> ah((size_t)m * k), wh((size_t)n * k);
    for (auto& v : ah) v = dist(rng) * 3.0f;
    for (auto& v : wh) v = dist(rng) * 2.0f;

    // quantise A per (row, 32) with a power-of-two scale (the runtime path)
    std::vector<uint8_t> a((size_t)m * k);
    std::vector<float> asc((size_t)m * (k / 32));
    for (int r = 0; r < m; r++) {
        for (int kb = 0; kb < k / 32; kb++) {
            float amax = 0.0f;
            for (int kk = 0; kk < 32; kk++) amax = fmaxf(amax, fabsf(ah[(size_t)r * k + kb * 32 + kk]));
            const float sc = fmaxf(h_fast_round_scale(amax, 1.0f / 448.0f), 1e-30f);
            asc[(size_t)r * (k / 32) + kb] = sc;
            for (int kk = 0; kk < 32; kk++) {
                float v = fminf(fmaxf(ah[(size_t)r * k + kb * 32 + kk] / sc, -448.0f), 448.0f);
                a[(size_t)r * k + kb * 32 + kk] = e4m3_of(v);
            }
        }
    }
    // quantise W per (32, 32) tile with a ue8m0 scale
    std::vector<uint8_t> w((size_t)n * k);
    std::vector<uint8_t> wsc((size_t)(n / 32) * (k / 32));
    for (int cb = 0; cb < n / 32; cb++) {
        for (int kb = 0; kb < k / 32; kb++) {
            float amax = 0.0f;
            for (int c = 0; c < 32; c++)
                for (int kk = 0; kk < 32; kk++)
                    amax = fmaxf(amax, fabsf(wh[(size_t)(cb * 32 + c) * k + kb * 32 + kk]));
            const float sc = fmaxf(h_fast_round_scale(amax, 1.0f / 448.0f), 1e-30f);
            // ue8m0 byte: exponent + 127 (the weight-scale format)
            wsc[(size_t)cb * (k / 32) + kb] = h_ue8m0_byte(sc);
            for (int c = 0; c < 32; c++)
                for (int kk = 0; kk < 32; kk++) {
                    float v = fminf(fmaxf(wh[(size_t)(cb * 32 + c) * k + kb * 32 + kk] / sc, -448.0f), 448.0f);
                    w[(size_t)(cb * 32 + c) * k + kb * 32 + kk] = e4m3_of(v);
                }
        }
    }

    std::vector<float> bias(n), ref;
    for (int i = 0; i < n; i++) bias[i] = with_bias ? dist(rng) : 0.0f;
    ref_gemm(a, asc, w, wsc, m, n, k, ref);
    if (with_bias)
        for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++) ref[(size_t)r * n + c] += bias[c];

    // ---- GPU ----
    uint8_t *da = nullptr, *dw = nullptr, *dwsc = nullptr;
    float *dasc = nullptr, *dbias = nullptr, *dout = nullptr;
    cudaMalloc(&da, a.size()); cudaMalloc(&dw, w.size()); cudaMalloc(&dwsc, wsc.size());
    cudaMalloc(&dasc, asc.size() * sizeof(float));
    cudaMalloc(&dbias, with_bias ? n * sizeof(float) : 1);
    cudaMalloc(&dout, ref.size() * sizeof(float));
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dwsc, wsc.data(), wsc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dasc, asc.data(), asc.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (with_bias) cudaMemcpy(dbias, bias.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int rc = dsv41_gemm_fp8_mx(da, dasc, dw, dwsc, with_bias ? dbias : nullptr,
                               dout, m, n, k, 0);
    cudaDeviceSynchronize();
    std::vector<float> got(ref.size());
    cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float maxdiff = 0.0f, maxval = 0.0f;
    for (size_t i = 0; i < ref.size(); i++) {
        maxdiff = fmaxf(maxdiff, fabsf(got[i] - ref[i]));
        maxval = fmaxf(maxval, fabsf(ref[i]));
    }
    const float rel = maxval > 0 ? maxdiff / maxval : maxdiff;
    printf("  [gemm m=%3d n=%3d k=%4d bias=%d] rc=%d maxdiff=%.3e rel=%.2e %s\n", m, n, k,
           (int)with_bias, rc, maxdiff, rel, (rel < 5e-3f ? "OK" : "*** MISMATCH ***"));
    cudaFree(da); cudaFree(dw); cudaFree(dwsc); cudaFree(dasc); cudaFree(dbias); cudaFree(dout);
    return rel < 5e-3f ? 0 : 1;
}

// swapAB M=1 GEMV (dsv41_gemm_fp8_swapab): the weight is the MMA's M and the
// activation is the B column 0. Same block-32 scale scheme as ref_gemm, so the
// only tolerated difference is the tensor core's internal summation order --
// hence a relative tolerance, NOT bit equality (see the kernel's note).
static int run_swapab_case(int n, int k, bool with_bias) {
    std::mt19937 rng(4321u + n * 13 + k);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> ah((size_t)k), wh((size_t)n * k);
    for (auto& v : ah) v = dist(rng) * 3.0f;
    for (auto& v : wh) v = dist(rng) * 2.0f;

    // activation: one row, per-32-block power-of-two scale
    std::vector<uint8_t> a((size_t)k);
    std::vector<float> asc((size_t)(k / 32));
    for (int kb = 0; kb < k / 32; kb++) {
        float amax = 0.0f;
        for (int kk = 0; kk < 32; kk++) amax = fmaxf(amax, fabsf(ah[kb * 32 + kk]));
        const float sc = fmaxf(h_fast_round_scale(amax, 1.0f / 448.0f), 1e-30f);
        asc[kb] = sc;
        for (int kk = 0; kk < 32; kk++) {
            const float v = fminf(fmaxf(ah[kb * 32 + kk] / sc, -448.0f), 448.0f);
            a[kb * 32 + kk] = e4m3_of(v);
        }
    }
    // weights: per-(32-row, 32-col) ue8m0 scale. The row-block count is ceil(n/32)
    // -- a warp's 16 rows need only n % 16 == 0, but the last (partial) 32-row
    // block still owns a scale row the kernel indexes.
    const int ncb = (n + 31) / 32;
    std::vector<uint8_t> w((size_t)n * k);
    std::vector<uint8_t> wsc((size_t)ncb * (k / 32));
    for (int cb = 0; cb < ncb; cb++) {
        const int rows = std::min(32, n - cb * 32);
        for (int kb = 0; kb < k / 32; kb++) {
            float amax = 0.0f;
            for (int c = 0; c < rows; c++)
                for (int kk = 0; kk < 32; kk++)
                    amax = fmaxf(amax, fabsf(wh[(size_t)(cb * 32 + c) * k + kb * 32 + kk]));
            const float sc = fmaxf(h_fast_round_scale(amax, 1.0f / 448.0f), 1e-30f);
            wsc[(size_t)cb * (k / 32) + kb] = h_ue8m0_byte(sc);
            for (int c = 0; c < rows; c++)
                for (int kk = 0; kk < 32; kk++) {
                    const float v =
                        fminf(fmaxf(wh[(size_t)(cb * 32 + c) * k + kb * 32 + kk] / sc, -448.0f), 448.0f);
                    w[(size_t)(cb * 32 + c) * k + kb * 32 + kk] = e4m3_of(v);
                }
        }
    }

    std::vector<float> ref((size_t)n, 0.0f), bias((size_t)n, 0.0f);
    for (int c = 0; c < n; c++) {
        float acc = 0.0f;
        for (int kb = 0; kb < k / 32; kb++) {
            float part = 0.0f;
            for (int kk = 0; kk < 32; kk++)
                part += e4m3_to_f_ref(a[kb * 32 + kk]) *
                        e4m3_to_f_ref(w[(size_t)c * k + kb * 32 + kk]);
            acc += part * asc[kb] * h_ue8m0_to_f(wsc[(size_t)(c / 32) * (k / 32) + kb]);
        }
        ref[c] = acc;
    }
    if (with_bias) {
        for (int c = 0; c < n; c++) bias[c] = dist(rng);
        for (int c = 0; c < n; c++) ref[c] += bias[c];
    }

    uint8_t *da = nullptr, *dw = nullptr, *dwsc = nullptr;
    float *dasc = nullptr, *dbias = nullptr, *dout = nullptr;
    cudaMalloc(&da, a.size()); cudaMalloc(&dw, w.size()); cudaMalloc(&dwsc, wsc.size());
    cudaMalloc(&dasc, asc.size() * sizeof(float));
    cudaMalloc(&dbias, with_bias ? (size_t)n * sizeof(float) : 1u);
    cudaMalloc(&dout, ref.size() * sizeof(float));
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dwsc, wsc.data(), wsc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dasc, asc.data(), asc.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (with_bias) cudaMemcpy(dbias, bias.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    int rc = dsv41_gemm_fp8_swapab(da, dasc, dw, dwsc, with_bias ? dbias : nullptr, dout, n, k, 0);
    cudaDeviceSynchronize();
    std::vector<float> got((size_t)n);
    cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float maxdiff = 0.0f, maxval = 0.0f;
    for (int i = 0; i < n; i++) {
        maxdiff = fmaxf(maxdiff, fabsf(got[i] - ref[i]));
        maxval = fmaxf(maxval, fabsf(ref[i]));
    }
    const float rel = maxval > 0 ? maxdiff / maxval : maxdiff;
    printf("  [swapab n=%4d k=%4d bias=%d] rc=%d maxdiff=%.3e rel=%.2e %s\n", n, k, (int)with_bias,
           rc, maxdiff, rel, (rel < 5e-3f ? "OK" : "*** MISMATCH ***"));
    cudaFree(da); cudaFree(dw); cudaFree(dwsc); cudaFree(dasc); cudaFree(dbias); cudaFree(dout);
    return rel < 5e-3f ? 0 : 1;
}

int main() {
    printf("== dsv41 dense fp8 GEMM self-test (dsv41_gemm_fp8_mx) ==\n");
    int bad = 0;
    bad += run_case(16, 64, 64, false);
    bad += run_case(16, 64, 128, true);
    bad += run_case(16, 128, 256, true);
    bad += run_case(32, 64, 512, false);
    bad += run_case(16, 96, 64, true);
    printf("== swapAB M=1 fp8 GEMV self-test (dsv41_gemm_fp8_swapab) ==\n");
    bad += run_swapab_case(32, 32, false);    // exactly one stage, one k block
    bad += run_swapab_case(96, 64, true);     // 3 full weight-scale row blocks
    bad += run_swapab_case(48, 544, false);   // n%32==16 partial block, partial ring stage (544 = 17*32)
    bad += run_swapab_case(256, 512, true);   // full stage, multi-block grid
    printf("RESULT: %s\n", bad ? "FAILURES" : "all cases OK");
    return bad ? 1 : 0;
}
