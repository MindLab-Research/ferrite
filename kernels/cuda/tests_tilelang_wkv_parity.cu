// tests_tilelang_wkv_parity.cu — TileLang wkv 承接的最小验证台（micro bench，无 e2e）。
//
// 目的（对应 tilelang-integration-design.md §5.2 门 1 / 门 2）：
//   门 1（内部契约，逐位）：同一 TileLang 程序 m=1 的 row 0 vs m=6 的 row 0 -> memcmp 逐位相同；
//                          同一输入重复跑两遍 -> 逐位相同。
//   门 2（跨程序，禁逐位）：TileLang vs 老 `dsv41_gemm_fp8_mrows` 的
//                          maxrel/meanrel 分布（与 f64 真值同口径）+ argmax 一致率。
//   另加：激活只分配 m 行（**不 pad 到 16**）——这是 runtime-m 谓词"不越界读"的硬证据
//        （越界即 fault）。
//
// 编译（远端 B300，sm_103a）：
//   nvcc -arch=sm_103a -O3 -std=c++17 \
//        -I . -I tilelang_inc -I tilelang_gen \
//        -o /tmp/tl_wkv_parity tests_tilelang_wkv_parity.cu
// 运行：/tmp/tl_wkv_parity
//
// 老 kernel 走 `#include "dsv41_kernels.cu"`（与原型 bench_old.cu 同一手法）；
// TileLang 走 `#include "tilelang_gen/wkv_shim.cu"`（真 shim，不是复刻——测的就是接线件）。

#include "dsv41_kernels.cu"
#include "tilelang_gen/wkv_shim.cu"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

// ---- e4m3 / ue8m0 解码（与 ferrite 的 e4m3_to_f 同位模式：s1/e4/m3, bias 7）----
// 用 ldexp 而非 pow：--use_fast_math（.so 的构建口径）会降低 pow 的精度，而参考值必须是
// 二进制精确的 —— 这里的换算全部由 2 的整数次幂构成，ldexp 是精确的。
static double e4m3_to_d(uint8_t b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0xF && m == 0x7) return NAN;  // NaN
    if (e == 0) return (s ? -1.0 : 1.0) * ((double)m / 8.0) * ldexp(1.0, -6);
    return (s ? -1.0 : 1.0) * (1.0 + (double)m / 8.0) * ldexp(1.0, e - 7);
}
static double ue8m0_to_d(uint8_t b) { return ldexp(1.0, (int)b - 127); }

// 确定性伪随机（不用 rand，保证可复现）
static uint32_t g_seed = 12345u;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed; }
static float rndf(float lo, float hi) { return lo + (hi - lo) * ((float)(rnd() >> 8) / (float)(1u << 24)); }

int main() {
    const int N = 512, K = 5120;
    const int NB = N / 32, KB = K / 32;
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");

    // ---- 权重 W[n,k] fp8 + WSC[n/32,k/32] ue8m0 ----
    std::vector<uint8_t> hW((size_t)N * K), hWSC((size_t)NB * KB);
    for (auto& x : hW) x = (uint8_t)(rnd() & 0xFF);          // 全位模式（含 NaN/inf，最严）
    for (auto& x : hWSC) x = (uint8_t)(123 + (rnd() % 8));    // 2^-4 .. 2^0
    uint8_t *W, *WSC;
    ck(cudaMalloc(&W, hW.size()), "W");
    ck(cudaMalloc(&WSC, hWSC.size()), "WSC");
    ck(cudaMemcpy(W, hW.data(), hW.size(), cudaMemcpyHostToDevice), "W h2d");
    ck(cudaMemcpy(WSC, hWSC.data(), hWSC.size(), cudaMemcpyHostToDevice), "WSC h2d");

    // ---- 激活：每个 m 只分配 m 行（不 pad）----
    const int M1 = 1, M6 = 6;
    std::vector<uint8_t> hA1((size_t)M1 * K), hA6((size_t)M6 * K);
    std::vector<float> hAS1((size_t)M1 * KB), hAS6((size_t)M6 * KB);
    for (auto& x : hA1) x = (uint8_t)(rnd() & 0xFF);
    for (auto& x : hA6) x = (uint8_t)(rnd() & 0xFF);
    for (auto& x : hAS1) x = rndf(0.75f, 1.25f);
    for (auto& x : hAS6) x = rndf(0.75f, 1.25f);
    uint8_t *A1, *A6; float *AS1, *AS6;
    ck(cudaMalloc(&A1, hA1.size()), "A1");
    ck(cudaMalloc(&A6, hA6.size()), "A6");
    ck(cudaMalloc(&AS1, hAS1.size() * 4), "AS1");
    ck(cudaMalloc(&AS6, hAS6.size() * 4), "AS6");
    ck(cudaMemcpy(A1, hA1.data(), hA1.size(), cudaMemcpyHostToDevice), "A1 h2d");
    ck(cudaMemcpy(A6, hA6.data(), hA6.size(), cudaMemcpyHostToDevice), "A6 h2d");
    ck(cudaMemcpy(AS1, hAS1.data(), hAS1.size() * 4, cudaMemcpyHostToDevice), "AS1 h2d");
    ck(cudaMemcpy(AS6, hAS6.data(), hAS6.size() * 4, cudaMemcpyHostToDevice), "AS6 h2d");

    float *outTL1, *outTL6, *outOld1, *outOld6, *outTL1b;
    ck(cudaMalloc(&outTL1, (size_t)M1 * N * 4), "oT1");
    ck(cudaMalloc(&outTL6, (size_t)M6 * N * 4), "oT6");
    ck(cudaMalloc(&outOld1, (size_t)M1 * N * 4), "oO1");
    ck(cudaMalloc(&outOld6, (size_t)M6 * N * 4), "oO6");
    ck(cudaMalloc(&outTL1b, (size_t)M1 * N * 4), "oT1b");

    // ---- 发射前先看 rc ----
    int rcTL1 = dsv41_gemm_fp8_tilelang_wkv(A1, AS1, W, WSC, nullptr, outTL1, M1, N, K, N, s);
    int rcTL6 = dsv41_gemm_fp8_tilelang_wkv(A6, AS6, W, WSC, nullptr, outTL6, M6, N, K, N, s);
    printf("shim rc: m=1 -> %d, m=6 -> %d  (0 == launched, 2 == declined)\n", rcTL1, rcTL6);
    if (rcTL1 != 0 || rcTL6 != 0) { printf("SHIM DECLINED — abort\n"); return 2; }
    ck(cudaStreamSynchronize(s), "sync-after-shim");

    int rcO1 = dsv41_gemm_fp8_mrows(A1, AS1, W, WSC, nullptr, outOld1, M1, N, K, N, s);
    int rcO6 = dsv41_gemm_fp8_mrows(A6, AS6, W, WSC, nullptr, outOld6, M6, N, K, N, s);
    printf("old  rc: m=1 -> %d, m=6 -> %d\n", rcO1, rcO6);
    ck(cudaStreamSynchronize(s), "sync-after-old");

    // ---- 门 1：同一 TileLang 程序 m=1/m=6 的 row 0 逐位；重复性 ----
    dsv41_gemm_fp8_tilelang_wkv(A1, AS1, W, WSC, nullptr, outTL1b, M1, N, K, N, s);
    ck(cudaStreamSynchronize(s), "sync-rep");
    std::vector<float> hTL1((size_t)M1 * N), hTL1b((size_t)M1 * N), hTL6((size_t)M6 * N);
    std::vector<float> hO1((size_t)M1 * N), hO6((size_t)M6 * N);
    ck(cudaMemcpy(hTL1.data(), outTL1, hTL1.size() * 4, cudaMemcpyDeviceToHost), "d2h TL1");
    ck(cudaMemcpy(hTL1b.data(), outTL1b, hTL1b.size() * 4, cudaMemcpyDeviceToHost), "d2h TL1b");
    ck(cudaMemcpy(hTL6.data(), outTL6, hTL6.size() * 4, cudaMemcpyDeviceToHost), "d2h TL6");
    ck(cudaMemcpy(hO1.data(), outOld1, hO1.size() * 4, cudaMemcpyDeviceToHost), "d2h O1");
    ck(cudaMemcpy(hO6.data(), outOld6, hO6.size() * 4, cudaMemcpyDeviceToHost), "d2h O6");

    bool tl_rep_ok = (memcmp(hTL1.data(), hTL1b.data(), hTL1.size() * 4) == 0);
    bool tl_m_ok = (memcmp(hTL1.data(), hTL6.data(), hTL1.size() * 4) == 0);  // m=6 的 row0 == m=1 的 row0
    printf("gate1  TileLang repeatability (m=1 twice): %s\n", tl_rep_ok ? "BIT-IDENTICAL" : "DIFFERS");
    printf("gate1  TileLang m=6 row0 == m=1 row0      : %s\n", tl_m_ok ? "BIT-IDENTICAL" : "DIFFERS");

    // ---- 门 2：与 f64 真值比（只统计 |truth| > 0.05*max 的元素）+ argmax 一致率 ----
    // 真值：out[r][j] = Σ_k (a[r][k]*as[r][k/32]) * (w[j][k]*ws[j/32][k/32])，double 累加。
    auto stats = [&](const std::vector<float>& got, const std::vector<uint8_t>& A,
                     const std::vector<float>& AS, int m, const char* tag) {
        double maxrel = 0, sumrel = 0; long cnt = 0, bad = 0;
        for (int r = 0; r < m; ++r) {
            // truth 先算一遍拿到 max|truth|
            std::vector<double> truth(N);
            double tmax = 0;
            for (int j = 0; j < N; ++j) {
                double acc = 0;
                for (int k = 0; k < K; ++k) {
                    double a = e4m3_to_d(A[(size_t)r * K + k]);
                    double w = e4m3_to_d(hW[(size_t)j * K + k]);
                    if (std::isnan(a) || std::isnan(w)) { acc = NAN; break; }
                    acc += a * (double)AS[(size_t)r * KB + k / 32] * w * ue8m0_to_d(hWSC[(size_t)(j / 32) * KB + k / 32]);
                }
                truth[j] = acc;
                if (!std::isnan(acc)) tmax = std::max(tmax, std::fabs(acc));
            }
            for (int j = 0; j < N; ++j) {
                double g = got[(size_t)r * N + j], t = truth[j];
                if (std::isnan(t) || std::fabs(t) <= 0.05 * tmax) continue;
                double rel = std::fabs(g - t) / std::fabs(t);
                maxrel = std::max(maxrel, rel); sumrel += rel; ++cnt;
                if (rel > 1e-2) ++bad;
            }
        }
        printf("gate2  %-22s maxrel=%.3e  meanrel=%.3e  n(rel>1e-2)=%ld / %ld\n",
               tag, maxrel, cnt ? sumrel / cnt : 0.0, bad, cnt);
    };
    stats(hTL1, hA1, hAS1, M1, "TileLang m=1");
    stats(hTL6, hA6, hAS6, M6, "TileLang m=6");
    stats(hO1, hA1, hAS1, M1, "old mrows m=1");
    stats(hO6, hA6, hAS6, M6, "old mrows m=6");

    // argmax 一致率（跨程序；先例 c42ab14：f32 非结合性足以翻转近 tie）
    auto argmax_rate = [&](const std::vector<float>& a, const std::vector<float>& b, int m) {
        long same = 0, tot = 0;
        for (int r = 0; r < m; ++r) {
            int ia = (int)(std::max_element(a.begin() + (size_t)r * N, a.begin() + (size_t)(r + 1) * N) - (a.begin() + (size_t)r * N));
            int ib = (int)(std::max_element(b.begin() + (size_t)r * N, b.begin() + (size_t)(r + 1) * N) - (b.begin() + (size_t)r * N));
            if (ia == ib) ++same;
            ++tot;
        }
        return (double)same / (double)tot;
    };
    printf("gate2  argmax agreement old-vs-TL: m=6 %.3f  (per-row over n=%d)\n",
           argmax_rate(hO6, hTL6, M6), N);

    // ---- 计时（CUDA event；减掉 launch 底噪用空 kernel 校准）----
    auto bench = [&](const char* tag, int m, bool tl) {
        const int IT = 2000;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int i = 0; i < 50; ++i) {  // warmup
            if (tl) dsv41_gemm_fp8_tilelang_wkv(A6, AS6, W, WSC, nullptr, outTL6, m, N, K, N, s);
            else    dsv41_gemm_fp8_mrows(A6, AS6, W, WSC, nullptr, outOld6, m, N, K, N, s);
        }
        cudaStreamSynchronize(s);
        cudaEventRecord(e0, s);
        for (int i = 0; i < IT; ++i) {
            if (tl) dsv41_gemm_fp8_tilelang_wkv(A6, AS6, W, WSC, nullptr, outTL6, m, N, K, N, s);
            else    dsv41_gemm_fp8_mrows(A6, AS6, W, WSC, nullptr, outOld6, m, N, K, N, s);
        }
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        printf("bench  %-22s m=%d  %.2f us/call  (%d iters)\n", tag, m, ms * 1000.0f / IT, IT);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    };
    bench("TileLang wkv (shim)", M1, true);
    bench("TileLang wkv (shim)", M6, true);
    bench("old mrows wkv", M1, false);
    bench("old mrows wkv", M6, false);

    printf("done.\n");
    return 0;
}
