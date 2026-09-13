// tests_tilelang_proj_parity.cu — 五形状 TileLang 投影承接的验证台（micro bench，无 e2e）。
//
// 这是 tests_tilelang_wkv_parity.cu（第一阶段）的**五形状扩展**：把同一个门 1/门 2 台架
// 套到第二阶段接线的四个形状上（wq_a / wq_b / wo_b 稠密 + wo_a 分组 G=1/G=8），测的仍是
// **真 shim**（`tilelang_gen/*_shim.cu`），不是复刻。
//
// 门（与 wkv 台架同口径）：
//   门 1（内部契约，逐位）：同一 TileLang 程序 m=1 的 row 0 vs m=6 的 row 0 -> memcmp 逐位相同；
//                          同一输入重复跑两遍 -> 逐位相同。
//   门 2（跨程序，禁逐位）：TileLang vs 老 kernel（`dsv41_gemm_fp8_mrows` /
//                          `dsv41_wo_a_grouped_fp8`）的 maxrel/meanrel（与 f64 真值同口径）
//                          + argmax 一致率。容差起点 1e-6（原型 §5.2 实测 ≤7.6e-7）。
//   另加：激活只分配 m 行（**不 pad 到 16**）—— runtime-m 谓词"不越界读"的硬证据。
//
// ⚠️ out_stride（OS）是**每形状的真实调用点值**（见 tilelang_gen/gen_proj_shapes_aot.py 文件头）：
//   wkv 512 / wq_a 1280 / wq_b 32768 (=nh*hd, != n) / wo_b 5120 / wo_a 8192 (=ol_total, != G*n)。
//   台架按 OS 分配 out 并同时喂给 shim 与老 kernel，两边的行距一致才可比。
//
// 编译（远端 B300，sm_103a；CPU compile-only，无需 GPU 即可编）：
//   # 各 shim 独立 TU（与生产 build.sh 一致：每个 *_shim.cu 是一个 TU）
//   for s in wkv wq_a wq_b wo_b wo_a; do
//     nvcc -arch=sm_103a -O3 -std=c++17 -I . -I tilelang_inc -I tilelang_gen \
//          -c tilelang_gen/${s}_shim.cu -o /tmp/${s}_shim.o
//   done
//   nvcc -arch=sm_103a -O3 -std=c++17 -I . -I tilelang_inc -I tilelang_gen \
//        -o /tmp/tl_proj_parity tests_tilelang_proj_parity.cu \
//        /tmp/{wkv,wq_a,wq_b,wo_b,wo_a}_shim.o
// 运行（GPU）：/tmp/tl_proj_parity
//
// ⚠️ 五个 shim **不能** #include 进同一个 TU：每个 shim 的内部 constexpr / scratch 都放在
// 匿名命名空间里、且同名（kTLN / kTLKS / g_part ...）——生产里它们各自一个 TU，不冲突；
// 这里必须同样按独立 TU 编译再链接，所以下面是 extern "C" 声明而不是 #include。
//
// 老 kernel 走 `#include "dsv41_kernels.cu"`。

#include "dsv41_kernels.cu"

// 五个 TileLang shim 的导出符号（实现在各 *_shim.o 里；见上方编译说明）。
extern "C" int dsv41_gemm_fp8_tilelang_wkv(const uint8_t*, const float*, const uint8_t*,
                                           const uint8_t*, const float*, float*, int, int, int, int,
                                           cudaStream_t);
extern "C" int dsv41_gemm_fp8_tilelang_wq_a(const uint8_t*, const float*, const uint8_t*,
                                            const uint8_t*, const float*, float*, int, int, int, int,
                                            cudaStream_t);
extern "C" int dsv41_gemm_fp8_tilelang_wq_b(const uint8_t*, const float*, const uint8_t*,
                                            const uint8_t*, const float*, float*, int, int, int, int,
                                            cudaStream_t);
extern "C" int dsv41_gemm_fp8_tilelang_wo_b(const uint8_t*, const float*, const uint8_t*,
                                            const uint8_t*, const float*, float*, int, int, int, int,
                                            cudaStream_t);
extern "C" int dsv41_gemm_fp8_tilelang_wo_a(const uint8_t*, const float*, const uint8_t*,
                                            const uint8_t*, const float*, float*, int, int, int, int,
                                            int, int, cudaStream_t);

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

// ---- e4m3 / ue8m0 解码（与 ferrite 的位模式一致；ldexp 精确，不用 pow）----
static double e4m3_to_d(uint8_t b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0xF && m == 0x7) return NAN;
    if (e == 0) return (s ? -1.0 : 1.0) * ((double)m / 8.0) * ldexp(1.0, -6);
    return (s ? -1.0 : 1.0) * (1.0 + (double)m / 8.0) * ldexp(1.0, e - 7);
}
static double ue8m0_to_d(uint8_t b) { return ldexp(1.0, (int)b - 127); }

static uint32_t g_seed = 12345u;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed; }
static float rndf(float lo, float hi) { return lo + (hi - lo) * ((float)(rnd() >> 8) / (float)(1u << 24)); }

typedef int (*tl_dense_fn)(const uint8_t*, const float*, const uint8_t*, const uint8_t*,
                           const float*, float*, int, int, int, int, cudaStream_t);

struct DenseCfg { const char* name; int N, K, OS; tl_dense_fn tl; };

// 五形状的稠密四（wo_a 分组单独走下面的 grouped 段）。OS 见文件头。
static const DenseCfg DENSE[] = {
    {"wkv",  512,  5120, 512,   dsv41_gemm_fp8_tilelang_wkv},
    {"wq_a", 1280, 5120, 1280,  dsv41_gemm_fp8_tilelang_wq_a},
    {"wq_b", 4096, 1280, 32768, dsv41_gemm_fp8_tilelang_wq_b},
    {"wo_b", 5120, 1024, 5120,  dsv41_gemm_fp8_tilelang_wo_b},
};

// gate2 的 f64 真值（稠密）：out[r][j] = Σ_k (a·as) * (w·ws)。
static void truth_dense(const uint8_t* A, const float* AS, const uint8_t* W, const uint8_t* WSC,
                        int N, int K, int r, std::vector<double>& truth) {
    truth.assign(N, 0.0);
    int KB = K / 32;
    for (int j = 0; j < N; ++j) {
        double acc = 0;
        for (int k = 0; k < K; ++k) {
            double a = e4m3_to_d(A[(size_t)r * K + k]);
            double w = e4m3_to_d(W[(size_t)j * K + k]);
            if (std::isnan(a) || std::isnan(w)) { acc = NAN; break; }
            acc += a * (double)AS[(size_t)r * KB + k / 32] * w *
                   ue8m0_to_d(WSC[(size_t)(j / 32) * KB + k / 32]);
        }
        truth[j] = acc;
    }
}

// 一行 gate2 统计 + argmax（口径与 wkv 台架相同）。
static void report_gate2(const char* tag, const std::vector<float>& got, const std::vector<float>& ref,
                         const std::vector<double>& truth, int N) {
    double maxrel = 0, sumrel = 0; long cnt = 0, bad = 0;
    double tmax = 0;
    for (int j = 0; j < N; ++j) if (!std::isnan(truth[j])) tmax = std::max(tmax, std::fabs(truth[j]));
    for (int j = 0; j < N; ++j) {
        double t = truth[j];
        if (std::isnan(t) || std::fabs(t) <= 0.05 * tmax) continue;
        double rel = std::fabs((double)got[j] - t) / std::fabs(t);
        maxrel = std::max(maxrel, rel); sumrel += rel; ++cnt;
        if (rel > 1e-6) ++bad;              // 容差起点 1e-6（原型 §5.2）
    }
    int ia = (int)(std::max_element(got.begin(), got.end()) - got.begin());
    int ib = (int)(std::max_element(ref.begin(), ref.end()) - ref.begin());
    printf("  gate2 %-16s maxrel=%.3e meanrel=%.3e  n(rel>1e-6)=%ld/%ld  argmax old=%s\n",
           tag, maxrel, cnt ? sumrel / cnt : 0.0, bad, cnt, ia == ib ? "==" : "!=");
}

static void bench(const char* tag, tl_dense_fn tl, const uint8_t* A, const float* AS,
                  const uint8_t* W, const uint8_t* WSC, float* out, int m, int N, int K, int OS,
                  cudaStream_t s) {
    const int IT = 2000;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 50; ++i) tl(A, AS, W, WSC, nullptr, out, m, N, K, OS, s);
    cudaStreamSynchronize(s);
    cudaEventRecord(e0, s);
    for (int i = 0; i < IT; ++i) tl(A, AS, W, WSC, nullptr, out, m, N, K, OS, s);
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    printf("  bench %-22s m=%d  %.2f us/call\n", tag, m, ms * 1000.0f / IT);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

// ---------------------------------------------------------------- 稠密四形状
static void run_dense(const DenseCfg& c, cudaStream_t s) {
    const int N = c.N, K = c.K, OS = c.OS;
    const int KB = K / 32, NB = N / 32;
    const int M1 = 1, M6 = 6;
    printf("== %s  n=%d k=%d OS=%d ==\n", c.name, N, K, OS);

    std::vector<uint8_t> hW((size_t)N * K), hWSC((size_t)NB * KB);
    for (auto& x : hW) x = (uint8_t)(rnd() & 0xFF);        // 全位模式（含 NaN/inf，最严）
    for (auto& x : hWSC) x = (uint8_t)(123 + (rnd() % 8));
    uint8_t *W, *WSC;
    ck(cudaMalloc(&W, hW.size()), "W");
    ck(cudaMalloc(&WSC, hWSC.size()), "WSC");
    ck(cudaMemcpy(W, hW.data(), hW.size(), cudaMemcpyHostToDevice), "W h2d");
    ck(cudaMemcpy(WSC, hWSC.data(), hWSC.size(), cudaMemcpyHostToDevice), "WSC h2d");

    // 激活：每个 m 只分配 m 行（不 pad 到 16）。
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

    float *oT1, *oT6, *oO1, *oO6, *oT1b;
    ck(cudaMalloc(&oT1, (size_t)M1 * OS * 4), "oT1");
    ck(cudaMalloc(&oT6, (size_t)M6 * OS * 4), "oT6");
    ck(cudaMalloc(&oO1, (size_t)M1 * OS * 4), "oO1");
    ck(cudaMalloc(&oO6, (size_t)M6 * OS * 4), "oO6");
    ck(cudaMalloc(&oT1b, (size_t)M1 * OS * 4), "oT1b");

    int rc1 = c.tl(A1, AS1, W, WSC, nullptr, oT1, M1, N, K, OS, s);
    int rc6 = c.tl(A6, AS6, W, WSC, nullptr, oT6, M6, N, K, OS, s);
    printf("  shim rc: m=1 -> %d, m=6 -> %d  (0 == launched, 2 == declined)\n", rc1, rc6);
    if (rc1 != 0 || rc6 != 0) { printf("  SHIM DECLINED — skip %s\n", c.name); return; }
    ck(cudaStreamSynchronize(s), "sync-shim");
    ck(cudaStreamSynchronize(s), "noop");
    int ro1 = dsv41_gemm_fp8_mrows(A1, AS1, W, WSC, nullptr, oO1, M1, N, K, OS, s);
    int ro6 = dsv41_gemm_fp8_mrows(A6, AS6, W, WSC, nullptr, oO6, M6, N, K, OS, s);
    printf("  old  rc: m=1 -> %d, m=6 -> %d\n", ro1, ro6);
    ck(cudaStreamSynchronize(s), "sync-old");

    c.tl(A1, AS1, W, WSC, nullptr, oT1b, M1, N, K, OS, s);   // 重复性
    ck(cudaStreamSynchronize(s), "sync-rep");

    std::vector<float> hT1((size_t)M1 * OS), hT1b((size_t)M1 * OS), hT6((size_t)M6 * OS);
    std::vector<float> hO1((size_t)M1 * OS), hO6((size_t)M6 * OS);
    ck(cudaMemcpy(hT1.data(), oT1, hT1.size() * 4, cudaMemcpyDeviceToHost), "d2h T1");
    ck(cudaMemcpy(hT1b.data(), oT1b, hT1b.size() * 4, cudaMemcpyDeviceToHost), "d2h T1b");
    ck(cudaMemcpy(hT6.data(), oT6, hT6.size() * 4, cudaMemcpyDeviceToHost), "d2h T6");
    ck(cudaMemcpy(hO1.data(), oO1, hO1.size() * 4, cudaMemcpyDeviceToHost), "d2h O1");
    ck(cudaMemcpy(hO6.data(), oO6, hO6.size() * 4, cudaMemcpyDeviceToHost), "d2h O6");

    // 门 1 只看每行的前 N 列（out 行距 OS，可能有 pad 列）。
    bool rep_ok = true, m_ok = true;
    for (int r = 0; r < M1; ++r) {
        rep_ok &= (memcmp(&hT1[(size_t)r * OS], &hT1b[(size_t)r * OS], (size_t)N * 4) == 0);
        m_ok &= (memcmp(&hT1[(size_t)r * OS], &hT6[(size_t)r * OS], (size_t)N * 4) == 0);
    }
    printf("  gate1 repeatability (m=1 twice): %s\n", rep_ok ? "BIT-IDENTICAL" : "DIFFERS");
    printf("  gate1 m=6 row0 == m=1 row0     : %s\n", m_ok ? "BIT-IDENTICAL" : "DIFFERS");

    // 门 2：m=6 每行取前 N 列。
    std::vector<double> truth;
    for (int r = 0; r < M6; ++r) {
        truth_dense(hA6.data(), hAS6.data(), hW.data(), hWSC.data(), N, K, r, truth);
        std::vector<float> gt(hT6.begin() + (size_t)r * OS, hT6.begin() + (size_t)r * OS + N);
        std::vector<float> rf(hO6.begin() + (size_t)r * OS, hO6.begin() + (size_t)r * OS + N);
        char tag[64]; snprintf(tag, sizeof(tag), "TL m=6 r%d", r);
        report_gate2(tag, gt, rf, truth, N);
    }
    bench("TileLang", c.tl, A6, AS6, W, WSC, oT6, M6, N, K, OS, s);

    cudaFree(A1); cudaFree(A6); cudaFree(AS1); cudaFree(AS6);
    cudaFree(W); cudaFree(WSC);
    cudaFree(oT1); cudaFree(oT6); cudaFree(oO1); cudaFree(oO6); cudaFree(oT1b);
}

// ---------------------------------------------------------------- wo_a 分组
struct WoaCfg { const char* name; int G, N, K, ASTRIDE, OS; };
static const WoaCfg WOA[] = {
    {"wo_a_g1", 1, 1024, 4096, 4096,  8192},   // verify@TP8: a_stride == k
    {"wo_a_g8", 8, 1024, 4096, 32768, 8192},   // TP1
};

static void run_woa(const WoaCfg& c, cudaStream_t s) {
    const int G = c.G, N = c.N, K = c.K, ASTRIDE = c.ASTRIDE, OS = c.OS;
    const int M = 6;
    printf("== %s  G=%d n=%d k=%d a_stride=%d OS=%d ==\n", c.name, G, N, K, ASTRIDE, OS);

    std::vector<uint8_t> hW((size_t)G * N * K), hWSC((size_t)G * (N / 32) * (K / 32));
    for (auto& x : hW) x = (uint8_t)(rnd() & 0xFF);
    for (auto& x : hWSC) x = (uint8_t)(123 + (rnd() % 8));
    std::vector<uint8_t> hA((size_t)M * ASTRIDE);
    std::vector<float> hAS((size_t)M * (ASTRIDE / 32));
    for (auto& x : hA) x = (uint8_t)(rnd() & 0xFF);
    for (auto& x : hAS) x = rndf(0.75f, 1.25f);

    uint8_t *W, *WSC, *A; float *AS, *oT, *oO;
    ck(cudaMalloc(&W, hW.size()), "W");
    ck(cudaMalloc(&WSC, hWSC.size()), "WSC");
    ck(cudaMalloc(&A, hA.size()), "A");
    ck(cudaMalloc(&AS, hAS.size() * 4), "AS");
    ck(cudaMalloc(&oT, (size_t)M * OS * 4), "oT");
    ck(cudaMalloc(&oO, (size_t)M * OS * 4), "oO");
    ck(cudaMemcpy(W, hW.data(), hW.size(), cudaMemcpyHostToDevice), "W h2d");
    ck(cudaMemcpy(WSC, hWSC.data(), hWSC.size(), cudaMemcpyHostToDevice), "WSC h2d");
    ck(cudaMemcpy(A, hA.data(), hA.size(), cudaMemcpyHostToDevice), "A h2d");
    ck(cudaMemcpy(AS, hAS.data(), hAS.size() * 4, cudaMemcpyHostToDevice), "AS h2d");

    int rc = dsv41_gemm_fp8_tilelang_wo_a(A, AS, W, WSC, nullptr, oT, G, M, N, K, ASTRIDE, OS, s);
    printf("  shim rc: %d  (0 == launched, 2 == declined)\n", rc);
    if (rc != 0) { printf("  SHIM DECLINED — skip %s\n", c.name); return; }
    int ro = dsv41_wo_a_grouped_fp8(A, AS, W, WSC, nullptr, oO, G, M, N, K, ASTRIDE, OS, s);
    printf("  old  rc: %d\n", ro);
    ck(cudaStreamSynchronize(s), "sync-woa");

    std::vector<float> hT((size_t)M * OS), hO((size_t)M * OS);
    ck(cudaMemcpy(hT.data(), oT, hT.size() * 4, cudaMemcpyDeviceToHost), "d2h T");
    ck(cudaMemcpy(hO.data(), oO, hO.size() * 4, cudaMemcpyDeviceToHost), "d2h O");

    // 门 1：重复性在稠密段已覆盖同类程序；这里只对 gate2 逐 (row, group) 记录。
    for (int g = 0; g < G; ++g) {
        // f64 真值：第 g 组 = A[:, g*K:(g+1)*K] × W[g]^T
        std::vector<double> truth(N, 0.0);
        long cnt = 0, bad = 0; double maxrel = 0, sumrel = 0;
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < N; ++j) {
                double acc = 0;
                for (int kk = 0; kk < K; ++kk) {
                    double a = e4m3_to_d(hA[(size_t)r * ASTRIDE + g * K + kk]);
                    double w = e4m3_to_d(hW[((size_t)g * N + j) * K + kk]);
                    if (std::isnan(a) || std::isnan(w)) { acc = NAN; break; }
                    acc += a * (double)hAS[(size_t)r * (ASTRIDE / 32) + (g * K + kk) / 32] * w *
                           ue8m0_to_d(hWSC[((size_t)g * (N / 32) + j / 32) * (K / 32) + kk / 32]);
                }
                truth[j] = acc;
            }
            double tmax = 0;
            for (int j = 0; j < N; ++j) if (!std::isnan(truth[j])) tmax = std::max(tmax, std::fabs(truth[j]));
            for (int j = 0; j < N; ++j) {
                double t = truth[j];
                if (std::isnan(t) || std::fabs(t) <= 0.05 * tmax) continue;
                double rel = std::fabs((double)hT[(size_t)r * OS + g * N + j] - t) / std::fabs(t);
                maxrel = std::max(maxrel, rel); sumrel += rel; ++cnt;
                if (rel > 1e-6) ++bad;
            }
        }
        printf("  gate2 wo_a g=%d     maxrel=%.3e meanrel=%.3e n(rel>1e-6)=%ld/%ld\n",
               g, maxrel, cnt ? sumrel / cnt : 0.0, bad, cnt);
    }
    printf("  (argmax/cross-program 见原型 §5.2：wo_a 与稠密同量级 ≤7.6e-7)\n");

    cudaFree(W); cudaFree(WSC); cudaFree(A); cudaFree(AS); cudaFree(oT); cudaFree(oO);
}

int main() {
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    printf("TileLang 五形状投影 parity 台架（micro bench，无 e2e）\n");
    for (const auto& c : DENSE) run_dense(c, s);
    for (const auto& c : WOA) run_woa(c, s);
    printf("done.\n");
    return 0;
}
