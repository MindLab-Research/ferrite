// gemv_fp8_v2 micro-bench + CORRECTNESS harness (no serve, no weight load).
//
// Why: every attempt to change this kernel in-serve (bf16 MMA gemv, T=4)
// burned ~1h each and was only caught by a 2-minute end-to-end run. This
// harness validates the kernel in seconds: it builds deterministic x/w/scale,
// runs a naive fp32 reference kernel with the SAME e4m3 decode, reports the
// max relative error, then times the real launcher.
//
// Build (remote b300, after `bash build.sh 103a`):
//   nvcc -O3 -arch=sm_103a -o /tmp/gemv_bench gemv_bench.cu \
//        -L. -lferrite_kernels -lcudart
// Run:
//   LD_LIBRARY_PATH=. /tmp/gemv_bench [iters]
//
// Shapes = the real TP8 decode gemvs at n=16 (q_a / kv_a / head).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" {
cudaError_t ferrite_gemv_fp8_v2(const float* x, const void* w,
                                const float* scale, const float* bias,
                                float* out, int in_f, int out_f,
                                int nrows, int srows, int scols,
                                cudaStream_t s);
}

// Naive reference: out[t][r] = sum_k x[t][k] * dequant(w[r][k]) * bias-free.
// One thread per (t, r). Uses the same e4m3 -> half -> float decode as the
// kernel so the only difference is the accumulation order.
__global__ void ref_kernel(const float* x, const unsigned char* w,
                           const float* scale, float* out,
                           int in_f, int out_f, int n, int scols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (r >= out_f || t >= n) return;
    const float* srow = scale + (size_t)(r >> 7) * scols;
    double acc = 0.0;
    for (int k = 0; k < in_f; k++) {
        const float wv = __half2float(__nv_cvt_fp8_to_halfraw(w[(size_t)r * in_f + k], __NV_E4M3))
                         * srow[k >> 7];
        acc += (double)x[(size_t)t * in_f + k] * (double)wv;
    }
    out[(size_t)t * out_f + r] = (float)acc;
}

static float bench(int in_f, int out_f, int n, int iters, float tol) {
    size_t wx = (size_t)out_f * in_f;
    int srows = (out_f + 127) / 128, scols = (in_f + 127) / 128;
    float *x, *out, *ref, *bias, *scale;
    unsigned char* w;
    cudaMalloc(&x, (size_t)n * in_f * 4);
    cudaMalloc(&w, wx);
    cudaMalloc(&scale, (size_t)srows * scols * 4);
    cudaMalloc(&bias, (size_t)out_f * 4);
    cudaMalloc(&out, (size_t)n * out_f * 4);
    cudaMalloc(&ref, (size_t)n * out_f * 4);

    std::vector<float> hx((size_t)n * in_f), hs((size_t)srows * scols), hb(out_f);
    std::vector<unsigned char> hw(wx);
    srand(1234);
    for (auto& v : hx) v = (float)(rand() % 2000 - 1000) / 1000.f;
    for (auto& v : hw) v = (unsigned char)(rand() & 0x7f);   // positive e4m3 range
    for (auto& v : hs) v = 0.002f + 0.001f * (rand() % 5);
    for (auto& v : hb) v = 0.f;
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), wx, cudaMemcpyHostToDevice);
    cudaMemcpy(scale, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);

    dim3 rgrid((out_f + 255) / 256, n);
    ref_kernel<<<rgrid, 256>>>(x, w, scale, ref, in_f, out_f, n, scols);
    cudaError_t e0 = cudaDeviceSynchronize();

    cudaError_t e1 = ferrite_gemv_fp8_v2(x, w, scale, bias, out,
                                         in_f, out_f, n, srows, scols, 0);
    cudaError_t e2 = cudaDeviceSynchronize();
    if (e0 || e1 || e2) {
        printf("  [%5d x %6d n=%2d] LAUNCH ERROR: ref=%s kern=%s sync=%s\n",
               out_f, in_f, n, cudaGetErrorString(e0), cudaGetErrorString(e1),
               cudaGetErrorString(e2));
        return -1.f;
    }

    std::vector<float> ho((size_t)n * out_f), hr((size_t)n * out_f);
    cudaMemcpy(ho.data(), out, ho.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), ref, hr.size() * 4, cudaMemcpyDeviceToHost);
    double maxrel = 0.0;
    int nbad = 0;
    for (size_t i = 0; i < ho.size(); i++) {
        double d = fabs(ho[i] - hr[i]);
        double m = fmax(fabs(hr[i]), 1e-6);
        double rel = d / m;
        if (rel > maxrel) maxrel = rel;
        if (rel > tol) nbad++;
    }

    // timing
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    ferrite_gemv_fp8_v2(x, w, scale, bias, out, in_f, out_f, n, srows, scols, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++)
        ferrite_gemv_fp8_v2(x, w, scale, bias, out, in_f, out_f, n, srows, scols, 0);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    ms /= iters;

    printf("  out=%6d in=%5d n=%2d | %.3f ms/call | maxrel=%.2e bad=%d/%zu %s\n",
           out_f, in_f, n, ms, maxrel, nbad, ho.size(),
           (nbad == 0 ? "OK" : "*** MISMATCH ***"));

    cudaFree(x); cudaFree(w); cudaFree(scale); cudaFree(bias);
    cudaFree(out); cudaFree(ref);
    return ms;
}


// ---- 2-stream overlap probe -------------------------------------------
// The decode chain is ONE stream: independent gemvs (q_a and kv_a both read
// the same normed x) run back-to-back. Both are latency-bound with 23%
// occupancy, so co-scheduling them should cost ~max(t1,t2), not t1+t2.
// This measures exactly that before touching the graph capture.
static void overlap_probe(int iters) {
    const int in_f = 4096, o1 = 1536, o2 = 512, n = 16;
    int srows1 = (o1 + 127) / 128, scols = (in_f + 127) / 128;
    int srows2 = (o2 + 127) / 128;
    float *x, *o1d, *o2d, *b1, *b2, *sc1, *sc2;
    unsigned char *w1, *w2;
    cudaMalloc(&x, (size_t)n * in_f * 4);
    cudaMalloc(&w1, (size_t)o1 * in_f); cudaMalloc(&w2, (size_t)o2 * in_f);
    cudaMalloc(&sc1, (size_t)srows1 * scols * 4); cudaMalloc(&sc2, (size_t)srows2 * scols * 4);
    cudaMalloc(&b1, (size_t)o1 * 4); cudaMalloc(&b2, (size_t)o2 * 4);
    cudaMalloc(&o1d, (size_t)n * o1 * 4); cudaMalloc(&o2d, (size_t)n * o2 * 4);
    cudaMemset(x, 0, (size_t)n * in_f * 4);
    cudaMemset(w1, 1, (size_t)o1 * in_f); cudaMemset(w2, 1, (size_t)o2 * in_f);
    cudaMemset(sc1, 0, (size_t)srows1 * scols * 4); cudaMemset(sc2, 0, (size_t)srows2 * scols * 4);
    cudaMemset(b1, 0, (size_t)o1 * 4); cudaMemset(b2, 0, (size_t)o2 * 4);

    cudaStream_t sA, sB;
    cudaStreamCreate(&sA); cudaStreamCreate(&sB);
    cudaEvent_t a, b, mid;
    cudaEventCreate(&a); cudaEventCreate(&b); cudaEventCreate(&mid);

    // sequential (same stream)
    for (int i = 0; i < 20; i++) {
        ferrite_gemv_fp8_v2(x, w1, sc1, b1, o1d, in_f, o1, n, srows1, scols, 0);
        ferrite_gemv_fp8_v2(x, w2, sc2, b2, o2d, in_f, o2, n, srows2, scols, 0);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) {
        ferrite_gemv_fp8_v2(x, w1, sc1, b1, o1d, in_f, o1, n, srows1, scols, 0);
        ferrite_gemv_fp8_v2(x, w2, sc2, b2, o2d, in_f, o2, n, srows2, scols, 0);
    }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float seq = 0.f; cudaEventElapsedTime(&seq, a, b); seq /= iters;

    // overlapped (two streams; event join before stopping)
    for (int i = 0; i < 20; i++) {
        ferrite_gemv_fp8_v2(x, w1, sc1, b1, o1d, in_f, o1, n, srows1, scols, sA);
        ferrite_gemv_fp8_v2(x, w2, sc2, b2, o2d, in_f, o2, n, srows2, scols, sB);
    }
    cudaStreamSynchronize(sA); cudaStreamSynchronize(sB);
    cudaEventRecord(a, sA);
    for (int i = 0; i < iters; i++) {
        ferrite_gemv_fp8_v2(x, w1, sc1, b1, o1d, in_f, o1, n, srows1, scols, sA);
        ferrite_gemv_fp8_v2(x, w2, sc2, b2, o2d, in_f, o2, n, srows2, scols, sB);
    }
    cudaEventRecord(mid, sA); cudaStreamWaitEvent(sB, mid, 0);
    cudaEventRecord(b, sB); cudaEventSynchronize(b);
    float ovl = 0.f; cudaEventElapsedTime(&ovl, a, b); ovl /= iters;

    printf("\n2-stream overlap probe (q_a 1536 + kv_a 512, both from the same x):\n");
    printf("  sequential (1 stream): %.3f ms/pair\n", seq);
    printf("  overlapped (2 streams): %.3f ms/pair  (%.0f%% of sequential)\n",
           ovl, 100.f * ovl / seq);
    cudaFree(x); cudaFree(w1); cudaFree(w2); cudaFree(sc1); cudaFree(sc2);
    cudaFree(b1); cudaFree(b2); cudaFree(o1d); cudaFree(o2d);
    cudaStreamDestroy(sA); cudaStreamDestroy(sB);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200;
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("gemv_fp8_v2 bench on %s (iters=%d)\n", prop.name, iters);
    // tolerance: the kernel accumulates fp16 chunks (3 extra mantissa bits lost
    // per 16-element chunk) then fp32 -> ~1e-3 relative is expected.
    const float tol = 3e-3f;
    bench(4096, 1536, 16, iters, tol);     // q_a
    bench(4096, 512, 16, iters, tol);      // kv_a
    bench(1536, 64, 16, iters, tol);       // q_b
    bench(4096, 19360, 16, iters, tol);    // lm_head (TP8 shard)
    bench(4096, 1536, 1, iters, tol);      // n=1 sanity
    overlap_probe(iters);
    printf("done\n");
    return 0;
}
