// Isolated timing for the two GEMV families the DSV4.1 decode step spends most
// of its time in: the bf16 GEMV (MoE gate at 384 rows, the full-vocab lm_head at
// 129280 rows) and the fp8 GEMV (wq_b-shaped 1024 rows, wq_a+wkv-shaped 1664).
// Linked against the production .so, medians after warmup, one shape per line.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

extern "C" int dsv41_gemv_bf16(const void* w, const float* x, float* out, int n, int k,
                               cudaStream_t s);
extern "C" int dsv41_gemm_fp8_mx(const uint8_t* a, const float* a_scale, const uint8_t* w,
                                 const uint8_t* w_scale, const float* bias, float* out, int m,
                                 int n, int k, cudaStream_t s);

static int g_n = 0, g_k = 0;

static void run_b(void* w, const void* x, void* out) {
    dsv41_gemv_bf16(w, (const float*)x, (float*)out, g_n, g_k, 0);
}
// fp8: w = weights (n*k bytes) immediately followed by the per-32-row scale
// blocks ((n/32)*(k/32) bytes); x = fp8 activation (k bytes) over k/32 f32
// scales.
static void run_f(void* w, const void* x, void* out) {
    const uint8_t* a = (const uint8_t*)x;
    const float* asc = (const float*)(a + g_k);
    const uint8_t* wq = (const uint8_t*)w;
    const uint8_t* wsc = wq + (size_t)g_n * (size_t)g_k;
    dsv41_gemm_fp8_mx(a, asc, wq, wsc, nullptr, (float*)out, 1, g_n, g_k, 0);
}

static void bench(const char* tag, int n, int k, int fp8, int reps) {
    g_n = n; g_k = k;
    size_t wbytes = fp8 ? (size_t)n * k + (size_t)(n / 32) * (k / 32) : (size_t)n * k * 2;
    void* w = nullptr;
    if (cudaMalloc(&w, wbytes) != cudaSuccess) { printf("  %s: malloc %zu failed\n", tag, wbytes); return; }
    cudaMemset(w, 0x3c, wbytes);
    void* x = nullptr;
    cudaMalloc(&x, (size_t)k * 4 + 256);
    cudaMemset(x, 0, (size_t)k * 4 + 256);
    void* out = nullptr;
    cudaMalloc(&out, (size_t)n * 4);
    void (*run)(void*, const void*, void*) = fp8 ? run_f : run_b;
    for (int i = 0; i < 5; ++i) run(w, x, out);
    cudaDeviceSynchronize();
    std::vector<float> ts;
    for (int i = 0; i < reps; ++i) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, 0);
        run(w, x, out);
        cudaEventRecord(b, 0);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        ts.push_back(ms);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    std::sort(ts.begin(), ts.end());
    printf("  %-18s n=%-7d k=%d  w=%7.2fMB  median %8.2f us  (min %8.2f)\n", tag, n, k,
           wbytes / 1048576.0, ts[ts.size() / 2] * 1000.0, ts.front() * 1000.0);
    cudaFree(w); cudaFree(x); cudaFree(out);
}

int main() {
    cudaFree(0);
    int dev = 0;
    cudaGetDevice(&dev);
    printf("GEMV isolation (device %d)\n", dev);
    bench("bf16 gate", 384, 5120, 0, 200);
    bench("bf16 gate(v2)", 384, 5120, 0, 200);
    bench("bf16 lm_head(full)", 129280, 5120, 0, 30);
    bench("bf16 lm_head(/8)", 129280 / 8, 5120, 0, 200);
    bench("fp8 wq_b 1024", 1024, 5120, 1, 200);
    bench("fp8 wq_a+wkv", 1664, 5120, 1, 200);
    bench("fp8 sharedexp", 256, 5120, 1, 200);
    return 0;
}
