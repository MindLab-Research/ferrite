// Isolated repro: dsv41_gemm_fp8_mx at the DSpark main_proj shape —
// m=1, n=5120, k=15360 (the LARGEST k in the whole model; every backbone
// projection is k<=5120). The draft's first gemm is exactly this call, and the
// shadow step dies with cuda error 700 inside dsv41_gemm_fp8_mx.
//
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//            tests_dsv41_gemm_k15360.cu -o /tmp/t_k15360
// Run:   CUDA_VISIBLE_DEVICES=0 /tmp/t_k15360
#include "dsv41_kernels.cu"

#include <cstdio>

int main() {
    printf("gemm_fp8_mx k=15360 repro\n");
    const int m = 1, n = 5120, k = 15360;
    float *a, *out;
    unsigned char *a8, *w8;
    float *asc;
    unsigned char *wsc;
    cudaMallocManaged(&a, (size_t)k * sizeof(float));
    cudaMallocManaged(&a8, (size_t)k);
    cudaMallocManaged(&asc, (size_t)(k / 32 + 8) * sizeof(float));
    cudaMallocManaged(&w8, (size_t)n * k);
    cudaMallocManaged(&wsc, (size_t)(n / 32) * (k / 32));
    cudaMallocManaged(&out, (size_t)n * sizeof(float));
    for (int i = 0; i < k; ++i) a[i] = 0.01f;
    for (int i = 0; i < k; ++i) a8[i] = 100;
    for (int i = 0; i < k / 32 + 8; ++i) asc[i] = 1.0f;
    for (size_t i = 0; i < (size_t)n * k; ++i) w8[i] = 100;
    for (size_t i = 0; i < (size_t)(n / 32) * (k / 32); ++i) wsc[i] = 127;  // ue8m0 = 1.0
    cudaDeviceSynchronize();

    cudaStream_t s;
    cudaStreamCreate(&s);
    int rc = dsv41_gemm_fp8_mx(a8, asc, w8, wsc, nullptr, out, m, n, k, s);
    cudaError_t e = cudaStreamSynchronize(s);
    printf("launcher rc=%d sync=%s\n", rc, cudaGetErrorString(e));
    if (e != cudaSuccess) {
        printf("=> REPRODUCED at k=%d\n", k);
        return 1;
    }
    // also sweep the k where it starts failing
    for (int kk : {5120, 6144, 8192, 10240, 12288}) {
        cudaGetLastError();  // clear sticky
        rc = dsv41_gemm_fp8_mx(a8, asc, w8, wsc, nullptr, out, m, n, kk, s);
        e = cudaStreamSynchronize(s);
        printf("  k=%5d rc=%d sync=%s\n", kk, rc, cudaGetErrorString(e));
        if (e != cudaSuccess) { printf("=> first failure at k=%d\n", kk); return 1; }
    }
    printf("no failure — the k hypothesis is WRONG, look elsewhere\n");
    return 0;
}
