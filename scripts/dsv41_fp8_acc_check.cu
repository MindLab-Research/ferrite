// Decides the fp8 accumulation question on the device, with the SAME compiler
// and the SAME --use_fast_math as the production build: does
//     acc += A * sa * (B * sb)                      (the baseline loop)
// compile to the same number as
//     acc = __fmaf_rn(__fmul_rn(A,sa), __fmul_rn(B,sb), acc)
// or to something else? Both accumulate over identical inputs in one kernel, so
// no host-side formula can be blamed for a difference.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ float ue8m0_to_f(uint8_t b) {
    return __int_as_float((int)((b == 0xFFu ? 0x7FC00000u : ((uint32_t)b) << 23)));
}
__device__ __forceinline__ float e4m3_to_f(uint8_t b) {
    const uint32_t s = (b & 0x80u) ? 0x80000000u : 0u;
    const uint32_t e = (b >> 3) & 0x0Fu;
    const uint32_t m = b & 0x07u;
    if (e == 0) { const float v = (float)m * (1.0f / 512.0f); return s ? -v : v; }
    const float v = (1.0f + (float)m * 0.125f) * exp2f((float)((int)e - 7));
    return s ? -v : v;
}

__global__ void acc_test(const uint8_t* __restrict__ a, const float* __restrict__ ascale,
                         const uint8_t* __restrict__ w, const uint8_t* __restrict__ wscale,
                         float* __restrict__ out, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    const int nb_k = k >> 5;
    float old_acc = 0.f, new_acc = 0.f, alt_acc = 0.f;
    for (int kb = 0; kb < nb_k; ++kb) {
        const float sb = ue8m0_to_f(wscale[(row >> 5) * nb_k + kb]);
        const float sa = ascale[kb];
        const int j = kb * 32 + lane;
        const float A = e4m3_to_f(a[j]);
        const float B = e4m3_to_f(w[(size_t)row * (size_t)k + j]);
        old_acc += A * sa * (B * sb);                                      // baseline, as written
        new_acc = __fmaf_rn(__fmul_rn(A, sa), __fmul_rn(B, sb), new_acc);  // fma of the two products
        alt_acc = __fadd_rn(alt_acc, __fmul_rn(__fmul_rn(A, sa), __fmul_rn(B, sb)));  // 4 roundings
    }
    for (int o = 16; o > 0; o >>= 1) {
        old_acc += __shfl_xor_sync(0xFFFFFFFFu, old_acc, o);
        new_acc += __shfl_xor_sync(0xFFFFFFFFu, new_acc, o);
        alt_acc += __shfl_xor_sync(0xFFFFFFFFu, alt_acc, o);
    }
    if (lane == 0) {
        out[row] = old_acc;
        out[n + row] = new_acc;
        out[2 * n + row] = alt_acc;
    }
}

int main() {
    const int n = 64, k = 5120;
    const size_t wbytes = (size_t)n * k + (size_t)(n / 32) * (k / 32);
    uint8_t *w, *a;
    float *asc, *out;
    cudaMalloc(&w, wbytes);
    cudaMalloc(&a, k);
    cudaMalloc(&asc, (k / 32) * 4);
    cudaMalloc(&out, 2 * n * 4);
    uint8_t* hw = new uint8_t[wbytes];
    uint8_t* ha = new uint8_t[k];
    float* hasc = new float[k / 32];
    for (size_t i = 0; i < (size_t)n * k; ++i) hw[i] = (uint8_t)((i * 37 + 11) & 0x7Eu);
    for (size_t i = (size_t)n * k; i < wbytes; ++i) hw[i] = 0x7Fu;   // ue8m0 = 1.0
    for (int i = 0; i < k; ++i) ha[i] = (uint8_t)((i * 53 + 7) & 0x7Eu);
    for (int i = 0; i < k / 32; ++i) hasc[i] = 1.0f;
    cudaMemcpy(w, hw, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(a, ha, k, cudaMemcpyHostToDevice);
    cudaMemcpy(asc, hasc, (k / 32) * 4, cudaMemcpyHostToDevice);
    acc_test<<<n, 32>>>(a, asc, w, (const uint8_t*)w + (size_t)n * k, out, n, k);
    cudaError_t e = cudaDeviceSynchronize();
    printf("kernel: %s\n", cudaGetErrorString(e));
    float* ho = new float[2 * n];
    cudaMemcpy(ho, out, 2 * n * 4, cudaMemcpyDeviceToHost);
    int diff = 0;
    for (int i = 0; i < n; ++i) {
        if (ho[i] != ho[n + i]) {
            if (diff < 5) printf("  row %2d: baseline %.9g   fmaf %.9g\n", i, ho[i], ho[n + i]);
            ++diff;
        }
    }
    printf("rows where the two forms differ: %d / %d\n", diff, n);
    printf("sample: baseline %.9g %.9g | fmaf %.9g %.9g\n", ho[0], ho[1], ho[n], ho[n + 1]);
    return 0;
}
