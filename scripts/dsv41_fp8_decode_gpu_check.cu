// Does the bit-composed e4m3_to_f produce the SAME f32 as the exp2f formula on
// the DEVICE? The host-side exhaustive check compared against the exact
// exp2f, but under --use_fast_math the device's exp2f is an approximation, so
// the two implementations can differ in the last bits - which is exactly the
// kind of difference that needs a text A/B before it ships.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ float e4m3_old(uint8_t b) {
    const uint32_t s = (b & 0x80u) ? 0x80000000u : 0u;
    const uint32_t e = (b >> 3) & 0x0Fu;
    const uint32_t m = b & 0x07u;
    if (e == 0) { const float v = (float)m * (1.0f / 512.0f); return s ? -v : v; }
    const float v = (1.0f + (float)m * 0.125f) * exp2f((float)((int)e - 7));
    return s ? -v : v;
}
__device__ __forceinline__ float e4m3_new(uint8_t b) {
    const uint32_t s = ((uint32_t)b & 0x80u) << 24;
    const uint32_t e = ((uint32_t)b >> 3) & 0x0Fu;
    const uint32_t m = (uint32_t)b & 0x07u;
    if (e == 0u) { const float v = (float)m * (1.0f / 512.0f); return (b & 0x80u) ? -v : v; }
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}

__global__ void cmp(const uint8_t* __restrict__ in, float* __restrict__ ov,
                    float* __restrict__ nv, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ov[i] = e4m3_old(in[i]);
    nv[i] = e4m3_new(in[i]);
}

int main() {
    const int n = 256;
    uint8_t* in;
    float *ov, *nv;
    cudaMalloc(&in, n);
    cudaMalloc(&ov, n * 4);
    cudaMalloc(&nv, n * 4);
    uint8_t h_in[256];
    for (int i = 0; i < n; ++i) h_in[i] = (uint8_t)i;
    cudaMemcpy(in, h_in, n, cudaMemcpyHostToDevice);
    cmp<<<1, 256>>>(in, ov, nv, n);
    printf("kernel: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    float h_ov[256], h_nv[256];
    cudaMemcpy(h_ov, ov, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nv, nv, n * 4, cudaMemcpyDeviceToHost);
    int diff = 0;
    for (int i = 0; i < n; ++i) {
        if (i == 0x7F || i == 0xFF) continue;
        if (h_ov[i] != h_nv[i]) {
            if (diff < 10) printf("  b=%02x old=%.9g new=%.9g\n", i, h_ov[i], h_nv[i]);
            ++diff;
        }
    }
    printf("device e4m3 differing codes: %d / 254\n", diff);
    return 0;
}
