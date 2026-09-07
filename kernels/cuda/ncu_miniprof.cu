// ncu toolchain validation: 3-kernel mini program with cudaProfilerStart/Stop
// window (mirrors ferrite's FERRITE_NCU pattern). Verify:
//   sudo /usr/local/cuda-13.2/bin/ncu --profile-from-start off \
//     --metrics gpu__time_duration.sum --csv --log-file /tmp/mini.csv ./a.out
// CSV must contain exactly the 3 kernels between Start/Stop (the pre-window
// kernel must NOT appear).
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void warmup_kernel(float* p) { p[blockIdx.x] = threadIdx.x; }
__global__ void axpy_kernel(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
__global__ void dot_kernel(const float* x, int n, float* out) {
    __shared__ float s[256];
    s[threadIdx.x] = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        s[threadIdx.x] += x[i];
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = s[threadIdx.x];
}

int main() {
    int n = 1 << 20;
    float *x, *y, *o;
    cudaMalloc(&x, n * 4); cudaMalloc(&y, n * 4); cudaMalloc(&o, 256 * 4);
    axpy_kernel<<<1, 32>>>(y, x, 1.f, 32);           // pre-window (must be skipped)
    cudaDeviceSynchronize();
    cudaProfilerStart();                              // window open
    axpy_kernel<<<4096, 256>>>(y, x, 2.f, n);
    dot_kernel<<<256, 256>>>(x, n, o);
    axpy_kernel<<<2048, 128>>>(y, x, 3.f, n);
    cudaDeviceSynchronize();
    cudaProfilerStop();                               // window close
    printf("mini done\n");
    cudaFree(x); cudaFree(y); cudaFree(o);
    return 0;
}
