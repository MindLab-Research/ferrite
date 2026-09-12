#include "dsv41_kernels.cu"
#include <cstdio>
int main() {
    const int n = 64, k = 64;
    printf("start\n"); fflush(stdout);
    uint8_t *da=nullptr, *dw=nullptr, *dwsc=nullptr; float *dout=nullptr, *dasc=nullptr;
    printf("malloc %d\n", (int)cudaMalloc(&da, k)); fflush(stdout);
    printf("malloc %d\n", (int)cudaMalloc(&dw, (size_t)n*k)); fflush(stdout);
    printf("malloc %d\n", (int)cudaMalloc(&dwsc, (size_t)(n/32)*(k/32))); fflush(stdout);
    printf("malloc %d\n", (int)cudaMalloc(&dasc, (size_t)(k/32)*sizeof(float))); fflush(stdout);
    printf("malloc %d\n", (int)cudaMalloc(&dout, (size_t)n*sizeof(float))); fflush(stdout);
    cudaMemset(da, 0x38, k); cudaMemset(dw, 0x38, (size_t)n*k);
    cudaMemset(dwsc, 127, (size_t)(n/32)*(k/32)); cudaMemset(dasc, 0, (size_t)(k/32)*sizeof(float));
    printf("before launch\n"); fflush(stdout);
    int rc = dsv41_gemm_fp8_swapab(da, dasc, dw, dwsc, nullptr, dout, n, k, 0);
    printf("rc=%d\n", rc); fflush(stdout);
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync=%s\n", cudaGetErrorString(e)); fflush(stdout);
    float o[64]; cudaMemcpy(o, dout, sizeof(o), cudaMemcpyDeviceToHost);
    printf("out[0]=%f out[63]=%f\n", o[0], o[63]); fflush(stdout);
    return 0;
}
