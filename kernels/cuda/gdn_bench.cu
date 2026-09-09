// Isolated repro of gdn_chunk_batched_kernel for ncu. Real decode shape:
// B=16, h=64, dk=dv=128 (state 128x128 per (seq,head)).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
extern "C" cudaError_t ferrite_gdn_chunk_batched(
    const float* q, const float* k, const float* v, const float* beta,
    const float* gate, const float* a_log, float* const* state_ptrs,
    float* out, int B, int h, int dk, int dv, cudaStream_t s);
#define CK(x) do { cudaError_t e_=(x); if(e_){printf("ERR %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);} } while(0)
int main(int argc, char** argv) {
    const int B = 16, H = 64, DK = 128, DV = 128;
    int iters = argc > 1 ? atoi(argv[1]) : 50;
    float *q,*k,*v,*beta,*gate,*alog,*out;
    CK(cudaMalloc(&q, (size_t)B*H*DK*4)); CK(cudaMemset(q,0,(size_t)B*H*DK*4));
    CK(cudaMalloc(&k, (size_t)B*H*DK*4)); CK(cudaMemset(k,0,(size_t)B*H*DK*4));
    CK(cudaMalloc(&v, (size_t)B*H*DV*4)); CK(cudaMemset(v,0,(size_t)B*H*DV*4));
    CK(cudaMalloc(&beta, (size_t)B*H*4)); CK(cudaMemset(beta,0,(size_t)B*H*4));
    CK(cudaMalloc(&gate, (size_t)B*H*DK*4)); CK(cudaMemset(gate,0,(size_t)B*H*DK*4));
    CK(cudaMalloc(&alog, (size_t)H*4)); CK(cudaMemset(alog,0,(size_t)H*4));
    CK(cudaMalloc(&out, (size_t)B*H*DV*4));
    std::vector<float*> sp(B), hp(B);
    for (int b = 0; b < B; b++) {
        CK(cudaMalloc(&sp[b], (size_t)H*DK*DV*4));
        CK(cudaMemset(sp[b], 0, (size_t)H*DK*DV*4));
        hp[b] = sp[b];
    }
    float** dsp; CK(cudaMalloc(&dsp, B*sizeof(void*)));
    CK(cudaMemcpy(dsp, hp.data(), B*sizeof(void*), cudaMemcpyHostToDevice));
    cudaEvent_t a,b2; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b2));
    for (int i=0;i<5;i++) CK(ferrite_gdn_chunk_batched(q,k,v,beta,gate,alog,(float* const*)dsp,out,B,H,DK,DV,0));
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i=0;i<iters;i++) CK(ferrite_gdn_chunk_batched(q,k,v,beta,gate,alog,(float* const*)dsp,out,B,H,DK,DV,0));
    CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b2));
    printf("gdn_chunk (B=%d h=%d dk=%d dv=%d): %.3f ms/call\n", B,H,DK,DV, ms/iters);
    return 0;
}
