// Isolated repro of sparse_attn_v2_batched_kernel for ncu deep-dive.
// Real B=16 decode shape: 64 heads, d=dv=256, live_k = 2048 slots.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

extern "C" cudaError_t ferrite_sparse_attn_v2_batched(
    const float* q, float* const* k_tbl, float* const* v_tbl,
    float* const* ksc_tbl, float* const* vsc_tbl,
    const float* idx, float* out, int B, const int* const* total_tbl,
    int h, int d, int dv, int topk, cudaStream_t s);

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("ERR %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

int main(int argc, char** argv) {
    const int B = 16, H = 64, D = 256, DV = 256, TOPK = 2051, T = 4096;
    int iters = argc > 1 ? atoi(argv[1]) : 20;
    std::vector<float*> h_k(B), h_v(B), h_ks(B), h_vs(B);
    std::vector<unsigned char*> d_kn(B), d_v(B);
    std::vector<float*> d_ks(B), d_vs(B);
    std::vector<const int*> h_tot(B);
    std::vector<int*> d_tot(B);
    std::vector<float*> h_idx(B), h_out(B);
    float* q; CK(cudaMalloc(&q, (size_t)B * H * D * 4));
    CK(cudaMemset(q, 0, (size_t)B * H * D * 4));
    for (int b = 0; b < B; b++) {
        CK(cudaMalloc(&d_kn[b], (size_t)T * H * D));        // fp8
        CK(cudaMalloc(&d_v[b], (size_t)T * H * DV));
        CK(cudaMemset(d_kn[b], 1, (size_t)T * H * D));
        CK(cudaMemset(d_v[b], 1, (size_t)T * H * DV));
        CK(cudaMalloc(&d_ks[b], (size_t)T * H * 4));
        CK(cudaMalloc(&d_vs[b], (size_t)T * H * 4));
        CK(cudaMemset(d_ks[b], 0, (size_t)T * H * 4));
        CK(cudaMemset(d_vs[b], 0, (size_t)T * H * 4));
        int* p; CK(cudaMallocHost(&p, 4)); *p = 2048;
        h_tot[b] = p;
        CK(cudaMalloc(&d_tot[b], 4)); CK(cudaMemcpy(d_tot[b], p, 4, cudaMemcpyHostToDevice));
        (void)h_idx; (void)h_out;
    }
    std::vector<const int*> totp(B);
    for (int b = 0; b < B; b++) totp[b] = h_tot[b];
    int** d_tot_tbl; CK(cudaMalloc(&d_tot_tbl, B * sizeof(void*)));
    CK(cudaMemcpy(d_tot_tbl, totp.data(), B * sizeof(void*), cudaMemcpyHostToDevice));
    std::vector<unsigned char*> kn_p(B), v_p(B);
    for (int b = 0; b < B; b++) { kn_p[b] = d_kn[b]; v_p[b] = d_v[b]; }
    unsigned char** d_kn_tbl; CK(cudaMalloc(&d_kn_tbl, B * sizeof(void*)));
    CK(cudaMemcpy(d_kn_tbl, kn_p.data(), B * sizeof(void*), cudaMemcpyHostToDevice));
    unsigned char** d_v_tbl; CK(cudaMalloc(&d_v_tbl, B * sizeof(void*)));
    CK(cudaMemcpy(d_v_tbl, v_p.data(), B * sizeof(void*), cudaMemcpyHostToDevice));
    std::vector<float*> ks_p(B), vs_p(B);
    for (int b = 0; b < B; b++) { ks_p[b] = d_ks[b]; vs_p[b] = d_vs[b]; }
    // idx/out are per-seq CONTIGUOUS buffers indexed as idx + seq*topk
    float* d_idx; CK(cudaMalloc(&d_idx, (size_t)B * TOPK * 4));
    float* d_out; CK(cudaMalloc(&d_out, (size_t)B * H * DV * 4));
    {
        std::vector<float> iv((size_t)B * TOPK);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < TOPK; i++) iv[(size_t)b * TOPK + i] = (i < 2048) ? (float)((i * 2 + b) % T) : -1.0f;
        CK(cudaMemcpy(d_idx, iv.data(), (size_t)B * TOPK * 4, cudaMemcpyHostToDevice));
    }
    float** d_ks_tbl; CK(cudaMalloc(&d_ks_tbl, B * sizeof(void*)));
    CK(cudaMemcpy(d_ks_tbl, ks_p.data(), B * sizeof(void*), cudaMemcpyHostToDevice));
    float** d_vs_tbl; CK(cudaMalloc(&d_vs_tbl, B * sizeof(void*)));
    CK(cudaMemcpy(d_vs_tbl, vs_p.data(), B * sizeof(void*), cudaMemcpyHostToDevice));


    cudaEvent_t a, b2; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b2));
    for (int i = 0; i < 5; i++)
        CK(ferrite_sparse_attn_v2_batched(q, (float* const*)d_kn_tbl, (float* const*)d_v_tbl,
            d_ks_tbl, d_vs_tbl, d_idx, d_out, B,
            (const int* const*)d_tot_tbl, H, D, DV, TOPK, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++)
        CK(ferrite_sparse_attn_v2_batched(q, (float* const*)d_kn_tbl, (float* const*)d_v_tbl,
            d_ks_tbl, d_vs_tbl, d_idx, d_out, B,
            (const int* const*)d_tot_tbl, H, D, DV, TOPK, 0));
    CK(cudaEventRecord(b2)); CK(cudaEventSynchronize(b2));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b2));
    printf("sparse_attn (B=%d h=%d d=%d live_k=2048): %.3f ms/call\n", B, H, D, ms / iters);
    return 0;
}
