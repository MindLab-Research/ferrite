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
    const int B = 16, H = 8, D = 256, DV = 1024, TOPK = 2051, T = 8192;
    int iters = argc > 1 ? atoi(argv[1]) : 20;
    std::vector<float*> h_k(B), h_v(B), h_ks(B), h_vs(B);
    std::vector<unsigned char*> d_kn(B), d_v(B);
    std::vector<float*> d_ks(B), d_vs(B);
    std::vector<const int*> h_tot(B);
    std::vector<int*> d_tot(B);
    std::vector<float*> h_idx(B), h_out(B);
    float* q; CK(cudaMalloc(&q, (size_t)B * H * D * 4));
    {   // realistic non-degenerate inputs (all-zero q/ksc makes the softmax
        // degenerate and is NOT representative of production)
        std::vector<float> qv((size_t)B * H * D);
        for (size_t i = 0; i < qv.size(); i++) qv[i] = 0.02f * (float)((int)(i % 37) - 18);
        CK(cudaMemcpy(q, qv.data(), qv.size() * 4, cudaMemcpyHostToDevice));
    }
    for (int b = 0; b < B; b++) {
        CK(cudaMalloc(&d_kn[b], (size_t)T * H * D));        // fp8
        CK(cudaMalloc(&d_v[b], (size_t)T * H * DV));
        {
            std::vector<unsigned char> kv((size_t)T * H * D);
            for (size_t i = 0; i < kv.size(); i++) kv[i] = (unsigned char)(0x20 + (i % 60));
            CK(cudaMemcpy(d_kn[b], kv.data(), kv.size(), cudaMemcpyHostToDevice));
            std::vector<unsigned char> vv((size_t)T * H * DV);
            for (size_t i = 0; i < vv.size(); i++) vv[i] = (unsigned char)(0x20 + ((i * 7) % 60));
            CK(cudaMemcpy(d_v[b], vv.data(), vv.size(), cudaMemcpyHostToDevice));
        }
        CK(cudaMalloc(&d_ks[b], (size_t)T * H * 4));
        CK(cudaMalloc(&d_vs[b], (size_t)T * H * 4));
        {
            std::vector<float> s1((size_t)T * H, 0.013f), s2((size_t)T * H, 0.017f);
            CK(cudaMemcpy(d_ks[b], s1.data(), s1.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_vs[b], s2.data(), s2.size() * 4, cudaMemcpyHostToDevice));
        }
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
    {   // sentinel: any float the kernel does NOT write keeps this value, so a
        // BLK=256 vs BLK=512 diff can distinguish "unwritten" from "differs".
        std::vector<float> sv((size_t)B * H * DV, 12345.0f);
        CK(cudaMemcpy(d_out, sv.data(), sv.size() * 4, cudaMemcpyHostToDevice));
    }
    {
        std::vector<float> iv((size_t)B * TOPK);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < TOPK; i++) iv[(size_t)b * TOPK + i] = (i < 2048) ? (float)((i * 2 + b) % 2048) : -1.0f;
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
    // optional: dump the output for a bit-exact BLK=256 vs BLK=512 comparison
    if (argc > 2) {
        std::vector<float> ov((size_t)B * H * DV);
        CK(cudaMemcpy(ov.data(), d_out, ov.size() * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(argv[2], "wb");
        fwrite(ov.data(), 4, ov.size(), f);
        fclose(f);
        printf("dumped %s (%zu floats)\n", argv[2], ov.size());
    }
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
