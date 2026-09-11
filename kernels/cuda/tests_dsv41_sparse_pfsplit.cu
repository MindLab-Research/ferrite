// Validation harness for the prefetch split sparse attention
// (DSV41_ATTN_PF_SPLIT), the kernel that moved the pf kernel's three-deep
// pipeline into the key-split shape.
//
// Two questions, one cheap non-model run:
//   1. CORRECTNESS. C=1 must be BIT-IDENTICAL to the single-block pf kernel
//      (one chunk, and the merge of a single partial reproduces its epilogue
//      exactly). C>1 differs only in the final summation grouping, so it is
//      judged by tolerance.
//   2. LATENCY. The split was rejected once already because losing the
//      pipeline cost 0.37 ms/step; this prints the per-call time of every
//      configuration so the split can be compared against pf on the same box
//      and the same data.
//
// Build (a GPU node; nvcc only, no model weights):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
//        -c dsv41_kernels.cu -o dsv41_kernels.o
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//        -o t_sparse_pfsplit tests_dsv41_sparse_pfsplit.cu dsv41_kernels.o
//
// Run each configuration in its OWN process: the launcher caches its env gates
// in function-local statics (deliberately - a per-call getenv is a hot-path
// slip), so one process can only ever exercise one arm.
//   DSV41_ATTN_PF_SPLIT=0 ./t_sparse_pfsplit pf   50   # single-block pf + write out_pf.bin
//   DSV41_ATTN_PF_SPLIT=1 ./t_sparse_pfsplit c1   50   # must be bit-identical to pf
//   DSV41_ATTN_PF_SPLIT=8 ./t_sparse_pfsplit c8   50   # tolerance vs pf
//
// Shape is the production one, scaled to TP8: window 128, 512 live compressed
// rows (topk = window + clen = 640), head_dim 512, 8 local heads - i.e. the
// 1.31 MB of kv rows the deep dive measured.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int dsv41_sparse_attn(const float* q, const float* kv, const float* sink,
                                 const int32_t* idxs, float* out, int b, int m, int h, int d,
                                 const int* clen, int window, int index_topk, float scale,
                                 cudaStream_t s);

// The dsv41 ABI returns a cudaError_t as int (not the enum), so it gets its own
// checker rather than a cast to cudaError_t.
#define SPA(...)                                                                       \
    do {                                                                               \
        int rc_ = dsv41_sparse_attn(__VA_ARGS__);                                      \
        if (rc_ != 0) {                                                                \
            printf("ERR sparse_attn %s @%d\n", cudaGetErrorString((cudaError_t)rc_),   \
                   __LINE__);                                                          \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            printf("ERR %s @%d\n", cudaGetErrorString(e_), __LINE__);                   \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static unsigned rng_state = 12345u;
static float rnd() {  // deterministic, non-degenerate, |x| < 0.05
    rng_state = rng_state * 1664525u + 1013904223u;
    return (((rng_state >> 8) & 0xFFFFu) / 65535.0f - 0.5f) * 0.1f;
}

int main(int argc, char** argv) {
    const char* label = argc > 1 ? argv[1] : "run";
    const int iters = argc > 2 ? atoi(argv[2]) : 50;
    // Clen is the LIVE compressed-row count in production (it grows with the
    // position), so the slot count is a real variable of the shape: topk =
    // window + min(clen, index_topk). argv[3] lets one binary sweep it.
    const int clen_arg = argc > 3 ? atoi(argv[3]) : 512;

    const int b = 1, m = 1, h = 8, d = 512;
    const int window = 128, index_topk = 2048;
    const int clen = clen_arg;
    const int n = window + clen;                                   // device-side in production
    const int topk = window + (clen < index_topk ? clen : index_topk);
    const float scale = 1.0f / sqrtf((float)d);

    // ---- host inputs -------------------------------------------------------
    std::vector<float> hq((size_t)h * d), hkv((size_t)n * d), hsink(h), href;
    std::vector<int32_t> hidx((size_t)topk);
    for (auto& x : hq) x = rnd();
    for (auto& x : hkv) x = rnd();
    for (auto& x : hsink) x = -0.3f + 0.1f * rnd();
    for (int t = 0; t < topk; ++t) {
        // The window block is the ring in age order; the compressed block sits
        // at [window, window + clen). -1 is the "skip this slot" marker and is
        // deliberately sprinkled INSIDE chunks too - the pipeline must keep the
        // pf kernel's semantics (no load, no state update, chain still advances).
        if (t < window)
            hidx[t] = (t + 5) % window;
        else
            hidx[t] = window + (int)(((long long)t * 7919) % clen);
        if (t > 200 && (t % 13) == 7) hidx[t] = -1;
    }

    float* dq = nullptr;
    float* dkv = nullptr;
    float* dsink = nullptr;
    float* dout = nullptr;
    int32_t* didx = nullptr;
    int* dclen = nullptr;
    CK(cudaMalloc(&dq, hq.size() * sizeof(float)));
    CK(cudaMalloc(&dkv, hkv.size() * sizeof(float)));
    CK(cudaMalloc(&dsink, hsink.size() * sizeof(float)));
    CK(cudaMalloc(&dout, (size_t)h * d * sizeof(float)));
    CK(cudaMalloc(&didx, hidx.size() * sizeof(int32_t)));
    CK(cudaMalloc(&dclen, sizeof(int)));
    CK(cudaMemcpy(dq, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkv, hkv.data(), hkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsink, hsink.data(), hsink.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(didx, hidx.data(), hidx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dclen, &clen, sizeof(int), cudaMemcpyHostToDevice));

    // ---- run + time --------------------------------------------------------
    for (int i = 0; i < 5; ++i)  // warmup (also faults the scratch in)
        SPA(dq, dkv, dsink, didx, dout, b, m, h, d, dclen, window, index_topk, scale,
            (cudaStream_t)0);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, (cudaStream_t)0));
    for (int i = 0; i < iters; ++i)
        SPA(dq, dkv, dsink, didx, dout, b, m, h, d, dclen, window, index_topk, scale,
            (cudaStream_t)0);
    CK(cudaEventRecord(e1, (cudaStream_t)0));
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));

    std::vector<float> out((size_t)h * d);
    CK(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double sum = 0, sumsq = 0, amax = 0;
    unsigned long long fnv = 1469598103934665603ull;  // bitwise digest (catches 1-ULP drift)
    for (float x : out) {
        sum += x;
        sumsq += (double)x * (double)x;
        amax = fmax(amax, fabs((double)x));
        unsigned int bits;
        memcpy(&bits, &x, 4);
        for (int k = 0; k < 4; ++k) {
            fnv ^= (bits >> (8 * k)) & 0xFFu;
            fnv *= 1099511628211ull;
        }
    }
    printf("%-4s  %8.3f us/call   sum %+.6e  max|x| %.6e  fnv %016llx\n", label,
           (double)ms * 1000.0 / iters, sum, amax, fnv);

    // ---- cross-run comparison ---------------------------------------------
    char path[64];
    snprintf(path, sizeof(path), "out_%s.bin", label);
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(out.data(), sizeof(float), out.size(), f);
        fclose(f);
    }
    if (strcmp(label, "pf") != 0) {
        FILE* r = fopen("out_pf.bin", "rb");
        if (r) {
            std::vector<float> ref(out.size());
            size_t got = fread(ref.data(), sizeof(float), ref.size(), r);
            fclose(r);
            if (got == ref.size()) {
                double md = 0, mr = 0;
                int nbits = 0;
                for (size_t i = 0; i < out.size(); ++i) {
                    const double dd = fabs((double)out[i] - (double)ref[i]);
                    const double den = fabs((double)ref[i]) > 1e-12 ? fabs((double)ref[i]) : 1e-12;
                    md = fmax(md, dd);
                    mr = fmax(mr, dd / den);
                    unsigned int a, c;
                    memcpy(&a, &out[i], 4);
                    memcpy(&c, &ref[i], 4);
                    if (a != c) ++nbits;
                }
                printf("      vs pf: max|d|=%.3e  max rel=%.3e  differing lanes=%d/%zu  %s\n", md,
                       mr, nbits, out.size(), nbits == 0 ? "BIT-IDENTICAL" : "tolerance");
            }
        } else {
            printf("      (out_pf.bin missing - run the pf arm first)\n");
        }
    }
    return 0;
}
