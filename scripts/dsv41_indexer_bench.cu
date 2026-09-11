// Isolated timing for dsv41_indexer_topk at the shapes the decode step uses
#include <cstdint>
// (index_n_heads=32, index_head_dim=128, index_topk=512) across the plausible
// live-latent counts. The profiler's 160 us per call fits none of these on
// paper, so this measures instead of guessing.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

extern "C" int dsv41_indexer_topk(const float* q, const float* index_k, const float* weights,
                                  const uint8_t* candidates, const int32_t* compress_lens,
                                  int32_t* out, int b, int m, int nh, int hd, int n_pos, int topk,
                                  int offset, float softmax_scale, float head_scale,
                                  int uses_candidates, cudaStream_t s);

static void bench_idx(int n_pos, int nh, int hd, int topk, int reps) {
    float* q = nullptr;
    float* ik = nullptr;
    float* w = nullptr;
    int32_t* lens = nullptr;
    int32_t* out = nullptr;
    cudaMalloc(&q, (size_t)nh * hd * 4);
    cudaMalloc(&ik, (size_t)n_pos * hd * 4);
    cudaMalloc(&w, (size_t)nh * 4);
    cudaMalloc(&lens, 4);
    cudaMalloc(&out, (size_t)topk * 4);
    cudaMemset(q, 0x3c, (size_t)nh * hd * 4);
    cudaMemset(ik, 0x3d, (size_t)n_pos * hd * 4);
    cudaMemset(w, 0x3c, (size_t)nh * 4);
    cudaMemcpy(lens, &n_pos, 4, cudaMemcpyHostToDevice);
    float scale = 1.0f / sqrtf((float)hd) / sqrtf((float)nh);
    for (int i = 0; i < 5; ++i)
        dsv41_indexer_topk(q, ik, w, nullptr, lens, out, 1, 1, nh, hd, n_pos, topk, 0, scale, 1.0f,
                           0, 0);
    cudaDeviceSynchronize();
    std::vector<float> ts;
    for (int i = 0; i < reps; ++i) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, 0);
        dsv41_indexer_topk(q, ik, w, nullptr, lens, out, 1, 1, nh, hd, n_pos, topk, 0, scale, 1.0f,
                           0, 0);
        cudaEventRecord(b, 0);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        ts.push_back(ms);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    std::sort(ts.begin(), ts.end());
    printf("  indexer n_pos=%-6d nh=%d hd=%d topk=%d  median %8.2f us  (min %8.2f)\n", n_pos, nh, hd,
           topk, ts[ts.size() / 2] * 1000.0, ts.front() * 1000.0);
    cudaFree(q); cudaFree(ik); cudaFree(w); cudaFree(lens); cudaFree(out);
}

int main() {
    cudaFree(0);
    int dev = 0;
    cudaGetDevice(&dev);
    printf("indexer_topk isolation (device %d)\n", dev);
    bench_idx(32, 32, 128, 512, 200);
    bench_idx(128, 32, 128, 512, 200);
    bench_idx(512, 32, 128, 512, 200);
    bench_idx(2048, 32, 128, 512, 200);
    bench_idx(4096, 32, 128, 512, 100);
    return 0;
}
