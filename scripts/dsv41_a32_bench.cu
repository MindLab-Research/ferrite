// Isolated GEMV timing for the a32 / occupancy experiment.
//
// Question: the M=1 fp8 GEMV carries a block-wide pre-decoded activation table
// (`s_af`, k f32 = 20 KB at the model's k=5120 - the "a32" optimisation). At
// k=5120 it is what pushes the dynamic shared memory past the point where the
// driver gives the block 8 resident blocks per SM; a32's measured win
// (-6/-8/-13%) came from the small-n probes (n=256/1024/1664), never from the
// production k. If dropping the table doubles the residency, the lost LDS
// latency may be paid back with interest.
//
// The table is NOT toggled by DSV41_GEMV_FP8_MODE (both staged modes build it -
// mode 3 merely reads the fp8 activation from global instead of the staged s_a
// copy). It has its own gate: DSV41_GEMV_A32 (1 = keep, 0 = drop). Both gates
// are process-static, so each ARM IS ONE PROCESS with the env set.
// scripts/dsv41_a32_bench.sh drives the arms and diffs the output.
//
// Linked against the PRODUCTION .so (LD_LIBRARY_PATH), one shape per line in a
// machine-readable key=value form. The last field of every SHAPE line is the
// numerical fingerprint (out[0..3]); it MUST be identical across arms - the a32
// on/off forms compute the same product, so a difference means the "free"
// occupancy win moved the sums and the arm is invalid.
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>

extern "C" int dsv41_gemm_fp8_mx(const uint8_t* a, const float* a_scale, const uint8_t* w,
                                 const uint8_t* w_scale, const float* bias, float* out, int m,
                                 int n, int k, cudaStream_t s);
extern "C" size_t dsv41_gemv_gsmem(int mode, int warps, int k);
extern "C" int dsv41_gemv_occupancy(int warps, size_t gsmem);

static int env_int(const char* name, int dflt) {
    const char* e = getenv(name);
    return (e == nullptr || e[0] == '\0') ? dflt : atoi(e);
}

// One timing run at (n, k). Mirrors scripts/dsv41_gemv_bench.cu's data setup: the
// weight bytes are legal e4m3 codes, the activation scales are pinned to 1.0 and
// the ue8m0 weight-scale bytes to 0x7F (=1.0) so the dot is neither NaN nor zero
// and the fingerprint can actually move if the arithmetic does.
static int bench_one(int n, int k, int mode, int warps, int reps, int iters_unused) {
    (void)iters_unused;
    const size_t wq_bytes = (size_t)n * (size_t)k;
    const size_t wbytes = wq_bytes + (size_t)(n / 32) * (size_t)(k / 32);
    void* w = nullptr;
    if (cudaMalloc(&w, wbytes) != cudaSuccess) {
        printf("FAIL n=%d k=%d malloc %zu\n", n, k, wbytes);
        return 1;
    }
    {
        std::vector<uint8_t> hw(wbytes);
        for (size_t i = 0; i < wq_bytes; ++i) hw[i] = (uint8_t)((i * 37 + 11) & 0x7Eu);
        for (size_t i = wq_bytes; i < wbytes; ++i) hw[i] = 0x7F;   // ue8m0 = 1.0
        cudaMemcpy(w, hw.data(), wbytes, cudaMemcpyHostToDevice);
    }
    void* x = nullptr;
    cudaMalloc(&x, (size_t)k * 4 + 256);
    {
        std::vector<uint8_t> hx((size_t)k * 4 + 256);
        for (size_t i = 0; i < (size_t)k; ++i) hx[i] = (uint8_t)((i * 53 + 7) & 0x7Eu);
        const float one = 1.0f;
        for (size_t i = 0; i < (size_t)(k / 32); ++i) std::memcpy(&hx[k + i * 4], &one, 4);
        cudaMemcpy(x, hx.data(), hx.size(), cudaMemcpyHostToDevice);
    }
    void* out = nullptr;
    cudaMalloc(&out, (size_t)n * 4);

    const uint8_t* a = (const uint8_t*)x;
    const float* asc = (const float*)(a + k);
    const uint8_t* wq = (const uint8_t*)w;
    const uint8_t* wsc = wq + wq_bytes;

    for (int i = 0; i < 20; ++i) dsv41_gemm_fp8_mx(a, asc, wq, wsc, nullptr, (float*)out, 1, n, k, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL n=%d k=%d launch err=%s\n", n, k, cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::vector<float> ts;
    ts.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, 0);
        dsv41_gemm_fp8_mx(a, asc, wq, wsc, nullptr, (float*)out, 1, n, k, 0);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        ts.push_back(ms * 1000.0f);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    std::sort(ts.begin(), ts.end());

    float fp[4] = {0, 0, 0, 0};
    cudaMemcpy(fp, out, sizeof(fp), cudaMemcpyDeviceToHost);
    const size_t gsmem = dsv41_gemv_gsmem(mode, warps, k);
    const int occ = dsv41_gemv_occupancy(warps, gsmem);
    printf("SHAPE n=%-6d k=%-6d mode=%d warps=%d smem=%zu occ=%d median=%9.3f us min=%9.3f us "
           "fp=%.9g,%.9g,%.9g,%.9g\n",
           n, k, mode, warps, gsmem, occ, ts[ts.size() / 2], ts.front(), fp[0], fp[1], fp[2], fp[3]);

    cudaFree(w);
    cudaFree(x);
    cudaFree(out);
    return 0;
}

int main(int argc, char** argv) {
    cudaFree(0);
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    const int mode = env_int("DSV41_GEMV_FP8_MODE", 4);
    const int a32 = env_int("DSV41_GEMV_A32", 1);
    const int warps = env_int("DSV41_GEMV_FP8_WARPS", 4);
    const int reps = env_int("DSV41_A32_REPS", 400);
    const int k = env_int("DSV41_A32_K", 5120);

    // Default shape list: the production k with every projection n in the decode
    // chain (sharedexp 256, q_b 1024, q_lora/idx 1280, wq_a+wkv 1664, wo_b /
    // idx_wq 2048, wq_b 4096). argv[1..] overrides with an n list.
    std::vector<int> ns;
    for (int i = 1; i < argc; ++i) ns.push_back(atoi(argv[i]));
    if (ns.empty()) ns = {256, 512, 1024, 1280, 1664, 2048, 4096};

    printf("# ARG device=%s mode=%d a32=%d warps=%d k=%d reps=%d\n", prop.name, mode, a32, warps, k,
           reps);
    int rc = 0;
    for (int n : ns) rc |= bench_one(n, k, mode, warps, reps, 0);
    return rc;
}
