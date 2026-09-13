// bench_head_old.cu — standalone micro-bench for the verify head's bf16 GEMV
// (`dsv41_head_gemv_bf16_mrows`, kernels/cuda/dsv41_glue.cu) at the shapes the
// TileLang head prototype (`kernels/tilelang/head_bf16_tilelang.py`) is compared
// against.  Same TU style as bench_proj_old.cu: #include the production TU and
// call the launcher instead of re-deriving the kernel.
//
// Shapes (code facts, not the task's): the head is `head.weight` =
// [vocab=129280, dim=5120] bf16, Shard::Replicated (weights.rs:90).  The verify
// head is NOT vocab-sliced (dspark-verify-perf-plan.md §P1 note) but the
// production *draft* path slices it to 129280/world = 16160 rows.  This bench
// runs both, plus the literal [16160, 256] shape the task named, so the
// TileLang number can be read against whichever the plan meant.
//
// Build (NO GPU needed):
//   nvcc -O3 --use_fast_math -std=c++17 -gencode arch=compute_103a,code=sm_103a \
//        bench_head_old.cu -o bench_head_old
// Run (ONE free GPU):
//   CUDA_VISIBLE_DEVICES=<free> ./bench_head_old [iters]
#include "dsv41_glue.cu"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static float* dev_alloc_f32(size_t n) {
    float* p = nullptr;
    if (cudaMalloc((void**)&p, n * sizeof(float)) != cudaSuccess) {
        printf("cudaMalloc f32 %zu failed\n", n);
        std::exit(1);
    }
    return p;
}
static __nv_bfloat16* dev_alloc_bf16(size_t n) {
    __nv_bfloat16* p = nullptr;
    if (cudaMalloc((void**)&p, n * sizeof(__nv_bfloat16)) != cudaSuccess) {
        printf("cudaMalloc bf16 %zu failed\n", n);
        std::exit(1);
    }
    return p;
}

// One (name, n, k) shape: report M1 and M6 µs/call and their ratio.
static void run_shape(const char* name, int n, int k, int iters) {
    const size_t nw = (size_t)n * k;
    const size_t nx = (size_t)8 * k;   // room for m up to 8
    const size_t no = (size_t)8 * n;

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t need = nw * 2 + nx * 4 + no * 4 + (64u << 20);
    if (free_b < need) {
        printf("  SKIP [%s] needs %.0f MB free, %.0f MB available\n", name,
               (double)need / 1048576.0, (double)free_b / 1048576.0);
        return;
    }

    __nv_bfloat16* w = dev_alloc_bf16(nw);
    float* x = dev_alloc_f32(nx);
    float* o = dev_alloc_f32(no);

    std::vector<__nv_bfloat16> hw(nw);
    for (size_t i = 0; i < nw; ++i) {
        uint16_t b = (uint16_t)(0x3C00u | (uint16_t)(i * 2654435761u & 0x3FFu));
        hw[i] = *reinterpret_cast<__nv_bfloat16*>(&b);
    }
    std::vector<float> hx(nx);
    for (size_t i = 0; i < nx; ++i) hx[i] = (float)((int)(i % 1999) - 999) * 1e-3f;
    cudaMemcpy(w, hw.data(), nw * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(x, hx.data(), nx * 4, cudaMemcpyHostToDevice);

    printf("  [%s] n=%d k=%d  weight=%.1f MB\n", name, n, k, (double)(nw * 2) / 1048576.0);

    double us[2] = {0, 0};
    const int ms[2] = {1, 6};
    for (int a = 0; a < 2; ++a) {
        const int m = ms[a];
        // warm-up
        for (int i = 0; i < 5; ++i)
            dsv41_head_gemv_bf16_mrows(w, x, o, m, n, k, nullptr);
        cudaDeviceSynchronize();
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, nullptr);
        for (int i = 0; i < iters; ++i)
            dsv41_head_gemv_bf16_mrows(w, x, o, m, n, k, nullptr);
        cudaEventRecord(e1, nullptr);
        cudaEventSynchronize(e1);
        float msf = 0.f;
        cudaEventElapsedTime(&msf, e0, e1);
        us[a] = (double)msf * 1000.0 / iters;
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        printf("      m=%d  %8.3f us/call   (achieved %.2f TB/s)\n", m, us[a],
               (double)(nw * 2) / (us[a] * 1e-6) / 1e12);
    }
    printf("      M6/M1 = %.3f\n", us[1] / us[0]);

    cudaFree(w); cudaFree(x); cudaFree(o);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 200;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("== head bf16 GEMV (dsv41_head_gemv_bf16_mrows) bench ==  device=%s  iters=%d\n",
           prop.name, iters);
    printf("   launch: ceil(n/8) blocks capped 4096, 256 threads (8 warps)\n");

    // The production draft/verify head slice (129280/8), full vocabulary, and
    // the literal task shape.
    run_shape("head_slice_16160x5120", 16160, 5120, iters);
    run_shape("head_full_129280x5120", 129280, 5120, iters);
    run_shape("csv_literal_16160x256", 16160, 256, iters);
    return 0;
}
