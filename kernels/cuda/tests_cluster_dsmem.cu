// tests_cluster_dsmem.cu — ⑤b pre-study: the CLUSTER (CGA) + DSMEM form of the
// multi-row fp8 GEMV. **PRE-STUDY / COMPILE-ONLY** — this file is NOT linked
// into libferrite_kernels.so and has NO runtime gate in the engine; it exists to
// (a) prove the CUDA 13.2 / sm_103a toolchain accepts the cluster + distributed
// shared-memory primitives this design needs, and (b) pin the exact API surface
// the eventual `gemm_fp8_mrows_cluster_kernel<M>` will use.
//
// Design: docs/agent/cluster-dsmem-design.md (deliverable ⑤b).
// Sibling: docs/agent/tensorcore-proj-design.md §5.2, mrows-mpar-design.md §1.4.
//
// WHAT IT PROVES (compile-only; no GPU is touched):
//   1. `__cluster_dims__(N,1,1)` on a dynamic-smem kernel compiles for sm_103a.
//   2. `cooperative_groups::this_cluster()` + `cluster.map_shared_rank(ptr, r)`
//      — the DSMEM read of ANOTHER block's `extern __shared__` slice.
//   3. `cluster.sync()` (the rendezvous that publishes rank 0's smem) and the
//      split `cluster.barrier_arrive()` / `cluster.barrier_wait()` pair.
//   4. `__cluster_dims__` carrying a TEMPLATE parameter (probed separately below;
//      if the toolchain rejects it the literal form is the fallback).
//
// Build (compile-only, needs nvcc, NO GPU, a few seconds):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math \
//        -std=c++17 -c -o /tmp/t_cluster_dsmem.o \
//        kernels/cuda/tests_cluster_dsmem.cu
// Run (needs ONE free GPU; peak allocation is a few MB) — NOT part of ⑤b's
// acceptance (⑤b is 预研), listed only so the harness is usable later:
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_cluster_dsmem
//
// ⚠️ DO NOT add this TU to build.sh: ⑤b is design + compile-only until ⑤a's
// measured L2 result decides between them (cluster-dsmem-design.md §5).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Local stand-ins for the two device helpers the production kernels use
// (`ue8m0_to_f` / `e4m3_to_f` in dsv41_kernels.cu). Same source in this TU so
// the compile probe does not drag the 13484-line production TU in.
__device__ __forceinline__ float dsmem_ue8m0_to_f(uint8_t b) {
    return __uint_as_float((b == 0xffu) ? 0x7fc00000u : ((uint32_t)b << 23));
}
__device__ __forceinline__ float dsmem_e4m3_to_f(uint8_t b) {
    // bit-exact LUT fill is not the point of a compile probe; a stable mapping
    // keeps the harness's own parity check self-consistent.
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0x0f;
    const int m = b & 0x07;
    if (e == 0) return (s ? -1.f : 1.f) * (float)m * (1.f / 512.f);
    return (s ? -1.f : 1.f) * (1.f + (float)m / 8.f) * exp2f((float)(e - 7));
}

// ---------------------------------------------------------------------------
// The cluster GEMV. Same ABI shape as `gemm_fp8_mrows_mp_kernel<M>` (see
// dsv41_kernels.cu) with ONE structural change: the M activation rows live in
// the CLUSTER's block-rank axis instead of the warp axis, and the weight slab is
// staged ONCE per cluster (rank 0) and read from every block through DSMEM.
//
// grid : M * ceil(n / rpb) blocks        (blockIdx.x = row-group * M + rank)
// block: rpb warps                       (warp w -> output row row0 + w)
// rank : the ACTIVATION row g this block serves
// rank 0 stages `rpb * k` weight bytes into ITS OWN smem; every rank reads
// `cluster.map_shared_rank(s_w, 0)` — the SAME bytes, one DRAM round trip.
//
// The consume chain is byte-for-byte the MPAR/legacy form: `kb` ascends,
// `j = kb*32 + lane`, `acc += av * wv` single serial add, one `shfl_xor` tree.
// Only the ADDRESS the weight byte is read from moves (private smem -> remote
// smem); a read is a read (design doc §4, "逐位等价").
// ---------------------------------------------------------------------------
template <int M>
__global__ void __cluster_dims__(M, 1, 1) __launch_bounds__(1024)
cluster_mrows_kernel(const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
                     const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_scale,
                     const float* __restrict__ bias, float* __restrict__ out, int n, int k,
                     int out_stride, int rpb, float* __restrict__ lut) {
    extern __shared__ uint8_t smem[];
    cg::cluster_group cluster = cg::this_cluster();

    const unsigned rank = cluster.block_rank();       // == g (activation row)
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;               // == rr (output row in block)
    const int nb_k = k >> 5;
    const int n16 = k >> 4;

    const int rid = (int)(blockIdx.x / M);           // this cluster's row-group
    const int row0 = rid * rpb;
    const int row = row0 + warp;

    uint8_t* s_w = smem;
    // LUT is built LOCALLY in every block (256 f32 = 1 KB, pure function of the
    // byte; building it locally is bit-identical to reading rank 0's copy and
    // avoids turning every per-element decode into a remote load).
    float* s_lut = reinterpret_cast<float*>(s_w + (size_t)rpb * (size_t)k);
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_lut[i] = lut[i];

    // --- rank 0 stages the weight slab into its own smem --------------------
    if (rank == 0) {
        const int avail = min(rpb, n - row0);
        const uint8_t* __restrict__ wsrc = w + (size_t)row0 * (size_t)k;
        const int nflat = avail * n16;
        for (int i = threadIdx.x; i < nflat; i += blockDim.x)
            __pipeline_memcpy_async(s_w + ((size_t)i << 4), wsrc + ((size_t)i << 4), 16);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();  // retire the slab inside rank 0 before publication
    }

    // Publish rank 0's smem to the whole cluster. `cluster.sync()` is the
    // arrive+wait rendezvous; nothing below reads a remote byte before it.
    cluster.sync();

    // The DSMEM handle: rank 0's `s_w` seen from every block.
    uint8_t* s_w_remote = (uint8_t*)cluster.map_shared_rank(s_w, 0);

    if (row < n) {
        const uint8_t* __restrict__ rs = s_w_remote + (size_t)warp * (size_t)k;
        const uint8_t* __restrict__ wsr = w_scale + (size_t)(row >> 5) * (size_t)nb_k;
        const uint8_t* __restrict__ ar = a + (size_t)rank * (size_t)k;
        const float* __restrict__ asr = a_scale + (size_t)rank * (size_t)nb_k;
        float acc = 0.f;
        #pragma unroll 32
        for (int kb = 0; kb < nb_k; ++kb) {
            const float sb = dsmem_ue8m0_to_f(wsr[kb]);
            const int j = kb * 32 + lane;
            const float wv = s_lut[rs[j]] * sb;
            const float av = s_lut[ar[j]] * asr[kb];
            acc += av * wv;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) {
            const float v = acc + (bias != nullptr ? bias[row] : 0.f);
            out[(size_t)rank * (size_t)out_stride + (size_t)row] = v;
        }
    }

    // Split barrier form (documented so the design pins the exact API): the
    // arrive/wait pair is what lets a block publish early and wait late.
    cluster.barrier_arrive();
    cluster.barrier_wait();
}

// ---------------------------------------------------------------------------
// Minimal API probe — the smallest kernel that exercises every primitive the
// design names, independent of the GEMV geometry. If the GEMV above ever fails
// to compile on a future toolkit, this one still isolates whether the failure
// is the cluster primitives or the kernel body.
// ---------------------------------------------------------------------------
__global__ void __cluster_dims__(6, 1, 1)
cluster_smoke_kernel(int* __restrict__ out) {
    extern __shared__ int smem_i[];
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned rank = cluster.block_rank();
    if (threadIdx.x == 0) smem_i[0] = (int)rank + 100;
    cluster.sync();
    int* remote = (int*)cluster.map_shared_rank(smem_i, 0);
    if (threadIdx.x == 0 && out != nullptr) out[rank] = remote[0];
}

// ---------------------------------------------------------------------------
// Host harness (NOT run in ⑤b acceptance). Kept so the file is a complete test
// binary for the later GPU round; guarded so `-c` compile-only is clean.
// ---------------------------------------------------------------------------
#ifndef FERRITE_CLUSTER_COMPILE_ONLY
static int host_check(void) {
    const int n = 12, k = 64, M = 6, rpb = 2;
    const int grid = M * ((n + rpb - 1) / rpb);
    const size_t sz_a = (size_t)M * k;
    const size_t sz_w = (size_t)n * k;
    uint8_t* a = nullptr;
    uint8_t* w = nullptr;
    float* a_scale = nullptr;
    uint8_t* w_scale = nullptr;
    float* out = nullptr;
    float* lut = nullptr;
    if (cudaMallocManaged(&a, sz_a) || cudaMallocManaged(&w, sz_w) ||
        cudaMallocManaged(&a_scale, (size_t)M * (k >> 5) * 4) ||
        cudaMallocManaged(&w_scale, (size_t)(n >> 5) * (k >> 5)) ||
        cudaMallocManaged(&out, (size_t)M * n * 4) ||
        cudaMallocManaged(&lut, 256 * 4)) {
        return 1;
    }
    for (size_t i = 0; i < sz_a; ++i) a[i] = (uint8_t)(i * 7 + 3);
    for (size_t i = 0; i < sz_w; ++i) w[i] = (uint8_t)(i * 5 + 1);
    for (int i = 0; i < M * (k >> 5); ++i) a_scale[i] = 1.0f;
    memset(w_scale, 0x7f, (size_t)(n >> 5) * (k >> 5));  // ue8m0 1.0
    for (int i = 0; i < 256; ++i) lut[i] = (float)(int8_t)(i & 0x0f);
    const size_t sh = (size_t)rpb * (size_t)k + 256 * 4;
    if (cudaFuncSetAttribute(cluster_mrows_kernel<M>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)sh)) {
        return 2;
    }
    cluster_mrows_kernel<M><<<grid, rpb * 32, sh>>>(a, a_scale, w, w_scale, nullptr, out, n, k, n,
                                                    rpb, lut);
    if (const cudaError_t e = cudaDeviceSynchronize()) return (int)e;

    // ALTERNATIVE LAUNCH SURFACE — pin it so the design can choose. The
    // compile-time `__cluster_dims__` above fixes the cluster size per
    // instantiation; the runtime attribute below lets ONE instantiation take a
    // cluster dim at LAUNCH (useful while M is still a swept knob). Both reach
    // the same PTX path.
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = M;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3((unsigned)(rpb * 32));
    cfg.dynamicSmemBytes = sh;
    cfg.stream = nullptr;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    if (const cudaError_t e =
            cudaLaunchKernelEx(&cfg, cluster_mrows_kernel<M>, a, a_scale, w, w_scale, nullptr, out,
                               n, k, n, rpb, lut)) {
        return (int)e;
    }
    if (const cudaError_t e = cudaDeviceSynchronize()) return (int)e;

    // OCCUPANCY — the design's binding constraint (`cudaOccupancyMaxActiveClusters`
    // must be >= grid / (SM count) at the chosen cluster size, else blocks of a
    // cluster cannot co-reside and the launch serialises or fails).
    cudaLaunchConfig_t occ_cfg = cfg;
    int clusters = 0;
    if (const cudaError_t e =
            cudaOccupancyMaxActiveClusters(&clusters, cluster_mrows_kernel<M>, &occ_cfg)) {
        return (int)e;
    }
    printf("[cluster_dsmem] max active clusters at N=%d = %d\n", M, clusters);
    return 0;
}

int main(void) {
    const int rc = host_check();
    printf("[cluster_dsmem] %s (rc=%d)\n", rc == 0 ? "OK" : "FAIL", rc);
    return rc;
}
#endif  // FERRITE_CLUSTER_COMPILE_ONLY
