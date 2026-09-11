// DeepSeek-V4.1-Flash MoE routing from pre-computed gate scores.
//
// The released checkpoint stores `ffn.gate.weight` in **bf16** — there is no
// `ffn.gate.scale` — so the gate GEMM runs on the bf16 tensor-core path and the
// routing decision is a separate step. `dsv41_moe_route` cannot serve this: it
// fuses an fp8 gate GEMM the checkpoint does not provide.
//
// Routing maths, exactly as the reference's Gate does it (`noaux_tc`):
//   act    = sqrtsoftplus
//   sel    = act(score) + gate_bias      the correction bias steers SELECTION
//   idx    = top-k(sel, k)               ties -> lower index (deterministic)
//   w      = act(score)[idx]             weights come from the UNBIASED values
//   w      = w / sum(w)                  when norm_topk_prob
//   w      = w * route_scale
//
// One block per row, 256 threads. `hist` (optional) counts assignments per
// expert so a caller can bucket them without a second scan.
//
// STATUS: this kernel is correct for n_experts <= blockDim (verified exactly at
// n_experts=6/topk=3) but picks a DIFFERENT expert set from the CPU reference at
// the production shape (n_experts=384/topk=6: 12/12 assignments differ, and they
// still differ with deliberately well-separated scores, so it is not a
// near-tie/precision artefact). The per-thread second candidate (threads
// 0..n_experts-blockDim-1 handle two experts) is the prime suspect. The earlier
// crash was a separate bug — a `used[]` flag array sized [topk] and indexed by
// the expert id — which is fixed (consumed experts are now marked by writing
// -INFINITY into s_sel).

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

__device__ __forceinline__ float dsv41_act(float v, int score_func) {
    if (score_func == 0) return 1.f / (1.f + expf(-v));   // sigmoid
    if (score_func == 2) {                                // sqrtsoftplus
        const float sp = log1pf(expf(-fabsf(v))) + fmaxf(v, 0.f);  // stable
        return sqrtf(fmaxf(sp, 0.f));
    }
    return v;                                             // identity
}

__global__ void route_topk_kernel(const float* __restrict__ scores,
                                  const float* __restrict__ bias, float* __restrict__ weights,
                                  int32_t* __restrict__ indices, int rows, int n_experts, int topk,
                                  int norm_topk_prob, float route_scale, int score_func) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    // layout: [n_experts] unbiased act | [n_experts] selection score | [topk] pick
    // A consumed expert is marked by writing -INFINITY into s_sel, so no
    // per-expert flag array is needed. (An earlier revision kept a `used` array
    // sized [topk] and indexed it by the expert id — an out-of-bounds smem
    // access that a small test shape happened not to trip.)
    extern __shared__ float sh[];
    float* s_act = sh;
    float* s_sel = s_act + n_experts;
    int* s_pick = (int*)(s_sel + n_experts);

    const float* sr = scores + (size_t)r * n_experts;
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        const float a = dsv41_act(sr[e], score_func);
        s_act[e] = a;
        s_sel[e] = a + (bias ? bias[e] : 0.f);
    }
    __syncthreads();

    __shared__ float s_bv[32];
    __shared__ int s_bi[32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int it = 0; it < topk; ++it) {
        float bv = -INFINITY;
        int bi = n_experts;
        for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
            const float v = s_sel[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_bv[warp] = bv;
            s_bi[warp] = bi;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (lane < nw) ? s_bv[lane] : -INFINITY;
            int i = (lane < nw) ? s_bi[lane] : n_experts;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, off);
                const int oi = __shfl_xor_sync(0xffffffffu, i, off);
                if (ov > v || (ov == v && oi < i)) {
                    v = ov;
                    i = oi;
                }
            }
            if (lane == 0) s_pick[it] = i;
        }
        __syncthreads();
        if (threadIdx.x == 0 && s_pick[it] < n_experts) {
            s_sel[s_pick[it]] = -INFINITY;  // consume it
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int t = 0; t < topk; ++t) {
            const int e = s_pick[t];
            sum += (e < n_experts) ? s_act[e] : 0.f;
        }
        for (int t = 0; t < topk; ++t) {
            const int e = s_pick[t];
            const float base = (e < n_experts) ? s_act[e] : 0.f;
            const float w = (norm_topk_prob && sum > 0.f) ? base / sum : base;
            indices[(size_t)r * topk + t] = e;
            weights[(size_t)r * topk + t] = w * route_scale;
        }
    }
}

__global__ void route_hist_kernel(const int32_t* __restrict__ indices, int32_t* __restrict__ hist,
                                  int n_asg, int n_experts) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_asg; i += gridDim.x * blockDim.x) {
        const int e = indices[i];
        if (e >= 0 && e < n_experts) atomicAdd(&hist[e], 1);
    }
}

}  // namespace

extern "C" int dsv41_route_topk(const float* scores, const float* bias, float* weights,
                                int32_t* indices, int32_t* hist, int rows, int n_experts, int topk,
                                int norm_topk_prob, float route_scale, int score_func,
                                cudaStream_t s) {
    if (rows <= 0 || n_experts <= 0 || topk <= 0 || topk > n_experts) {
        return (int)cudaErrorInvalidValue;
    }
    const size_t smem = (size_t)n_experts * 2 * sizeof(float) + (size_t)topk * sizeof(int);
    if (smem > 200 * 1024) return (int)cudaErrorInvalidValue;
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(route_topk_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return (int)e;
    }
    route_topk_kernel<<<rows, 256, smem, s>>>(scores, bias, weights, indices, rows, n_experts, topk,
                                              norm_topk_prob, route_scale, score_func);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    if (hist != nullptr) {
        cudaMemsetAsync(hist, 0, (size_t)n_experts * sizeof(int32_t), s);
        const int n_asg = rows * topk;
        route_hist_kernel<<<(n_asg + 255) / 256, 256, 0, s>>>(indices, hist, n_asg, n_experts);
    }
    return (int)cudaGetLastError();
}
