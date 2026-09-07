#!/usr/bin/env python3
"""Replace hc_pre_rest_kernel (single-block) with a cooperative multi-block
version (grid=(s, HC_SPLITS)) + cudaLaunchCooperativeKernel in
ferrite_hc_pre_split. Parity-critical: all reduction ORDERS preserved
(P1/P4 stay on block sp==0 with the original interleaved orders — the
TILE FIX lesson)."""
import re, sys

CU = "kernels/cuda/ferrite_kernels.cu"
src = open(CU).read()

# 1. include cooperative_groups
if "cooperative_groups.h" not in src:
    src = src.replace(
        "#include <cuda_runtime.h>",
        "#include <cuda_runtime.h>\n#include <cooperative_groups.h>\nnamespace cg = cooperative_groups;",
        1,
    )

# 2. replace the whole hc_pre_rest_kernel body (from its comment header to
# the closing brace before ferrite_hc_pre_split)
start = src.index("// The REST of hc_pre: reads pre-computed mx from global memory")
end = src.index('extern "C" cudaError_t ferrite_hc_pre_split')
new_kernel = r'''// The REST of hc_pre (COOPERATIVE multi-block: grid=(s, HC_SPLITS)).
// nsys (MoE-TP decode): hc_pre_rest was 5.8ms/step (90 calls × 64.4µs —
// 37% of the 15.6ms replay) at grid=s = ONE block on 148 SMs (0.7% SM).
// The HC_SPLITS blocks parallelize the latency-bound n×h reads (li) and
// the Σx²/Σli² prologues stay on block sp==0 in the ORIGINAL interleaved
// reduction orders (parity-critical — the TILE FIX lesson: partial-order
// changes garbled the text). Scratch (mx_scratch tail): rsq[t] | pre_s[t][n].
#define HC_SPLITS 4
__global__ void hc_pre_rest_kernel(const float* __restrict__ res,
                                   const float* __restrict__ mx_in,
                                   const float* __restrict__ scale,
                                   const float* __restrict__ base,
                                   const float* __restrict__ nw,
                                   float* __restrict__ li,
                                   float* __restrict__ post,
                                   float* __restrict__ comb,
                                   int s, int n, int h, int mix, int mix_ks,
                                   float rms_eps, float hc_eps, int iters) {
    cg::grid_group g = cg::this_grid();
    const int t = blockIdx.x;
    if (t >= s) return;
    const int sp = blockIdx.y;              // split 0..HC_SPLITS-1
    const int splits = gridDim.y;           // = HC_SPLITS
    const float* x = res + (size_t)t * n * h;
    const int nh = n * h;
    extern __shared__ float sm[];
    float* red = sm;                        // [8] warp partials
    float* mx_s = sm + 8;                   // [mix] (block 0's P2)
    float* li_seg = sm + 8 + 24;            // [h/splits] P3 staging

    // scratch layout (mx_scratch tail — cuda.rs allocs s*(mix*KS+1+n)):
    const size_t mx_tot = (size_t)s * mix * mix_ks;
    float* rsq_g = mx_in + mx_tot;                       // [s]
    float* pre_s = rsq_g + s;                            // [s][n]

    // P1 (block sp==0 ONLY — ORIGINAL orders: 1024-way interleaved Σx² +
    // mx reduce + pre/post/cb/sinkhorn; do NOT parallelize across splits).
    if (sp == 0) {
        float part = 0.f;
        for (int i = threadIdx.x; i < nh; i += blockDim.x) part += x[i] * x[i];
        for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffff, part, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = part;
        __syncthreads();
        float msq = 0.f;
        if (threadIdx.x == 0) {
            for (int w = 0; w < 32; w++) if (w < (blockDim.x + 31) >> 5) msq += red[w];
            rsq_g[t] = rsqrtf(msq / (float)nh + rms_eps);
        }
        __syncthreads();
        float r = rsq_g[t];
        for (int m = threadIdx.x; m < mix; m += blockDim.x) {
            float acc = 0.f;
            for (int z = 0; z < mix_ks; z++) acc += mx_in[((size_t)t * mix + m) * mix_ks + z];
            mx_s[m] = acc * r;
        }
        __syncthreads();
        const float* mx = mx_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            pre_s[(size_t)t * n + i] = 1.0f / (1.0f + __expf(-(mx[i] * scale[0] + base[i]))) + hc_eps;
            post[t * n + i] = 2.0f * (1.0f / (1.0f + __expf(-(mx[n + i] * scale[1] + base[n + i]))));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float cb[64];
            for (int i = 0; i < n; i++)
                for (int k = 0; k < n; k++)
                    cb[i * n + k] = mx[2 * n + i * n + k] * scale[2] + base[2 * n + i * n + k];
            for (int i = 0; i < n; i++) {
                float rmax = -INFINITY;
                for (int k = 0; k < n; k++) rmax = fmaxf(rmax, cb[i * n + k]);
                float denom = 0.f;
                for (int k = 0; k < n; k++) { cb[i * n + k] = __expf(cb[i * n + k] - rmax); denom += cb[i * n + k]; }
                for (int k = 0; k < n; k++) cb[i * n + k] = cb[i * n + k] / denom + hc_eps;
            }
            for (int k = 0; k < n; k++) {
                float colsum = 0.f;
                for (int i = 0; i < n; i++) colsum += cb[i * n + k];
                float d = colsum + hc_eps;
                for (int i = 0; i < n; i++) cb[i * n + k] /= d;
            }
            for (int it = 1; it < iters; it++) {
                for (int i = 0; i < n; i++) {
                    float rowsum = 0.f;
                    for (int k2 = 0; k2 < n; k2++) rowsum += cb[i * n + k2];
                    float d = rowsum + hc_eps;
                    for (int k2 = 0; k2 < n; k2++) cb[i * n + k2] /= d;
                }
                for (int k2 = 0; k2 < n; k2++) {
                    float colsum = 0.f;
                    for (int i = 0; i < n; i++) colsum += cb[i * n + k2];
                    float d = colsum + hc_eps;
                    for (int i = 0; i < n; i++) cb[i * n + k2] /= d;
                }
            }
            for (int i = 0; i < n * n; i++) comb[(size_t)t * n * n + i] = cb[i];
        }
    }
    g.sync();
    // P3 (each block): li over this split's h segment — the n×h reads (the
    // 64.4µs single-block hot spot) now HC_SPLITS-way parallel.
    {
        const float* pre_t = pre_s + (size_t)t * n;
        const int hseg = (h + splits - 1) / splits;
        const int lo = sp * hseg;
        const int hi = min(h, lo + hseg);
        float* li_g = li + (size_t)t * h;
        for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
            float acc = 0.f;
            for (int i = 0; i < n; i++) acc += pre_t[i] * x[(size_t)i * h + j];
            li_g[j] = acc; // raw (unnormalized) — P4/P5 complete the tail
        }
    }
    g.sync();
    // P4 (block sp==0 — ORIGINAL rmsnorm reduction order: 1024-way
    // interleaved over the full h from GLOBAL li; parity-critical).
    if (sp == 0) {
        float* li_g = li + (size_t)t * h;
        float ss = 0.f;
        for (int j = threadIdx.x; j < h; j += blockDim.x) ss += li_g[j] * li_g[j];
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tt = 0.f;
            for (int i = 0; i < 32; i++) if (i < (blockDim.x + 31) >> 5) tt += red[i];
            rsq_g[t] = rsqrtf(tt / h + rms_eps);
        }
        __syncthreads();
    }
    g.sync();
    // P5 (each block): li × rsq × nw writeback over this split's segment
    {
        const float inv = rsq_g[t];
        const int hseg = (h + splits - 1) / splits;
        const int lo = sp * hseg;
        const int hi = min(h, lo + hseg);
        float* li_g = li + (size_t)t * h;
        for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
            li_g[j] = li_g[j] * inv * nw[j];
        }
    }
}

'''
src = src[:start] + new_kernel + src[end:]

# 3. replace the phase-2 launch (<<<s, 1024, smem2, stream>>>) with
# cudaLaunchCooperativeKernel(grid=(s, HC_SPLITS))
launch_old = """    // smem: mx_s[mix] + cb[n*n] + pre_s[n] + li_s[h] + red[40]  (~16.7KB)
    size_t smem2 = ((size_t)(mix + n * n + n) + h + 48) * sizeof(float);
    hc_pre_rest_kernel<<<s, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    return cudaGetLastError();"""
launch_new = """    // smem: red[8] + mx_s[24] (P2, block 0 only) — li staging is per-split
    // global (li written raw by P3, normalized in place by P5).
    size_t smem2 = (8 + 24) * sizeof(float);
    // COOPERATIVE launch (grid=(s, HC_SPLITS)): the P3/P5 n×h reads/rites
    // are HC_SPLITS-way parallel; P1/P4 (order-critical reductions) stay
    // on block sp==0. cudaLaunchCooperativeKernel is stream-capture safe
    // (mega graph records it as a kernel node).
    {
        dim3 grid(s, HC_SPLITS);
        int s_v = s, n_v = n, h_v = h, mix_v = mix, mixks_v = HC_MIX_KS, iters_v = iters;
        float rms_v = rms_eps, eps_v = hc_eps;
        void* args[] = { (void*)&res, (void*)&mx_scratch, (void*)&scale, (void*)&base,
                         (void*)&nw, (void*)&li, (void*)&post, (void*)&comb,
                         &s_v, &n_v, &h_v, &mix_v, &mixks_v, &rms_v, &eps_v, &iters_v };
        cudaError_t e = cudaLaunchCooperativeKernel(
            (void*)hc_pre_rest_kernel, grid, dim3(1024), args, smem2, stream);
        if (e != cudaSuccess) return e;
    }
    return cudaGetLastError();"""
if launch_old not in src:
    print("LAUNCH BLOCK NOT FOUND — aborting", file=sys.stderr)
    sys.exit(1)
src = src.replace(launch_old, launch_new, 1)

open(CU, "w").write(src)
print("kernel replaced OK")
