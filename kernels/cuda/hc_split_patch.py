#!/usr/bin/env python3
"""hc_pre_rest two-phase split (no cooperative — the grid.sync×4 overhead
exceeded the P3/P5 parallelism gain, 56.4 vs 61.9 tok/s; REVERTED).

This patch splits hc_pre_rest_kernel into THREE kernels:
  rest_a (grid=s, single block): P1 Σx²+mx reduce + P2 pre/post/cb/sinkhorn
          (ORIGINAL orders — parity-critical per the TILE FIX lesson)
  rest_b (grid=(s,4)): P3 li (the 64µs single-block hot spot: n×h reads
          now 4-way split; no reductions — pure per-j dot)
  rest_c (grid=s, single block): P4 Σli² ORIGINAL order + P5 writeback
Launches: 3 (was 1) × 90/step ≈ +0.8ms launch cost vs P3's 4× (−18µs×90
= −1.6ms) → net −0.8ms/step expected."""
import sys

CU = "kernels/cuda/ferrite_kernels.cu"
src = open(CU).read()

start = src.index("// The REST of hc_pre: reads pre-computed mx from global memory")
end = src.index('extern "C" cudaError_t ferrite_hc_pre_split')

new_kernels = r'''// The REST of hc_pre — TWO-PHASE SPLIT (three kernels, no cooperative:
// the coop attempt's 4× grid.sync overhead exceeded the P3 parallelism
// gain — 56.4 vs 61.9 tok/s, reverted). rest_a (single block, ORIGINAL
// orders — parity-critical per the TILE FIX lesson: rmsnorm partial order
// changes garbled the text) does Σx²+mx+sinkhorn; rest_b (grid=(s,4))
// does the latency-bound n×h li reads 4-way; rest_c (single block,
// ORIGINAL Σli² order) normalizes+writes back.
// Scratch (mx_scratch tail, cuda.rs allocs s*(mix*KS+1+n)):
//   rsq[t] | pre_s[t][n]  (rest_a writes, rest_b reads).
__global__ void hc_pre_rest_kernel(const float* __restrict__ res,
                                   const float* __restrict__ mx_in,
                                   const float* __restrict__ scale,
                                   const float* __restrict__ base,
                                   const float* __restrict__ nw,
                                   float* __restrict__ li,
                                   float* __restrict__ post,
                                   float* __restrict__ comb,
                                   float* __restrict__ scratch,
                                   int s, int n, int h, int mix, int mix_ks,
                                   float rms_eps, float hc_eps, int iters) {
    int t = blockIdx.x;
    if (t >= s) return;
    const float* x = res + (size_t)t * n * h;
    const int nh = n * h;
    extern __shared__ float sm[];
    float* mx_s = sm;               // [mix]
    float* red = sm + 24;          // [8] warp partials
    // scratch tail (passed in — sized s*(1+n) by the caller)
    const size_t mx_tot = (size_t)s * mix * mix_ks;
    float* rsq_g = scratch + mx_tot;          // [s]
    float* pre_s = rsq_g + s;                 // [s][n] global (rest_b reads)

    // P1: Σx² (ORIGINAL 1024-way interleaved order) + mx reduce + rsq
    {
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
    }
    const float* mx = mx_s;
    // P2: pre_s (to global scratch for rest_b) + post + comb
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

// rest_b (grid=(s, SPLITS)): the latency-bound li n×h reads, 4-way split.
// No reductions — pure per-j dot (pre_s broadcast from scratch).
#define HC_SPLITS 4
__global__ void hc_pre_rest_b_kernel(const float* __restrict__ res,
                                     const float* __restrict__ nw,
                                     float* __restrict__ li,
                                     const float* __restrict__ scratch,
                                     int s, int n, int h, int mix, int mix_ks) {
    int t = blockIdx.x;
    if (t >= s) return;
    int sp = blockIdx.y;
    int splits = gridDim.y;
    const float* x = res + (size_t)t * n * h;
    const size_t mx_tot = (size_t)s * mix * mix_ks;
    const float* pre_t = scratch + mx_tot + s + (size_t)t * n;  // pre_s[t][n]
    const int hseg = (h + splits - 1) / splits;
    const int lo = sp * hseg;
    const int hi = min(h, lo + hseg);
    float* li_g = li + (size_t)t * h;
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < n; i++) acc += pre_t[i] * x[(size_t)i * h + j];
        li_g[j] = acc;  // raw (unnormalized) — rest_c completes
    }
}

// rest_c (grid=s, single block): P4 Σli² ORIGINAL order + P5 writeback.
__global__ void hc_pre_rest_c_kernel(const float* __restrict__ nw,
                                     float* __restrict__ li,
                                     const float* __restrict__ scratch,
                                     int s, int n, int h, int mix, int mix_ks,
                                     float rms_eps) {
    int t = blockIdx.x;
    if (t >= s) return;
    extern __shared__ float sm[];
    float* red = sm;  // [8]
    float* li_g = li + (size_t)t * h;
    const size_t mx_tot = (size_t)s * mix * mix_ks;
    const float* rsq_g = scratch + mx_tot;
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
    float inv = rsq_g[t];
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        li_g[j] = li_g[j] * inv * nw[j];
    }
}

'''
src = src[:start] + new_kernels + src[end:]

# launch: replace the single hc_pre_rest_kernel launch with 3 launches
launch_old = """    // smem: mx_s[mix] + cb[n*n] + pre_s[n] + li_s[h] + red[40]  (~16.7KB)
    size_t smem2 = ((size_t)(mix + n * n + n) + h + 48) * sizeof(float);
    hc_pre_rest_kernel<<<s, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    return cudaGetLastError();"""
launch_new = """    // smem rest_a: mx_s[24] + red[8]; rest_c: red[8]. li staging is global.
    size_t smem2 = (24 + 8) * sizeof(float);
    hc_pre_rest_kernel<<<s, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb, mx_scratch,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    cudaError_t e2 = cudaGetLastError();
    if (e2 != cudaSuccess) return e2;
    // P3: the 64µs single-block hot spot (n×h reads) — 4-way split
    dim3 split_grid(s, HC_SPLITS);
    hc_pre_rest_b_kernel<<<split_grid, 256, 0, stream>>>(
        res, nw, li, mx_scratch, s, n, h, mix, HC_MIX_KS);
    e2 = cudaGetLastError();
    if (e2 != cudaSuccess) return e2;
    // P4+P5: Σli² ORIGINAL order + writeback (single block)
    hc_pre_rest_c_kernel<<<s, 1024, 8 * sizeof(float), stream>>>(
        nw, li, mx_scratch, s, n, h, mix, HC_MIX_KS, rms_eps);
    return cudaGetLastError();"""
if launch_old not in src:
    print("LAUNCH BLOCK NOT FOUND — aborting", file=sys.stderr)
    sys.exit(1)
src = src.replace(launch_old, launch_new, 1)

open(CU, "w").write(src)
print("hc_pre_rest two-phase split OK")
