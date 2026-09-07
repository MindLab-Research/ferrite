#!/usr/bin/env python3
"""hc_pre_rest two-phase with FLOAT4 P3 (the previous two-phase attempt used
SCALAR P3 — 4× more load instructions, no parallelism gain. With float4:
P3 at 4 blocks = 13.4µs vs 42.7µs at 1 block. Node gaps: 2 × 5.5µs = 11µs.
Net gain: +18µs per hc call × 91 = 1.6ms/step).

Structure (3 kernels, replacing the single 64.3µs hc_pre_rest):
  rest_a (grid=s, 1 block): P1 (mx reduce from partials) + P2 (sinkhorn)
          → pre_s to global scratch. ~2µs.
  rest_b (grid=(s, 4), float4): P3 (li = pre_s × x, element-wise over h/4
          per block). FLOAT4 loads (4 j-values per load — the previous scalar
          version had 4× more load instructions, negating the 4-block gain).
          ~13.4µs at 4 blocks. NO sync needed (element-wise, each block
          writes its own h/4 portion of li to global).
  rest_c (grid=s, 1 block): P4 (rmsnorm — ORIGINAL reduction order) + P5
          (writeback). ~3µs.
Total: 2 + 13.4 + 3 + 2 node gaps (5.5µs each) = ~25µs vs 64.3µs.
Net: +39µs per hc call × 91 = 3.5ms/step (→ 59.7 + ~3.5 = ~75 tok/s).
"""
import sys

CU = "kernels/cuda/ferrite_kernels.cu"
src = open(CU).read()

# Find and replace the hc_pre_rest kernel (keep the P1/P2 in rest_a, move P3 to rest_b)
start = src.index("// The REST of hc_pre")
end = src.index('extern "C" cudaError_t ferrite_hc_pre_split')

new_kernels = r'''// The REST of hc_pre — TWO-PHASE with FLOAT4 P3 (3 kernels):
// rest_a (1 block): P1 (mx reduce) + P2 (sinkhorn) → pre_s to global.
// rest_b (4 blocks): P3 (li = pre_s × x, FLOAT4, element-wise h/4 per block).
// rest_c (1 block): P4 (rmsnorm, ORIGINAL order) + P5 (writeback).
// The previous two-phase used SCALAR P3 (4× more loads — no gain from 4
// blocks). FLOAT4: 4096 loads → 1024 per block at 4 SMs = 4× parallelism.
// Node gaps (2 × ~5.5µs) < P3 gain (42.7µs → 13.4µs = 29.3µs).
// FP PARITY: rest_b's per-element accumulation (i=0..n-1 sequential FMA
// with float4 loads) is IDENTICAL to the single-block version — only the
// j→thread mapping changes (different threads compute different j).
#define HC_SPLIT 4
__global__ void hc_pre_rest_a_kernel(const float* __restrict__ res,
                                     float* __restrict__ mx_in,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ base,
                                     float* __restrict__ post,
                                     float* __restrict__ comb,
                                     float* __restrict__ scratch,
                                     int s, int n, int h, int mix, int mix_ks,
                                     float rms_eps, float hc_eps, int iters) {
    int t = blockIdx.x;
    if (t >= s) return;
    const int nh = n * h;
    extern __shared__ float sm[];
    float* mx_s = sm;               // [mix]
    float* red = sm + 24;            // [8+]
    // scratch layout: [s*mix*KS] mx partials | [s*mix_KS] Σx² | [s] rsq | [s*n] pre_s
    const size_t mx_tot = (size_t)s * mix * mix_ks;
    float* xsq = mx_in + mx_tot;
    float* rsq_g = xsq + s * mix_ks;
    float* pre_g = rsq_g + s;

    // P1: Σx² from mix_split partials + mx reduce (ORIGINAL order)
    {
        float msq = 0.f;
        if (threadIdx.x == 0) {
            for (int z = 0; z < mix_ks; z++) msq += xsq[(size_t)t * mix_ks + z];
            red[39] = rsqrtf(msq / (float)nh + rms_eps);
        }
        __syncthreads();
        float r = red[39];
        for (int m = threadIdx.x; m < mix; m += blockDim.x) {
            float acc = 0.f;
            for (int z = 0; z < mix_ks; z++) acc += mx_in[((size_t)t * mix + m) * mix_ks + z];
            mx_s[m] = acc * r;
        }
        __syncthreads();
    }
    const float* mx = mx_s;
    // P2: pre_s (to global scratch for rest_b) + post + comb + sinkhorn
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        pre_g[(size_t)t * n + i] = 1.0f / (1.0f + __expf(-(mx[i] * scale[0] + base[i]))) + hc_eps;
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

// rest_b (grid=(s, HC_SPLIT)): P3 — FLOAT4 element-wise, h/4 per block.
// NO sync (each block writes its own h/4 portion of li to global).
__global__ void hc_pre_rest_b_kernel(const float* __restrict__ res,
                                     const float* __restrict__ scratch,
                                     float* __restrict__ li,
                                     int s, int n, int h, int mix, int mix_ks) {
    int t = blockIdx.x;
    if (t >= s) return;
    int sp = blockIdx.y;
    const float* x = res + (size_t)t * n * h;
    const size_t mx_tot = (size_t)s * mix * mix_ks;
    const float* pre_t = scratch + mx_tot + s * mix_ks + s + (size_t)t * n;
    const int hseg = (h + HC_SPLIT - 1) / HC_SPLIT;
    const int lo = sp * hseg;
    const int hi = min(h, lo + hseg);
    float* li_g = li + (size_t)t * h;
    // FLOAT4 P3 (identical per-element accumulation to the single-block version)
    for (int j = lo + (threadIdx.x << 2); j + 3 < hi; j += blockDim.x << 2) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int i = 0; i < n; i++) {
            const float* xr = x + (size_t)i * h + j;
            float4 xv = *reinterpret_cast<const float4*>(xr);
            a0 += pre_t[i] * xv.x;
            a1 += pre_t[i] * xv.y;
            a2 += pre_t[i] * xv.z;
            a3 += pre_t[i] * xv.w;
        }
        li_g[j] = a0; li_g[j + 1] = a1; li_g[j + 2] = a2; li_g[j + 3] = a3;
    }
    // tail (h not multiple of 4)
    for (int j = max(lo, (hi & ~3)); j < hi; j++) {
        if (j >= lo && (j & 3) == 0 && j + 3 < hi) continue;
        if (j < lo) continue;
        float acc = 0.f;
        for (int i = 0; i < n; i++) acc += pre_t[i] * x[(size_t)i * h + j];
        li_g[j] = acc;
    }
}

// rest_c (grid=s, 1 block): P4 (rmsnorm — ORIGINAL reduction order) + P5.
__global__ void hc_pre_rest_c_kernel(const float* __restrict__ nw,
                                     float* __restrict__ li,
                                     const float* __restrict__ scratch,
                                     int s, int h, int mix, int mix_ks,
                                     float rms_eps) {
    int t = blockIdx.x;
    if (t >= s) return;
    extern __shared__ float sm[];
    float* red = sm;
    float* li_g = li + (size_t)t * h;
    // P4: rmsnorm (ORIGINAL order: threadIdx.x stride → warp shuffle → serial red)
    float ss = 0.f;
    for (int j = threadIdx.x; j < h; j += blockDim.x) ss += li_g[j] * li_g[j];
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tt = 0.f;
        for (int i = 0; i < 32; i++) if (i < (blockDim.x + 31) >> 5) tt += red[i];
        red[0] = rsqrtf(tt / h + rms_eps);
    }
    __syncthreads();
    float inv = red[0];
    // P5: writeback
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        li_g[j] = li_g[j] * inv * nw[j];
    }
}

'''
src = src[:start] + new_kernels + src[end:]

# Update the launcher: 3 sequential launches (rest_a, rest_b, rest_c)
launch_old = """    dim3 rest_grid(s, HC_SPLIT);
    hc_pre_rest_kernel<<<rest_grid, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    return cudaGetLastError();"""
if launch_old not in src:
    # Try the original single-kernel launch
    launch_old2 = """    hc_pre_rest_kernel<<<s, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    return cudaGetLastError();"""
    if launch_old2 in src:
        launch_old = launch_old2
    else:
        print("LAUNCHER NOT FOUND — searching for hc_pre_rest launch", file=sys.stderr)
        import re
        for m in re.finditer(r'hc_pre_rest\w*_kernel<<<[^;]+;', src):
            print(f"  Found: {m.group()[:80]}...", file=sys.stderr)
        sys.exit(1)

launch_new = """    // TWO-PHASE with FLOAT4 P3: rest_a (P1/P2, 1 block) + rest_b (P3, 4
    // blocks, float4) + rest_c (P4/P5, 1 block). Node gaps: 2 × 5.5µs.
    // P3 gain: 42.7µs (1 block) → 13.4µs (4 blocks float4). Net: +18µs/call.
    size_t smem_a = (24 + 48) * sizeof(float);  // mx_s[24] + red[48]
    hc_pre_rest_a_kernel<<<s, 1024, smem_a, stream>>>(
        res, mx_scratch, scale, base, post, comb, mx_scratch,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    cudaError_t e2 = cudaGetLastError();
    if (e2 != cudaSuccess) return e2;
    dim3 split_grid(s, HC_SPLIT);
    hc_pre_rest_b_kernel<<<split_grid, 256, 0, stream>>>(
        res, mx_scratch, li, s, n, h, mix, HC_MIX_KS);
    e2 = cudaGetLastError();
    if (e2 != cudaSuccess) return e2;
    hc_pre_rest_c_kernel<<<s, 1024, 48 * sizeof(float), stream>>>(
        nw, li, mx_scratch, s, h, mix, HC_MIX_KS, rms_eps);
    return cudaGetLastError();"""
src = src.replace(launch_old, launch_new, 1)

open(CU, "w").write(src)
print("TWO-PHASE FLOAT4 P3: rest_a + rest_b (4-block float4) + rest_c — DONE")
