#!/usr/bin/env python3
"""P12-in-mix_split: the mix_split's last block does P1/P2 (mx reduce + sinkhorn
+ pre_s/post/comb to global scratch). The hc_pre_rest becomes P3/P4/P5 only
(reads pre_s from global). The atomic counter (192 blocks increment, last
block does P1/P2 — no spin, no deadlock for s=1; s>1 skips via early return).

This is the ENABLING STEP for the GEMV prologue fusion (P3 in the GEMV's
multi-block grid): pre_s available in global scratch → the GEMV can compute
li = pre_s × x in its prologue (element-wise, distributed across the GEMV's
blocks) + deferred rsq. The full fusion saves 5.22ms/step (+28 tok/s).
"""
import sys

CU = "kernels/cuda/ferrite_kernels.cu"
src = open(CU).read()

# Find the mix_split kernel's end (the closing brace after the Σx² fusion)
marker = """    if (m == 0) {
        float sq = 0.f;
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) sq += x[i] * x[i];
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot2 = 0.f;
            for (int w = 0; w < 8; w++) if (w < (blockDim.x + 31) >> 5) tot2 += red[w];
            mx_partial[(size_t)s * mix * KS + (size_t)t * KS + z] = tot2;
        }
    }
    (void)rms_eps;
}"""

if marker not in src:
    print("MARKER NOT FOUND — searching for the mix_split end", file=sys.stderr)
    # Try a partial match
    import re
    m = re.search(r'if \(m == 0\) \{\s*float sq.*?\(void\)rms_eps;\s*\}', src, re.DOTALL)
    if m:
        print(f"Found via regex at {m.start()}-{m.end()}", file=sys.stderr)
    sys.exit(1)

replacement = """    if (m == 0) {
        float sq = 0.f;
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) sq += x[i] * x[i];
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot2 = 0.f;
            for (int w = 0; w < 8; w++) if (w < (blockDim.x + 31) >> 5) tot2 += red[w];
            mx_partial[(size_t)s * mix * KS + (size_t)t * KS + z] = tot2;
        }
    }

    // ═══ P12-in-mix_split (v12): the LAST BLOCK does P1/P2 (mx reduce +
    // sinkhorn + pre_s/post/comb to the global scratch). No spin, no deadlock
    // (s>1 prefill skips via the early return — only s==1 decode fuses).
    // This is the ENABLING STEP for the GEMV prologue fusion: pre_s in the
    // global scratch → the GEMV's prologue can compute li = pre_s × x in
    // its multi-block grid (the hc_pre_rest's 42µs P3 distributed across
    // the GEMV's blocks). The atomic counter: the 192 blocks increment,
    // the last block (counter == 192) does the P1/P2. The FP is IDENTICAL
    // to the hc_pre_rest's P1/P2 (the same mx reduce order, the same
    // sinkhorn iteration order — the same pre_s/post/comb values). ═══
    if (s == 1) {  // decode only (s>1 prefill: too many blocks, skip — the
                   // hc_pre_rest handles P1/P2 for prefill via its normal path)
        __threadfence();  // the mix partials + Σx² are visible before the counter
        __syncthreads();
        __shared__ int is_last;
        if (threadIdx.x == 0) {
            unsigned prev = atomicAdd(ctr, 1u);
            is_last = (prev == (unsigned)(mix * KS - 1)) ? 1 : 0;
        }
        __syncthreads();
        if (is_last) {
            // ═══ P1: mx reduce (24 rows × 8 KS partials → mx_s) + Σx² → rsq ═══
            // (the IDENTICAL mx reduce order as the hc_pre_rest's P1: the thread 0
            // reads the partials serially, the same accumulation order — FP-safe)
            extern __shared__ float p12_sm[];  // mx_s[mix] + red[48]
            float* mx_s = p12_sm;               // [mix]
            float* red2 = p12_sm + mix;         // [48] (warp partials)
            float msq = 0.f;
            if (threadIdx.x == 0) {
                const float* xsq = mx_partial + (size_t)s * mix * KS;
                for (int z2 = 0; z2 < KS; z2++) msq += xsq[(size_t)t * KS + z2];
                red2[39] = rsqrtf(msq / (float)nh + rms_eps);
            }
            __syncthreads();
            float r = red2[39];
            for (int m2 = threadIdx.x; m2 < mix; m2 += blockDim.x) {
                float acc = 0.f;
                for (int z2 = 0; z2 < KS; z2++) acc += mx_partial[((size_t)t * mix + m2) * KS + z2];
                mx_s[m2] = acc * r;
            }
            __syncthreads();
            // ═══ P2: pre_s + post + comb (the sinkhorn) — the IDENTICAL order
            // as the hc_pre_rest's P2 (the same sigmoid, the same 4-iteration
            // sinkhorn, the same output values — FP-safe) ═══
            const float* mx = mx_s;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                pre_s_g[(size_t)t * n + i] = 1.0f / (1.0f + __expf(-(mx[i] * scale[0] + base[i]))) + hc_eps;
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
    }
    (void)rms_eps;
}"""

src = src.replace(marker, replacement, 1)

# Now add the new parameters to the mix_split kernel signature
old_sig = """__global__ void hc_pre_mix_split_kernel(const float* __restrict__ res,
                                        const float* __restrict__ fw,
                                        float* __restrict__ mx_partial,
                                        int s, int n, int h, int mix,
                                        float rms_eps) {"""
new_sig = """__global__ void hc_pre_mix_split_kernel(const float* __restrict__ res,
                                        const float* __restrict__ fw,
                                        float* __restrict__ mx_partial,
                                        // P12-in-mix_split params (v12): the last block does P1/P2
                                        const float* __restrict__ scale,
                                        const float* __restrict__ base,
                                        float* __restrict__ pre_s_g,  // [s][n] output
                                        float* __restrict__ post,     // [s][n] output
                                        float* __restrict__ comb,     // [s][n*n] output
                                        unsigned* __restrict__ ctr,   // [1] atomic counter
                                        int s, int n, int h, int mix,
                                        float rms_eps, float hc_eps, int iters) {"""
if old_sig not in src:
    print("SIG NOT FOUND", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_sig, new_sig, 1)

# Also need to zero the counter before each invocation (cudaMemsetAsync in the Rust side)
open(CU, "w").write(src)
print("P12-in-mix_split written (the last block does P1/P2 → pre_s/post/comb to global)")
