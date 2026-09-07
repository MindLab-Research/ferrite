#!/usr/bin/env python3
"""hc_pre_mix_split FUSION (Version A — P4/P5 separate for FP safety):
The mix_split (192 blocks) does the mix computation + counter (last block
detection) + P1/P2 (last block: mx reduce + sinkhorn → pre_s to global) +
flag spin + P3 (all blocks: h/192 elements each, li to global UNNORMALIZED).
The P4/P5 (rmsnorm + writeback) stays in a separate 1-block kernel
(hc_pre_p45) with the EXACT SAME summation order as the current hc_pre_rest
(1024 threads, threadIdx.x stride, warp shuffle — FP SAFE).

Eliminates: the hc_pre_rest's 64.3µs 1-block P3 (42µs L2 reads at 1 SM) →
0.2µs at 192 blocks. The P1/P2 moves from the hc_pre_rest to the mix_split's
last block (no extra kernel). The P4/P5 is a new small kernel (15µs vs the
hc_pre_rest's P4/P5 at 7µs — the P4 reads li from global instead of shared,
+8µs). The node gap between the mix_split and the p45: 18µs (same as the
current gap between the mix_split and the hc_pre_rest).

Expected: mix+P3 (19µs) + gap (18µs) + p45 (20µs) = 57µs vs current 87.6µs.
"""
import sys

CU = "kernels/cuda/ferrite_kernels.cu"
src = open(CU).read()

# 1. Modify hc_pre_mix_split_kernel: add counter + P1/P2 (last block) + flag + P3
mix_start = src.index("__global__ void hc_pre_mix_split_kernel(")
mix_end = src.index("// The REST of hc_pre")

new_mix = r'''__global__ void hc_pre_mix_split_kernel(const float* __restrict__ res,
                                        const float* __restrict__ fw,
                                        float* __restrict__ mx_partial,
                                        // FUSED P1/P2/P3 params (v10: mix_split fusion)
                                        const float* __restrict__ scale,
                                        const float* __restrict__ base,
                                        float* __restrict__ li,
                                        float* __restrict__ post,
                                        float* __restrict__ comb,
                                        int s, int n, int h, int mix,
                                        float rms_eps, float hc_eps, int iters) {
    // K-SPLIT: gridDim.z = KS lanes per mix row — 24 mix rows × 8 lanes =
    // 192 blocks (130% SM). Each lane dots its 1/KS segment.
    const int KS = gridDim.z;
    int t = blockIdx.x;
    int m = blockIdx.y;
    int z = blockIdx.z;
    if (t >= s || m >= mix) return;
    const float* x = res + (size_t)t * n * h;
    const int nh = n * h;
    const float* row = fw + (size_t)m * nh;
    int seg = (nh + KS - 1) / KS;
    int lo = z * seg;
    int hi = min(lo + seg, nh);

    float acc = 0.f;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) acc += row[i] * x[i];
    __shared__ float red[8];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int w = 0; w < 8; w++) if (w < (blockDim.x + 31) >> 5) tot += red[w];
        mx_partial[((size_t)t * mix + m) * KS + z] = tot;
    }
    // Σx² fusion (m==0 lane only): rides free on the existing x reads
    if (m == 0) {
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

    // ═══ FUSED P1/P2/P3 (v10: the hc_pre_rest's 64.3µs 1-block P3 → 192
    // blocks × h/192 elements each = 0.2µs). The "last block" (atomic
    // counter) does P1/P2 (mx reduce + sinkhorn → pre_s to global scratch);
    // all blocks spin on a flag then do P3 (their h/gridDim.y/KS slice).
    // The P4/P5 stays in hc_pre_p45 (separate kernel, FP-safe summation). ═══
    {
        // scratch layout (mx_partial tail):
        //   [s*mix*KS + s*KS]: Σx² partials (existing)
        //   [+s]: counter (atomic, "last block" detection)
        //   [+s]: flag (P1/P2 done)
        //   [+s*n]: pre_s (from P2, for P3)
        //   [+s*n + s]: rsq (from P1, for P4/P5 — NOT used in v10)
        const size_t mx_tot = (size_t)s * mix * KS;
        float* xsq = mx_partial + mx_tot;                       // [s][KS]
        volatile int* cntr = (volatile int*)(xsq + s * KS);     // [s]
        volatile int* flag = (volatile int*)(xsq + s * KS + s);  // [s]
        float* pre_g = xsq + s * KS + 2 * s;                     // [s][n]

        // Counter: each block atomicAdd(1). The LAST block (return == total)
        // does P1/P2 (mx reduce + sinkhorn) → pre_s to global + flag.
        __threadfence();  // mx_partial writes visible before the counter
        __syncthreads();
        int total_blocks = gridDim.y * gridDim.z;  // mix * KS = 192
        int is_last = 0;
        if (threadIdx.x == 0) {
            int old = atomicAdd((int*)&cntr[t], 1);
            is_last = (old == total_blocks - 1) ? 1 : 0;
        }
        __shared__ int is_last_sh;
        if (threadIdx.x == 0) is_last_sh = is_last;
        __syncthreads();
        if (is_last_sh) {
            // ═══ P1 (mx reduce from partials + Σx² from xsq) + P2 (sinkhorn) ═══
            // Same code as the hc_pre_rest's P1/P2 (1 block, 256 threads).
            // The mx reduce: 24 partials × 8 KS (from mx_partial).
            // The sinkhorn: 4×4 matrix on thread 0.
            // The pre_s: to global scratch (for the P3 by all blocks).
            float msq = 0.f;
            if (threadIdx.x == 0) {
                for (int zz = 0; zz < KS; zz++) msq += xsq[(size_t)t * KS + zz];
                // rsq (stored for the P4/P5 kernel — NOT used in the P3)
                // The P4/P5 kernel recomputes rsq from li (not from here).
            }
            // mx reduce: thread m sums its mix row's KS partials
            float r = 1.f;  // rsq (recomputed below with the full Σx²)
            if (threadIdx.x == 0) {
                float msq2 = 0.f;
                for (int zz = 0; zz < KS; zz++) msq2 += xsq[(size_t)t * KS + zz];
                r = rsqrtf(msq2 / (float)nh + rms_eps);
            }
            __syncthreads();
            // r is only valid on thread 0 — broadcast via shared
            __shared__ float r_sh;
            if (threadIdx.x == 0) r_sh = r;
            __syncthreads();
            r = r_sh;
            // mx_s: reduce the mix partials (24 rows × 8 KS)
            __shared__ float mx_s[64];  // [mix] (max 64 mix rows)
            for (int mm = threadIdx.x; mm < mix; mm += blockDim.x) {
                float a = 0.f;
                for (int zz = 0; zz < KS; zz++) a += mx_partial[((size_t)t * mix + mm) * KS + zz];
                mx_s[mm] = a * r;
            }
            __syncthreads();
            // P2: pre_s + post + comb + sinkhorn (thread 0 for the sinkhorn)
            const float* mx = mx_s;
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
            __threadfence();  // pre_g writes visible before the flag
            __syncthreads();
            if (threadIdx.x == 0) atomicExch((int*)&flag[t], 1);
        }

        // ═══ All blocks: spin on flag, then P3 (h/total_blocks elements each) ═══
        if (threadIdx.x == 0) {
            while (atomicAdd((int*)&flag[t], 0) < 1) __nanosleep(100);
        }
        __syncthreads();
        // P3: li[j] = Σ_i pre_s[i] × x[i][j] (each block handles its slice)
        // The flat block index: m * KS + z (0..total_blocks-1)
        int flat = m * KS + z;
        int hseg = (h + total_blocks - 1) / total_blocks;
        int jlo = flat * hseg;
        int jhi = min(h, jlo + hseg);
        const float* pre_t = pre_g + (size_t)t * n;
        float* li_g = li + (size_t)t * h;
        for (int j = jlo + threadIdx.x; j < jhi; j += blockDim.x) {
            float a = 0.f;
            for (int i = 0; i < n; i++) a += pre_t[i] * x[(size_t)i * h + j];
            li_g[j] = a;  // UNNORMALIZED (the P4/P5 kernel normalizes)
        }
    }
    (void)rms_eps;
}

'''
src = src[:mix_start] + new_mix + src[mix_end:]

# 2. Add the hc_pre_p45 kernel (P4: rmsnorm + P5: writeback — FP SAFE:
#    same 1024-thread warp-shuffle summation order as the original hc_pre_rest)
p45_marker = "// The REST of hc_pre"
p45_kernel = r'''// P4/P5 kernel (v10): rmsnorm + writeback — separated from the mix_split
// fusion for FP SAFETY (the summation order is EXACTLY the original
// hc_pre_rest's P4/P5: 1024 threads × threadIdx.x stride → warp shuffle →
// serial red[8]). Reads the UNNORMALIZED li from global (written by the
// mix_split's P3), computes Σli², rsqrt, and writes li × rsq × nw.
__global__ void hc_pre_p45_kernel(const float* __restrict__ nw,
                                   float* __restrict__ li,
                                   int s, int h, float rms_eps) {
    int t = blockIdx.x;
    if (t >= s) return;
    extern __shared__ float sm[];
    float* red = sm;
    float* li_g = li + (size_t)t * h;
    // P4: Σli² (ORIGINAL order: threadIdx.x stride → warp shuffle → serial red)
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
    // P5: writeback (li × rsq × nw)
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        li_g[j] = li_g[j] * inv * nw[j];
    }
}

// The REST of hc_pre'''
src = src.replace(p45_marker, p45_kernel, 1)

# 3. Update ferrite_hc_pre_split: call the modified mix_split + hc_pre_p45
# (remove the hc_pre_rest launch, add the p45 launch)
old_launch = """    dim3 mix_grid(s, mix, HC_MIX_KS);
    hc_pre_mix_split_kernel<<<mix_grid, 256, 0, stream>>>(
        res, fw, mx_scratch, s, n, h, mix, rms_eps);"""
new_launch = """    dim3 mix_grid(s, mix, HC_MIX_KS);
    hc_pre_mix_split_kernel<<<mix_grid, 256, 0, stream>>>(
        res, fw, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, rms_eps, hc_eps, iters);"""
if old_launch not in src:
    print("MIX LAUNCH NOT FOUND — searching", file=sys.stderr)
    import re
    for m2 in re.finditer(r'hc_pre_mix_split_kernel<<<[^;]+;', src):
        print(f"  Found: {m2.group()[:100]}...", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_launch, new_launch, 1)

# Replace the hc_pre_rest launch with the p45 launch
old_rest = """    hc_pre_rest_kernel<<<s, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    return cudaGetLastError();"""
new_rest = """    // P4/P5: rmsnorm + writeback (FP SAFE — same summation order as the
    // original hc_pre_rest's P4/P5). Reads the UNNORMALIZED li from global.
    hc_pre_p45_kernel<<<s, 1024, 48 * sizeof(float), stream>>>(
        nw, li, s, h, rms_eps);
    return cudaGetLastError();"""
if old_rest not in src:
    print("REST LAUNCH NOT FOUND — searching", file=sys.stderr)
    import re
    for m2 in re.finditer(r'hc_pre_rest_kernel<<<[^;]+;', src):
        print(f"  Found: {m2.group()[:100]}...", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_rest, new_rest, 1)

# 4. Update the mx_scratch allocation comment (the scratch now includes
#    counter + flag + pre_s — the Rust side allocates s*(mix*8 + 8 + 2 + n))
# (The Rust allocation is in cuda.rs — needs a separate update)

open(CU, "w").write(src)
print("mix_split FUSION v10: counter + P1/P2 (last block) + flag + P3 (all blocks) + hc_pre_p45 (FP-safe P4/P5)")
