#!/usr/bin/env python3
"""Clean 4-block atomic-sync hc_pre_rest rewrite (replaces the messy patch):
Block 0 (sp=0): P1 (mx+Σx²) + P2 (pre_s+sinkhorn) → flag → P3-seg-0 →
wait counter==3 → P4 (rmsnorm ALL h) + P5 (writeback ALL h) → reset flags.
Blocks 1-3 (sp=1..3): wait flag → P3-seg-1..3 → counter++.
Critical path: P1+P2 (2µs) + P3 (16µs, 4 SMs concurrent) + P4+P5 (3µs) ≈ 21µs
vs single-block 61.8µs → saves ~40µs × 90/step = 3.6ms/step."""
import sys

CU = "kernels/cuda/ferrite_kernels.cu"
src = open(CU).read()

# Find and replace the entire hc_pre_rest_kernel (from comment to the next extern)
start = src.index("// The REST of hc_pre")
end = src.index('extern "C" cudaError_t ferrite_hc_pre_split')

new_kernel = r'''// The REST of hc_pre — 4-BLOCK ATOMIC-SYNC SPLIT (grid=(s, HC_SPLIT)):
// nsys: hc_pre_rest was 61.8µs × 90/step = 5.6ms/step (35% of decode) at
// grid=s = ONE block on 148 SMs. The P3 (li: n=4 × h=4096 = 64KB reads
// at 1 SM's ~32 outstanding loads × 500ns L2 latency ≈ 64 batches) is
// the dominant cost. 4 blocks parallelize P3 across 4 SMs (~16µs).
// Sync: atomic flag/counter in global scratch (NOT cooperative launch —
// grid.sync×4 measured +5µs each = 20µs overhead making it SLOWER; NOT
// separate launches — 3×5µs launch overhead also SLOWER). Atomic spin
// ~0.5µs per handshake.
// FP PARITY: P3's per-element accumulation (i=0..n-1 sequential FMA) and
// P4's rmsnorm reduction (threadIdx.x stride → warp shuffle → serial
// red[]) are IDENTICAL to the single-block version — only the j→thread
// mapping changes (different threads compute different j, same formula).
// SCRATCH (mx_scratch tail): flag[s] + cntr[s] as int32 + pre_s[s][n].
// The mix_split's (t,0,0) block zeroes flag/cntr (free — mix_split runs
// before rest in the same stream).
#define HC_SPLIT 4
__global__ void hc_pre_rest_kernel(const float* __restrict__ res,
                                   float* __restrict__ mx_in,
                                   const float* __restrict__ scale,
                                   const float* __restrict__ base,
                                   const float* __restrict__ nw,
                                   float* __restrict__ li,
                                   float* __restrict__ post,
                                   float* __restrict__ comb,
                                   int s, int n, int h, int mix, int mix_ks,
                                   float rms_eps, float hc_eps, int iters) {
    const int t = blockIdx.x;
    if (t >= s) return;
    const int sp = blockIdx.y;              // 0..HC_SPLIT-1
    const float* x = res + (size_t)t * n * h;
    const int nh = n * h;

    // scratch layout (mx_scratch tail — cuda.rs allocs s*(mix*KS+KS+2+n)):
    //   mx:    [0, s*mix*KS)       — mix partials (mix_split writes)
    //   xsq:   [s*mix*KS, +s*KS)   — Σx² partials (mix_split m==0 lanes)
    //   flag:  [+s)  as int32      — block 0 → blocks 1-3 handshake
    //   cntr:  [+s)  as int32      — blocks 1-3 → block 0 handshake
    //   pre_s: [+s*n)              — block 0 P2 → blocks 1-3 P3
    const size_t mx_tot = (size_t)s * mix * mix_ks;
    float* xsq = mx_in + mx_tot;                       // [s][KS]
    volatile int* flag = (volatile int*)(xsq + s * mix_ks);         // [s]
    volatile int* cntr = (volatile int*)(xsq + s * mix_ks + s);     // [s]
    float* pre_g = xsq + s * mix_ks + 2 * s;            // [s][n]

    if (sp == 0) {
        // ═══ BLOCK 0: P1 (mx reduce + Σx²) + P2 (pre_s + sinkhorn) ═══
        extern __shared__ float sm[];
        float* mx_s = sm;               // [mix]
        float* red = sm + 24;            // [8+] warp partials
        // Σx² from mix_split's fused partials (8 adds, not 16K re-read)
        float msq = 0.f;
        if (threadIdx.x == 0) {
            for (int z = 0; z < mix_ks; z++) msq += xsq[(size_t)t * mix_ks + z];
            red[39] = rsqrtf(msq / (float)nh + rms_eps);
        }
        __syncthreads();
        float r = red[39];
        // mx reduce (24 partials, ORIGINAL order)
        for (int m = threadIdx.x; m < mix; m += blockDim.x) {
            float acc = 0.f;
            for (int z = 0; z < mix_ks; z++) acc += mx_in[((size_t)t * mix + m) * mix_ks + z];
            mx_s[m] = acc * r;
        }
        __syncthreads();
        const float* mx = mx_s;
        // pre_s (to GLOBAL scratch for blocks 1-3) + post + comb
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
        __threadfence();  // pre_g writes visible before flag
        __syncthreads();
        if (threadIdx.x == 0) atomicExch((int*)&flag[t], 1);
        // ═══ BLOCK 0: P3-seg-0 (li for hidden [0, h/HC_SPLIT)) — runs
        // CONCURRENTLY with blocks 1-3's segments on separate SMs ═══
        {
            const float* pre_t = pre_g + (size_t)t * n;
            const int hseg = (h + HC_SPLIT - 1) / HC_SPLIT;
            const int hi = min(h, hseg);
            float* li_g = li + (size_t)t * h;
            for (int j = threadIdx.x; j < hi; j += blockDim.x) {
                float acc = 0.f;
                for (int i = 0; i < n; i++) acc += pre_t[i] * x[(size_t)i * h + j];
                li_g[j] = acc;
            }
        }
        __threadfence();  // li writes visible before counter wait
        __syncthreads();
        // ═══ BLOCK 0: wait for blocks 1-3 (counter == HC_SPLIT-1) ═══
        if (threadIdx.x == 0) {
            while (atomicAdd((int*)&cntr[t], 0) < HC_SPLIT - 1) __nanosleep(100);
        }
        __syncthreads();
        // ═══ BLOCK 0: P4 (rmsnorm — ORIGINAL order) + P5 (writeback) ═══
        {
            float* li_g = li + (size_t)t * h;
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
            for (int j = threadIdx.x; j < h; j += blockDim.x) {
                li_g[j] = li_g[j] * inv * nw[j];
            }
        }
        // reset for next invocation (mega graph replays 90×/step)
        if (threadIdx.x == 0) { atomicExch((int*)&flag[t], 0); atomicExch((int*)&cntr[t], 0); }
    } else {
        // ═══ BLOCKS 1..3: wait flag → P3-seg-1..3 → counter++ ═══
        if (threadIdx.x == 0) {
            while (atomicAdd((int*)&flag[t], 0) < 1) __nanosleep(100);
        }
        __syncthreads();
        const float* pre_t = pre_g + (size_t)t * n;
        const int hseg = (h + HC_SPLIT - 1) / HC_SPLIT;
        const int lo = sp * hseg;
        const int hi = min(h, lo + hseg);
        float* li_g = li + (size_t)t * h;
        for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
            float acc = 0.f;
            for (int i = 0; i < n; i++) acc += pre_t[i] * x[(size_t)i * h + j];
            li_g[j] = acc;
        }
        __threadfence();  // li writes visible before counter
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd((int*)&cntr[t], 1);
    }
}

'''
src = src[:start] + new_kernel + src[end:]

# Also fix the launcher: grid=(s, HC_SPLIT) instead of grid=(s, 1)
launch_old = """    hc_pre_rest_kernel<<<s, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);"""
if launch_old not in src:
    # Try with the mx_scratch param name from the P1 fusion version
    launch_old2 = """    hc_pre_rest_kernel<<<s, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    return cudaGetLastError();"""
    if launch_old2 in src:
        launch_old = launch_old2
    else:
        print("LAUNCHER NOT FOUND — searching...", file=sys.stderr)
        import re
        m = re.search(r'hc_pre_rest_kernel<<<[^>]+>>>\s*\(', src)
        if m:
            print(f"Found at: {m.group()}", file=sys.stderr)
        sys.exit(1)

launch_new = """    dim3 rest_grid(s, HC_SPLIT);
    hc_pre_rest_kernel<<<rest_grid, 1024, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    return cudaGetLastError();"""
src = src.replace(launch_old, launch_new, 1)

# Also update the mix_split kernel to zero flag/cntr (block (t,0,0) does it free)
mix_split_marker = "mx_partial[((size_t)s * mix * KS + (size_t)t * KS + z] = tot2;"
if mix_split_marker not in src:
    # Search for the Σx² fusion write in mix_split
    mix_split_marker2 = "mx_partial[(size_t)s * mix * KS + (size_t)t * KS + z] = tot2;"
    if mix_split_marker2 in src:
        zero_code = mix_split_marker2 + """
    // Zero the flag/cntr for the rest kernel's atomic sync (free — runs
    // before rest in the same stream; the (t,0,0) block does it)
    if (m == 0 && z == 0 && threadIdx.x == 0) {
        volatile int* flag = (volatile int*)(mx_partial + (size_t)s * mix * KS + s * KS);
        volatile int* cntr = (volatile int*)(mx_partial + (size_t)s * mix * KS + s * KS + s);
        *flag = 0; *cntr = 0;
    }"""
        src = src.replace(mix_split_marker2, zero_code, 1)
        print("mix_split flag/cntr zeroing added")
    else:
        print("WARNING: Σx² fusion marker not found in mix_split — flag/cntr not zeroed", file=sys.stderr)
else:
    print("Found first marker variant")

open(CU, "w").write(src)
print("4-block atomic-sync hc_pre_rest: kernel + launcher + mix_split zeroing DONE")
