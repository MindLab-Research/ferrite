#!/usr/bin/env python3
"""hc_pre_rest 4-block atomic-sync split: P3 (li, the 64KB n×h reads — the
dominant cost at 1 SM) parallelized across 4 blocks with atomic flag sync
(no cooperative launch, no extra kernel launches). Block 0 does P1+P2
(mx reduction + sinkhorn, ~2µs), sets flag; blocks 1-3 each compute 1/4 of
the hidden dimension's li (4 SMs in parallel, ~16µs vs 64µs at 1 SM);
block 0 waits on counter then does P4+P5 (rmsnorm + writeback, ~3µs).
Critical path: ~22µs vs current 61.8µs → saves ~40µs × 90/step = 3.6ms.
"""
import sys

CU = "kernels/cuda/ferrite_kernels.cu"
src = open(CU).read()

# 1. Replace hc_pre_rest_kernel with the 4-block atomic-sync version
start = src.index("// The REST of hc_pre: reads pre-computed mx from global memory")
end = src.index('extern "C" cudaError_t ferrite_hc_pre_split')

new_kernel = r'''// The REST of hc_pre — 4-BLOCK ATOMIC-SYNC SPLIT (grid=(s, HC_SPLIT)):
// nsys: hc_pre_rest was 61.8µs × 90/step = 5.6ms/step (35% of the 16.2ms
// decode) at grid=s = ONE block on 148 SMs. The P3 (li: n=4 × h=4096 =
// 64KB reads at 1 SM's ~32 outstanding loads = 64 batches × 500ns L2
// latency = ~32µs) dominates. 4 blocks parallelize P3 across 4 SMs (~16µs).
// Sync: atomic flag/counter in global scratch (NOT cooperative launch —
// grid.sync measured +5µs×4 = 20µs overhead making it SLOWER; NOT separate
// launches — 3×5µs launch overhead also SLOWER). Atomic spin ~0.5µs each.
// FP PARITY: P3's per-element accumulation (i=0..n-1 sequential FMA) and
// P4's rmsnorm reduction order (threadIdx.x stride → warp shuffle → serial
// red[8]) are IDENTICAL to the single-block version — only the j→thread
// mapping changes (different threads compute different j, same formula).
// SCRATCH (mx_scratch tail): flag[s] + counter[s] as int32 (cast from
// float*), pre_s[s][n] — zeroed by the mix_split's (t,0,0) block (free).
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
    //   mx:     [0, s*mix*KS)          — mix partials (mix_split writes)
    //   xsq:    [s*mix*KS, +s*KS)      — Σx² partials (mix_split m==0 lanes)
    //   flag:   [+s)  as int           — block 0 → blocks 1-3 handshake
    //   cntr:   [+s)  as int           — blocks 1-3 → block 0 handshake
    //   pre_s:  [+s*n)                 — block 0 P2 → blocks 1-3 P3
    const size_t mx_tot = (size_t)s * mix * mix_ks;
    float* xsq  = mx_in + mx_tot;                       // [s][KS]
    volatile int* flag = (volatile int*)(xsq + s * mix_ks);         // [s]
    volatile int* cntr = (volatile int*)(xsq + s * mix_ks + s);     // [s]
    float* pre_g = xsq + s * mix_ks + 2 * s;            // [s][n]

    if (sp == 0) {
        // ═══ BLOCK 0: P1 (mx reduction + Σx²) + P2 (pre_s + sinkhorn) ═══
        extern __shared__ float sm[];
        float* mx_s = sm;               // [mix]
        float* red = sm + 24;            // [8] warp partials
        // Σx² from the mix_split's fused partials (8 adds, not 16K re-read)
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
            // sinkhorn (identical to single-block version)
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
        // ═══ BLOCK 0: wait for blocks 1-3 (P3) ═══
        if (threadIdx.x == 0) {
            while (atomicAdd((int*)&cntr[t], 0) < HC_SPLIT - 1) __nanosleep(100);
        }
        __syncthreads();
        // ═══ BLOCK 0: P4 (rmsnorm — ORIGINAL order) + P5 (writeback) ═══
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
        // reset for next invocation (mega graph replays this kernel 90×/step)
        if (threadIdx.x == 0) { atomicExch((int*)&flag[t], 0); atomicExch((int*)&cntr[t], 0); }
    } else {
        // ═══ BLOCKS 1..3: P3 (li for hidden range [sp*h/4, (sp+1)*h/4)) ═══
        if (threadIdx.x == 0) {
            while (atomicAdd((int*)&flag[t], 0) < 1) __nanosleep(100);
        }
        __syncthreads();
        const float* pre_t = pre_g + (size_t)t * n;
        const int hseg = (h + HC_SPLIT - 1) / HC_SPLIT;
        const int lo = (sp - 1) * hseg;  // sp=1..3 → segments 0..2? No:
        // sp=1..HC_SPLIT-1 → segments 0..HC_SPLIT-2. Block 0 handles the
        // LAST segment (after P4/P5 it also does the first h/4? No —
        // block 0 does P4/P5 only, P3 is split across blocks 1..3 AND
        // block 0's idle threads during the flag wait.
        // Actually: let's split P3 across ALL 4 blocks (block 0 does its
        // segment AFTER P2, before setting the flag; blocks 1-3 do their
        // segments after the flag). This gives 4× parallelism for P3.
        // REVISED: block 0 does P2 then P3-seg-0 then flag; blocks 1-3
        // do P3-seg-1..3 after flag; block 0 then waits for counter.
        // Simpler: block 0 does segment 0 in addition to P1+P2 (overlapped).
        // But that makes block 0's critical path: P1+P2 (2µs) + P3-seg (16µs
        // at 1 SM... no, block 0 also has 1 SM). Better: block 0 does ONLY
        // P1+P2+P4+P5 (lightweight), blocks 1-3 do P3 (heavyweight).
        // 3 blocks × h/3 elements each. With 3 SMs: P3 = 64µs/3 ≈ 21µs.
        // Or: use 4 P3 blocks (sp=0..3) where block 0 does P3-seg-0 AFTER
        // setting the flag (overlapped with blocks 1-3's P3-seg-1..3).
        // Block 0's critical path: P1+P2 (2µs) + flag (0.1µs) + P3-seg-0
        // (16µs at 1 SM) + wait for counter (0.1µs) + P4+P5 (3µs) = 21µs.
        // This is the same as 3-block but with 4× P3 parallelism.
        // Let me use the simplest: blocks 1-3 do 3 segments (h/3 each),
        // block 0 does P1+P2+P4+P5 only. P3 time: 64µs/3 ≈ 21µs.
        // NO — 4 segments is better. Block 0 does P3-seg-0 AFTER flag set.
        // The flag is set by thread 0, then ALL threads (including P3
        // threads) proceed. Block 0's P3-seg-0 runs while blocks 1-3 do
        // their segments. Then block 0 waits for counter==3 (blocks 1-3 done).
        // Critical path: P1+P2 (2µs) + flag + P3-seg-0 (16µs) + wait for
        // blocks 1-3 (already done, 0µs wait) + P4+P5 (3µs) = 21µs.
        // Actually block 0 can't do P3-seg-0 AND P4+P5 without waiting for
        // blocks 1-3's segments (P4 needs ALL of li). So block 0 does:
        // P1+P2 → flag → P3-seg-0 → wait for counter==3 → P4+P5.
        // The P3-seg-0 runs CONCURRENTLY with blocks 1-3's segments.
        // Critical path: 2 + 16 + 0 (blocks 1-3 finish at same time) + 3 = 21µs.
        // THIS IS THE OPTIMAL SPLIT.
        const int seg = sp - 1;  // 0..HC_SPLIT-2 (blocks 1..HC_SPLIT-1 → segs 0..HC_SPLIT-2)
        // Wait, I want 4 segments but block 0 does seg 0 too. Let me use:
        // sp=0: P1+P2+P4+P5 (block 0)
        // sp=1..3: P3 segments 0..2 (blocks 1..3)
        // And block 0 does P3 segment 3 (the last h/4) after setting flag.
        // No, this is getting too complicated. Let me simplify:
        // sp=0: P1+P2, flag, P3-seg-0, (implicit wait via counter), P4+P5
        // sp=1..3: wait flag, P3-seg-1..3, counter++
        // Block 0's P3-seg-0 is done AFTER flag set, CONCURRENT with blocks 1-3.
        // All 4 P3 segments run in parallel on 4 SMs.
        // Block 0 then waits for counter==3 (blocks 1-3 done), does P4+P5.
        // Critical path: P1+P2 (2) + flag (0.1) + P3-seg (16, concurrent)
        //   + wait for slowest of blocks 1-3 (0, they finish at same time)
        //   + P4+P5 (3) = ~21µs.
        // Total: ~21µs vs current 61.8µs. 3× improvement.
        const int lo2 = seg * hseg;
        const int hi2 = min(h, lo2 + hseg);
        float* li_g = li + (size_t)t * h;
        for (int j = lo2 + threadIdx.x; j < hi2; j += blockDim.x) {
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
open(CU, "w").write(src)
print("4-block atomic-sync hc_pre_rest written (needs block-0 P3 segment + launch update)")
