#!/usr/bin/env python3
"""MoE route fusion: the router GEMV ([160, 4096] dot product) + the route
(sigmoid + topk + renorm) in ONE kernel — the "last block" pattern (the atomic
counter). Eliminates the separate moe_route kernel's 12µs launch overhead
× 44 layers = 0.53ms/step (+2 tok/s).

The router GEMV: the [160, 4096] × [4096] → [160] logits (the 160 blocks,
1 expert per block, the 4096 dot product each). The last block (the atomic
counter == 159): the route (sigmoid + topk + renorm on the 160 logits from
the global scratch). The route: the same code as the moe_route_kernel."""
import sys

CU = "kernels/cuda/ferrite_kernels.cu"
src = open(CU).read()

# Find the moe_route_kernel and add the fused version after it
marker = 'extern "C" cudaError_t ferrite_moe_route('
if marker not in src:
    print("MARKER NOT FOUND", file=sys.stderr)
    sys.exit(1)

fused = r'''// ============================================================
// router_gemm_route_fused: the router GEMV + the route in ONE kernel
// (the "last block" pattern). The separate moe_route was 14.1µs × 44 layers
// = 0.62ms/step — 12µs launch overhead (the 1-block kernel's graph node
// dispatch) + 2µs execution. The fusion eliminates the launch overhead:
// the router GEMV's 160 blocks compute the logits, the last block (the
// atomic counter) does the route (sigmoid + topk + renorm). The route's
// FP is IDENTICAL to the moe_route_kernel (the same sigmoid, the same
// topk selection order, the same renorm).
// ============================================================
__global__ void router_gemm_route_fused_kernel(
    const float* __restrict__ x,           // [hidden] the input
    const __nv_bfloat16* __restrict__ w,   // [n_exp, hidden] the router weight
    const float* __restrict__ bias,        // [n_exp] the router bias
    float* __restrict__ probs,             // [n, topk] the output probs
    float* __restrict__ ids,               // [n, topk] the output ids
    float* __restrict__ logits,            // [n_exp] the scratch (the logits)
    unsigned* __restrict__ ctr,            // [1] the atomic counter
    int n_exp, int hidden, int topk, float scale) {

    int e = blockIdx.x; // 1 expert per block (grid = n_exp)
    if (e >= n_exp) return;
    int lane = threadIdx.x & 31;

    // ── GEMV: logit[e] = w[e,:] · x (uint4-vectorized, the same pattern as
    // gemv_bf16_v2<WPR=1> but per-block: 1 expert per block, 256 threads) ──
    const __nv_bfloat16* wr = w + (size_t)e * hidden;
    float acc = 0.f;
    // uint4 vectorized (8 bf16 per load — hidden % 8 == 0 for GLM-5.3)
    for (int k = threadIdx.x * 8; k + 7 < hidden; k += blockDim.x * 8) {
        float4 xa = *reinterpret_cast<const float4*>(x + k);
        float4 xb = *reinterpret_cast<const float4*>(x + k + 4);
        uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
        float2 f0 = __bfloat1622float2(w2[0]), f1 = __bfloat1622float2(w2[1]);
        float2 f2 = __bfloat1622float2(w2[2]), f3 = __bfloat1622float2(w2[3]);
        acc += xa.x * f0.x + xa.y * f0.y + xa.z * f1.x + xa.w * f1.y
             + xb.x * f2.x + xb.y * f2.y + xb.z * f3.x + xb.w * f3.y;
    }
    for (int k = threadIdx.x + ((hidden >> 3) << 3); k < hidden; k += blockDim.x) {
        acc += x[k] * __bfloat162float(wr[k]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    __shared__ float ws[16];
    if (lane == 0) ws[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int i = 0; i < (blockDim.x + 31) >> 5; i++) tot += ws[i];
        logits[e] = tot;
    }
    __threadfence();
    __syncthreads();

    // ── The "last block" pattern: the atomic counter ──
    __shared__ int is_last;
    if (threadIdx.x == 0) {
        unsigned prev = atomicAdd(ctr, 1u);
        is_last = (prev == (unsigned)(n_exp - 1)) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) return;

    // ── ROUTE (the last block only): sigmoid + topk + renorm ──
    // The SAME code as moe_route_kernel (the identical sigmoid, the same
    // topk selection order, the same renorm — FP-safe).
    extern __shared__ float sm[]; // [n_exp] sigmoid + [n_exp] choice
    float* ch = sm + n_exp;
    for (int j = threadIdx.x; j < n_exp; j += blockDim.x)
        sm[j] = 1.0f / (1.0f + expf(-logits[j]));
    __syncthreads();
    for (int j = threadIdx.x; j < n_exp; j += blockDim.x)
        ch[j] = sm[j] + bias[j];
    __syncthreads();
    // selection topk (the same as moe_route_kernel)
    for (int r = 0; r < topk; r++) {
        __shared__ int bidx[32];
        __shared__ float bval[32];
        int best = -1;
        float bv = -1e30f;
        for (int j = threadIdx.x; j < n_exp; j += blockDim.x) {
            if (ch[j] > bv) { bv = ch[j]; best = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, bv, off);
            int oi = __shfl_down_sync(0xffffffff, best, off);
            if (ov > bv) { bv = ov; best = oi; }
        }
        int warp = threadIdx.x >> 5;
        if (lane == 0) { bidx[warp] = best; bval[warp] = bv; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int sel = -1;
            float sv = -1e30f;
            for (int wv = 0; wv < (blockDim.x >> 5); wv++) {
                if (bval[wv] > sv) { sv = bval[wv]; sel = bidx[wv]; }
            }
            if (sel >= 0) {
                ids[r] = (float)sel;
                ch[sel] = -1e30f;
            } else {
                ids[r] = -1.0f;
            }
        }
        __syncthreads();
    }
    // renorm (the same as moe_route_kernel)
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int r = 0; r < topk; r++) {
            int j = (int)ids[r];
            float val = sm[j];
            probs[r] = val;
            sum += val;
        }
        for (int r = 0; r < topk; r++)
            probs[r] = probs[r] / (sum + 1e-9f) * scale;
        *ctr = 0u; // reset for the next invocation (the mega graph replays)
    }
}

extern "C" cudaError_t ferrite_router_gemm_route_fused(
    const float* x, const void* w, const float* bias,
    float* probs, float* ids, float* logits, unsigned* ctr,
    int n_exp, int hidden, int topk, float scale, cudaStream_t s) {
    dim3 block(256);
    dim3 grid(n_exp);
    size_t smem = 2 * (size_t)n_exp * sizeof(float);
    router_gemm_route_fused_kernel<<<grid, block, smem, s>>>(
        x, (const __nv_bfloat16*)w, bias, probs, ids, logits, ctr,
        n_exp, hidden, topk, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_moe_route('''

src = src.replace(marker, fused, 1)
open(CU, "w").write(src)
print("router_gemm_route_fused kernel added (the last block pattern: GEMV + sigmoid + topk + renorm)")
