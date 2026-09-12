// DeepSeek-V4.1-Flash glue kernels: engram gated write-back, SwiGLU with the
// training clamps, row gather / scatter-add, and the hc pre-mix collapse.
//
// No quantised weights and no tensor-core work here: these are plain f32
// elementwise/streaming ops that sit between the big kernels of the chain.
// Every semantic is pinned by the ABI in crates/ferrite-dsv41/src/kernels.rs
// and mirrored by the CPU golden in crates/ferrite-dsv41/src/ops.rs.
//
// Determinism (each kernel also documents its own case):
//   * engram_apply / swiglu_limit / gather_rows / hc_collapse: every output
//     element is produced by exactly one thread and never revisited, so the
//     results are bitwise stable run-to-run. engram_apply's only cross-thread
//     reduction is a fixed-shape tree; hc_collapse keeps the golden's
//     ascending-i FMA chain.
//   * scatter_add_rows: atomic adds -- rows that receive two or more
//     contributions accumulate in scheduler-defined order and are therefore
//     NOT bitwise reproducible (see the kernel comment for the analysis).
//
// Build (same TU style as the rest of the DSv4.1 kernels):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -c dsv41_glue.cu

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

// ===========================================================================
// engram gated write-back  (ops.rs::engram_forward)
// ===========================================================================

// Block-wide all-reduce of three f32 lanes over a 128-thread (4-warp) block.
// Fixed reduction shape -> deterministic given (dim, blockDim).
__device__ __forceinline__ void block_sum3(float& a, float& b, float& c) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xFFFFFFFFu, a, off);
        b += __shfl_xor_sync(0xFFFFFFFFu, b, off);
        c += __shfl_xor_sync(0xFFFFFFFFu, c, off);
    }
    __shared__ float red[3][8];  // up to 8 warps
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        red[0][wid] = a;
        red[1][wid] = b;
        red[2][wid] = c;
    }
    __syncthreads();
    if (wid == 0) {
        const int nw = blockDim.x >> 5;  // <= 8
        a = (lane < nw) ? red[0][lane] : 0.f;
        b = (lane < nw) ? red[1][lane] : 0.f;
        c = (lane < nw) ? red[2][lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_xor_sync(0xFFFFFFFFu, a, off);
            b += __shfl_xor_sync(0xFFFFFFFFu, b, off);
            c += __shfl_xor_sync(0xFFFFFFFFu, c, off);
        }
        if (lane == 0) {
            red[0][0] = a;
            red[1][0] = b;
            red[2][0] = c;
        }
    }
    __syncthreads();
    a = red[0][0];
    b = red[1][0];
    c = red[2][0];
}

// One block per (row, i < hc) slot of x -- the slot is exclusively owned by
// this block, so the in-place x += gate * value update is race-free. `kv` rows
// are [hc*dim key | dim value]; q_weight/k_weight are [hc, dim].
//
// gate = sigmoid( signed_sqrt(max(|dot|, 1e-6)) ), with
//   dot  = <h, q_w[i] * k_w[i] * k> * rstd * dim^-0.5,
//   rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(k^2) + eps),
// exactly the reference's normalised dot (ops.rs::engram_forward).
// token_mask may be null; mask[row] == 0 forces gate = 0.
__global__ void engram_apply_kernel(float* __restrict__ x, const float* __restrict__ kv,
                                    const float* __restrict__ q_weight,
                                    const float* __restrict__ k_weight,
                                    const uint8_t* __restrict__ token_mask, int rows, int hc,
                                    int dim, float eps) {
    const int r = blockIdx.x;
    const int i = blockIdx.y;
    if (r >= rows || i >= hc) return;
    const size_t span = (size_t)hc * dim + dim;
    float* h = x + ((size_t)r * hc + i) * dim;                   // read + written back
    const float* k = kv + (size_t)r * span + (size_t)i * dim;
    const float* value = kv + (size_t)r * span + (size_t)hc * dim;
    const float* qw = q_weight + (size_t)i * dim;
    const float* kw = k_weight + (size_t)i * dim;

    // Only hc blocks are launched (4 in the production config) and each block's
    // whole job is a dim-long strided scan plus a fixed-shape tree reduction, so
    // the kernel is latency-bound, not bandwidth-bound: it moves ~90 KB per
    // block (dim=5120 x 5 arrays x 4 B) against a ~0.1 us DRAM floor yet
    // measured 32.6 us/call. The float4 body cuts the per-thread round trips 4x
    // (dim/4 = 1280 vectors over 256 threads = 5 rounds instead of 40 scalar
    // ones) and the launcher uses 256 threads (8 warps, the block_sum3 limit).
    // The reduction order changes, so this is NOT bit-identical to the scalar
    // body - the same trade the engram f32 direct-read path already made; the
    // numerical self-test (tests_dsv41_glue.cu, 1e-6) is the gate.
    // vec4 needs 16B everywhere: dim % 4 == 0 makes every row offset a multiple
    // of 16 B, and the four bases are cudaMalloc'd tensors (the launcher never
    // hands in an interior pointer).
    const bool vec4 = ((dim & 3) == 0) &&
                      ((((uintptr_t)h | (uintptr_t)k | (uintptr_t)qw | (uintptr_t)kw) & 15u) == 0);

    float hss = 0.f, kss = 0.f, dot = 0.f;
    const int n4 = dim >> 2;
    if (vec4) {
        const float4* h4 = reinterpret_cast<const float4*>(h);
        const float4* k4 = reinterpret_cast<const float4*>(k);
        const float4* q4 = reinterpret_cast<const float4*>(qw);
        const float4* w4 = reinterpret_cast<const float4*>(kw);
        for (int c = threadIdx.x; c < n4; c += blockDim.x) {
            const float4 hv = h4[c], kv = k4[c], qv = q4[c], wv = w4[c];
            hss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
            kss += kv.x * kv.x + kv.y * kv.y + kv.z * kv.z + kv.w * kv.w;
            dot += hv.x * qv.x * wv.x * kv.x + hv.y * qv.y * wv.y * kv.y +
                   hv.z * qv.z * wv.z * kv.z + hv.w * qv.w * wv.w * kv.w;
        }
    } else {
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            const float hv = h[c];
            const float kval = k[c];
            hss += hv * hv;
            kss += kval * kval;
            dot += hv * qw[c] * kw[c] * kval;
        }
    }
    block_sum3(hss, kss, dot);

    const float rstd = (1.f / sqrtf(hss / (float)dim + eps)) *
                       (1.f / sqrtf(kss / (float)dim + eps));
    dot *= rstd * (1.f / sqrtf((float)dim));
    // signed sqrt with the >=1e-6 magnitude clamp (copysign matches Rust's
    // signum, including the -0.0 case)
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f)) * copysignf(1.f, dot);
    float gate = 1.f / (1.f + expf(-mag));
    if (token_mask != nullptr && token_mask[r] == 0u) gate = 0.f;

    if (vec4) {
        float4* h4 = reinterpret_cast<float4*>(h);
        const float4* v4 = reinterpret_cast<const float4*>(value);
        for (int c = threadIdx.x; c < n4; c += blockDim.x) {
            const float4 hv = h4[c], vv = v4[c];
            h4[c] = make_float4(hv.x + gate * vv.x, hv.y + gate * vv.y, hv.z + gate * vv.z,
                                hv.w + gate * vv.w);
        }
    } else {
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            h[c] = h[c] + gate * value[c];
        }
    }
}

// ===========================================================================
// SwiGLU with the training clamps  (fused gate_up [rows, 2*inter])
// ===========================================================================

// For i < inter, reading g = gate_up[r, i] and u = gate_up[r, inter + i]:
//   g = min(g, limit), u = clamp(u, -limit, limit)   (limit > 0 only)
//   gate_up[r, i] = silu(g) * u
// The up half is never written (no out-of-bounds store either way). limit <= 0
// disables the clamps. This is the epilogue of dsv41_expert_gate_up_fp4's
// output layout (gate first, up second), silu = g / (1 + exp(-g)).
__global__ void swiglu_limit_kernel(float* __restrict__ gate_up, int rows, int inter,
                                    float limit) {
    const size_t total = (size_t)rows * inter;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(t / (size_t)inter);
        const int i = (int)(t % (size_t)inter);
        float* row = gate_up + (size_t)r * 2 * inter;
        float g = row[i];
        float u = row[inter + i];
        if (limit > 0.f) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        row[i] = (g / (1.f + expf(-g))) * u;
    }
}

// A4: swiglu_limit_kernel + the T1 fp8 epilogue. The f32 this kernel writes back
// is exactly what the caller's next quant1(ex_act) would quantise, so emit the
// e4m3 byte and its per-32-block scale here and the consumer drops that launch
// (40 launches/step for the shared expert). This is hc_front's collapse technique
// (dsv41_kernels.cu, the xq != nullptr epilogue) applied to the swiglu write-back.
//
// One WARP per 32-element scale block: with inter % 32 == 0 and a 32-multiple
// blockDim, a warp's lanes stay inside ONE block, so the amax is a single shuffle
// tree - no barrier, no second pass over global memory. The scale and the byte are
// quant_kernel's arithmetic term for term (fast_round_scale(amax, 1/448), clamp
// +-448, __nv_fp8_e4m3), and `v` is the SAME value the f32 store writes (one
// register path, not a global re-read), so the emitted (xq, xsc) pair is
// bit-identical to the quant1 launch it replaces.
//
// This file's own copy of the scale helper: dsv41_glue.cu is a separate
// translation unit from dsv41_kernels.cu (build.sh compiles each .cu on its own
// and links the objects), so the __device__ helper there is not visible here.
// Same arithmetic as dsv41_kernels.cu's fast_round_scale, term for term.
__device__ __forceinline__ float glue_fast_round_scale(float amax, float max_inv) {
    const uint32_t bits = __float_as_uint(amax * max_inv);
    const int exp = (int)((bits >> 23) & 0xFFu);
    const uint32_t man = bits & 0x7FFFFFu;
    const int e = exp - 127 + (man != 0 ? 1 : 0);
    return __int_as_float((e + 127) << 23);
}

__global__ void swiglu_limit_q_kernel(float* __restrict__ gate_up, int rows, int inter,
                                      float limit, uint8_t* __restrict__ xq,
                                      float* __restrict__ xsc) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wpb = (blockDim.x + 31) >> 5;
    const int nb = inter >> 5;
    const size_t total = (size_t)rows * (size_t)nb;
    for (size_t t = (size_t)blockIdx.x * wpb + warp; t < total; t += (size_t)gridDim.x * wpb) {
        const int r = (int)(t / (size_t)nb);
        const int b = (int)(t % (size_t)nb);
        float* row = gate_up + (size_t)r * 2 * inter;
        const int i = (b << 5) + lane;
        float g = row[i];
        float u = row[inter + i];
        if (limit > 0.f) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        const float v = (g / (1.f + expf(-g))) * u;   // identical to swiglu_limit_kernel
        row[i] = v;                                   // f32 write-back, unchanged
        float a = fabsf(v);
        for (int off = 16; off > 0; off >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
        const float sc = fmaxf(glue_fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
        if (lane == 0) xsc[(size_t)r * nb + b] = sc;
        const float q = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
        const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
        xq[(size_t)r * inter + i] = *(const uint8_t*)&f8;
    }
}

// ===========================================================================
// Row gather / scatter-add
// ===========================================================================

// out[i, :] = src[idx[i], :]; idx may repeat (pure copy, bitwise exact).
__global__ void gather_rows_kernel(const float* __restrict__ src, const int32_t* __restrict__ idx,
                                   float* __restrict__ out, int n, int dim) {
    const size_t total = (size_t)n * dim;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(t / (size_t)dim);
        const int c = (int)(t % (size_t)dim);
        out[t] = src[(size_t)idx[i] * dim + c];
    }
}

// dst[idx[i], :] += src[i, :] * weight[i] -- one atomicAdd per element.
//
// Why atomics (documented choice): this ABI carries no dst-row count, so the
// destination extent is defined by the idx values alone; a single-pass atomic
// scatter is the natural O(n * dim) implementation and it cannot lose updates
// -- every duplicate contribution reaches the row (the read-modify-write
// serialises at the L2 slice).
//
// Determinism: NON-deterministic summation order. When one dst row receives
// two or more contributions, f32 addition is not associative and the atomic
// order is scheduler-defined, so those rows can differ run-to-run in the last
// ulp(s) (and from the sequential CPU golden). Single-contribution rows are
// exact, and no update is ever dropped. Impact on reproducibility: invisible
// to tolerance-based comparisons (<=1e-5 style), NOT safe for bitwise
// replay-diffing of rows that receive duplicates. The deterministic
// alternative -- one warp/block per dst row, accumulating in ascending i
// order (== the golden's order) -- costs O(dst_rows * n) index reads and would
// need a max-idx pre-pass the ABI does not provide; it stays the documented
// fallback if bitwise reproducibility ever becomes a requirement.
__global__ void scatter_add_rows_kernel(const float* __restrict__ src,
                                        const int32_t* __restrict__ idx,
                                        const float* __restrict__ weight,
                                        float* __restrict__ dst, int n, int dim) {
    const size_t total = (size_t)n * dim;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(t / (size_t)dim);
        const int c = (int)(t % (size_t)dim);
        atomicAdd(&dst[(size_t)idx[i] * dim + c], src[t] * weight[i]);
    }
}

// ===========================================================================
// hc pre-mix collapse  (ops.rs::hc_pre)
// ===========================================================================

// out[r, c] = sum_i pre[r*hc + i] * x[(r*hc + i)*dim + c], accumulated as an
// FMA chain over ascending i -- the golden's accumulation order. (GLM's
// ferrite_hc_contract is the UNWEIGHTED sum, a different op.)
__global__ void hc_collapse_kernel(const float* __restrict__ x, const float* __restrict__ pre,
                                   float* __restrict__ out, int rows, int hc, int dim) {
    const size_t total = (size_t)rows * dim;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(t / (size_t)dim);
        const int c = (int)(t % (size_t)dim);
        const float* pre_r = pre + (size_t)r * hc;
        const float* x_r = x + (size_t)r * hc * dim + c;
        float acc = 0.f;
        for (int i = 0; i < hc; i++) acc = fmaf(pre_r[i], x_r[(size_t)i * dim], acc);
        out[t] = acc;
    }
}

// ===========================================================================
// Device-side all-reduce completion (takes the HOST out of the critical path)
// ===========================================================================
//
// The old protocol was: issue the peer copies (async), then cudaDeviceSynchronize
// so this rank's copies are certainly complete, then a host barrier. That sync
// ran on every collective call — ~90 per decode step — and while it runs the
// host cannot enqueue the next layer's work, so CPU and GPU never overlap. These
// two kernels move the waiting onto the device instead: the stamp kernel runs
// after the peer copies on the same stream (so it inherits their completion) and
// writes this rank's round into every rank's stamp array; the reduce kernel
// spins until every peer has stamped that round, then sums. The host only
// enqueues.

__global__ void ar_stamp_kernel(const unsigned long long* __restrict__ peer_stamps, int world,
                                int rank, unsigned round) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();
        for (int p = 0; p < world; ++p) {
            unsigned* dst = (unsigned*)(peer_stamps[p] + (size_t)rank * sizeof(unsigned));
            *dst = round;
        }
        __threadfence_system();
    }
}

// AR v5 lives in the SHARED kernel set now: DSV41 calls ferrite_p2p_ar_v5
// (ferrite_kernels.cu) with its own staging tables. The three DSV41-specific
// kernels (ar_v5_store/publish/reduce) and their 
static inline int dsv41_smem_ceiling(const void* /*kern*/) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int dev = 0; cudaGetDevice(&dev);
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cached = optin - 1024;
    return cached;
}

// Legacy extern "C" launchers were removed — one protocol, one implementation.

// ===========================================================================
// Lean M=1 GEMV (single-token projections)
// ===========================================================================
//
// cuBLAS's GemmEx with N=1 picked `gemv2T_kernel` at ~40 GFLOP/s: 396us per call,
// 72 calls per decode step — while the weight read alone (e.g. wq_b, 335MB bf16)
// floors at ~112us. These keep a plain f32 accumulation with the same precision
// but read the weights once, coalesced along K, one warp per output row.
__global__ void gemv_bf16_kernel(const __nv_bfloat16* __restrict__ w, const float* __restrict__ x,
                                 float* __restrict__ out, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    // REVERTED (2026-09-11): staging the activation row into shared memory measured
    // a 2.3x STEP-TIME regression once the dynamic-smem argument was actually
    // allocated (the staging had silently run past a zero-sized allocation before,
    // which is why the earlier arm looked neutral). Reading `x` straight from
    // global is the correct baseline here: every block touches the same k floats,
    // so they stay L2-resident, and the barrier-free row loop keeps the launch
    // footprint at zero shared memory.
    for (int row = blockIdx.x * nwarp + wid; row < n; row += gridDim.x * nwarp) {
        const __nv_bfloat16* wr = w + (size_t)row * (size_t)k;
        float acc = 0.f;
        // NO #pragma unroll here: measured 13.39ms with it vs 13.30ms without
        // (same session, same binary path). The fp8 gemv gained from an unroll
        // pragma but this one loses - the two kernels have different
        // register/occupancy profiles.
        for (int c = lane; c < k; c += 32) acc += __bfloat162float(wr[c]) * x[c];
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) out[row] = acc;
    }
}

__global__ void gemv_f32_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                float* __restrict__ out, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    for (int row = blockIdx.x * nwarp + wid; row < n; row += gridDim.x * nwarp) {
        const float* wr = w + (size_t)row * (size_t)k;
        float acc = 0.f;
        // NO #pragma unroll (see gemv_bf16's note: lost 0.09ms with it).
        for (int c = lane; c < k; c += 32) acc += wr[c] * x[c];
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) out[row] = acc;
    }
}

// ===========================================================================
// Multi-row head GEMV (`dsv41_head_gemv_bf16_mrows` below): m activation rows
// against ONE pass over the weight.
// ===========================================================================
// The DSpark verify block (`step_rows`, chain_dev.rs) called `dsv41_gemv_bf16`
// once per row. Its head is the step's single largest weight — `head.weight` is
// Shard::Replicated [129280, 5120] bf16 = 1262 MB on every rank (weights.rs:90),
// which is 298us of streamed bytes — so six rows paid it six times: 1.79ms, the
// verify's largest single term after the projection family (docs/agent/
// dspark-verify-perf-plan.md §1.2/§2.4). Here one block decodes its row tile
// ONCE and folds every row against it.
//
// NUMERICS — for ANY m, row r of the m-row launch is BIT-IDENTICAL to the
// PRODUCTION single-row head GEMV of row r's activation. That reference is
// `gemv_bf16_nt_kernel`'s (ferrite_kernels.cu:3357) per-token body at WPR == 1
// — the same body `gemv_bf16_v2_kernel` runs, i.e. the program the head
// (out_f = 129280) actually gets from `gv2_wpr`/the nt launcher, which return
// WPR = 1 for out_f >= 16384. Per (output row, token) it computes
//     y(row,t) = T( SUM_{c = lane*8, lane*8 + 32*8, ... < k} (float)w[row][c..c+7]
//                      * x[t][c..c+7] )
// i.e. a lane-local ascending chain of uint4-granular 8-element groups, each
// group folded as two 4-term FMA groups (`xa.x*f0.x + xa.y*f0.y + xa.z*f1.x +
// xa.w*f1.y`, then the xb/f2/f3 half), through T = the fixed
// `__shfl_down_sync(0xffffffff, acc, off)` tree (off = 16,8,4,2,1), ending in
// `(bias ? bias[row] : 0.f) + acc` (bias is null for the head). This kernel
// computes exactly that, per row:
//     y(row,r) = the same expression, over the same lane->c mapping (C1), in the
//                same position (C2), through the same tree (C3), with NO
//                cross-row recombination — acc[r] is an independent chain and
//                nothing is ever added across r (C4) — and no K-split or smem
//                fold to reorder (C5, see the DOMAIN note: WPR == 1 is what
//                makes C5 available).
// The body below is TRANSCRIBED from that kernel, not re-derived from it: same
// source text in the same position under the same flags is the strongest
// guarantee that the two compile to the same FMA chain, and a re-derived
// "equivalent" expression is exactly how the earlier v1-referenced version went
// wrong (its scalar `__fmaf_rn(wv, x, acc)` chain is v1's program, not the
// head's).
// What the m-fold is allowed to change is only the WEIGHT RE-READ: `wv` and its
// four decodes are hoisted out of the r loop because they are per-c values —
// the identical decode of the identical bytes, reused M times, so C2 holds and
// the head is streamed once instead of m times. The row->warp mapping and the
// grid are not part of the contract either: rows are independent (C4), so which
// warp computes a row cannot change its value.
//
// ⚠️ DOMAIN OF THE PARITY (read before reusing this kernel):
//   * `out_f >= 16384` (hence WPR == 1). That is `gv2_wpr`'s first branch and
//     the nt launcher's, and it is what the head (129280) gets. BELOW it the
//     production GEMV K-SPLITS the row across WPR warps and folds the partials
//     in shared memory (`sum += part[t][base + j]`, j ascending): a different
//     program in a different order, so a small-n reuse has to re-derive it
//     rather than inherit it. The verify head is the only caller and it is deep
//     inside the WPR == 1 domain.
//   * `k % 8 == 0` (guaranteed by the launcher's decline below): the uint4/
//     float4 body needs it for the 16B alignment of `wr + c` AND for the vector
//     loop's coverage of [0, k).
//   * build flags: the SAME set build.sh uses for both TUs (`-O3
//     --use_fast_math`). The 8-element groups above are plain operators, and
//     the contraction/reassociation the compiler applies to them has to be the
//     same on both sides — which is why they are transcribed verbatim and why
//     the reduction tree keeps its source form too.
//
// `x` is [m, k] f32 (the verify's `xn_r`), `out` is [m, n] f32 (`logits_r`).
// M is a template parameter so each row owns a register chain; the launcher
// dispatches m = 1..8 and the caller keeps its per-row loop above 8. Grid and
// block mirror dsv41_gemv_bf16's (ceil(n/8) blocks capped at 4096, 8 warps),
// so the two launches differ in the row fold and nothing else.
// ===========================================================================
template <int M>
__global__ void head_gemv_bf16_mrows_kernel(const __nv_bfloat16* __restrict__ w,
                                            const float* __restrict__ x,
                                            float* __restrict__ out, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    for (int row = blockIdx.x * nwarp + wid; row < n; row += gridDim.x * nwarp) {
        const __nv_bfloat16* wr = w + (size_t)row * (size_t)k;
        float acc[M];
        #pragma unroll
        for (int r = 0; r < M; ++r) acc[r] = 0.f;
        // THE K ORDER IS gemv_bf16_nt_kernel's WPR == 1 BODY, TRANSCRIBED (see
        // the header): same lane -> k mapping (k0 == 0, step 32*8), same uint4
        // weight load, same four `__bfloat1622float2` decodes, same two 4-term
        // FMA groups in the same expression position and order. `wv`/f0..f3 sit
        // OUTSIDE the r loop: one decode of the row's 8 weights per k-step,
        // reused by every row — the m-fold's whole point, and identical in value
        // to the per-row decode it replaces.
        //
        // The slice arithmetic is v2/nt's with WPR == 1 folded away:
        //   kper = ((k + 1 - 1) / 1 + 7) & ~7 == k   (the launcher guarantees
        //                                             k % 8 == 0)
        //   k0 = kw * kper == 0,  k1 = min(k0 + kper, k) == k
        // so the loop below is their `for (k = k0 + lane*8; k + 7 < k1; k += 32*8)`
        // with lane*8 and 32*8 kept as literals. Nothing is added across r: each
        // acc[r] is its own ascending chain (C4), which is also why the row's
        // value cannot depend on the row->warp mapping.
        #pragma unroll 2
        for (int c = lane * 8; c + 7 < k; c += 32 * 8) {
            uint4 wv = *reinterpret_cast<const uint4*>(wr + c);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            float2 f0 = __bfloat1622float2(w2[0]);
            float2 f1 = __bfloat1622float2(w2[1]);
            float2 f2 = __bfloat1622float2(w2[2]);
            float2 f3 = __bfloat1622float2(w2[3]);
            #pragma unroll
            for (int r = 0; r < M; ++r) {
                const float* xr = x + (size_t)r * (size_t)k;
                float4 xa = *reinterpret_cast<const float4*>(xr + c);
                float4 xb = *reinterpret_cast<const float4*>(xr + c + 4);
                acc[r] += xa.x * f0.x + xa.y * f0.y + xa.z * f1.x + xa.w * f1.y;
                acc[r] += xb.x * f2.x + xb.y * f2.y + xb.z * f3.x + xb.w * f3.y;
            }
        }
        // The reduction tree keeps v2/nt's source form verbatim (shfl_down, off
        // = 16,8,4,2,1): the lane-0 partial is the same binary tree the
        // single-row launch produces. The epilogue is their
        // `(bias ? bias[row] : 0.f) + acc` with the null bias the head passes —
        // `__fadd_rn(0.f, a)`, not a bare `a`, so a -0.0 accumulator cannot
        // cross the two paths as a sign flip.
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            float a = acc[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a += __shfl_down_sync(0xffffffff, a, off);
            }
            if (lane == 0) out[(size_t)r * (size_t)n + row] = __fadd_rn(0.f, a);
        }
    }
}

// Publish this rank's payload into EVERY rank's staging slot (including our own)
// directly from the device. The old path issued `world` host-side peer copies per
// collective — ~8 API calls per all-reduce, ~16 per layer — with the host in the
// dependency chain. One kernel replaces them and stays on the GPU.
__global__ void ar_store_kernel(const unsigned long long* __restrict__ peer_slots, int world,
                                int rank, const float* __restrict__ src, long n, long slot_f,
                                long parity_off, const unsigned* __restrict__ reduced,
                                unsigned round, const unsigned long long* __restrict__ peer_stamps,
                                unsigned* __restrict__ ctr) {
    // Credit wait, on the DEVICE: the staging is double buffered by round parity,
    // so this write lands in the half last used by round-2 and must not start
    // until every peer has finished REDUCING round-2. EVERY block spins, not just
    // block 0 — the others would otherwise overwrite the slot mid-read.
    if (round >= 3 && threadIdx.x == 0) {
        for (int p = 0; p < world; ++p) {
            const volatile unsigned* m = reduced + p;
            while (*m < round - 2) {
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = src[i];
        for (int p = 0; p < world; ++p) {
            float* dst = (float*)(peer_slots[p]) + parity_off + (long)rank * slot_f + i;
            dst[0] = v;
        }
    }
    // Publish THIS round's stamp from inside the writing kernel. A valid release
    // needs the fence and the signal in the same threads: an earlier version
    // stamped from a separate kernel, whose __threadfence_system() only ordered
    // that kernel's own writes, so a peer could see the stamp before the data —
    // exactly the gap the host barrier had been masking.
    if (ctr == nullptr) return;  // default path: the separate stamp kernel does this
    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        const unsigned prev = atomicAdd(ctr, 1u);
        is_last = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        __threadfence_system();
        for (int p = 0; p < world; ++p) {
            unsigned* d = (unsigned*)(peer_stamps[p] + (size_t)rank * sizeof(unsigned));
            *d = round;
        }
        __threadfence_system();
        *ctr = 0;  // ready for the next round (stream order makes this safe)
    }
}

// Announced AFTER the reduce completes, so the peers' next store knows this rank
// is done reading that parity half.
__global__ void ar_mark_kernel(const unsigned long long* __restrict__ peer_reduced, int world,
                               int rank, unsigned round) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();
        for (int p = 0; p < world; ++p) {
            unsigned* dst = (unsigned*)(peer_reduced[p] + (size_t)rank * sizeof(unsigned));
            *dst = round;
        }
        __threadfence_system();
    }
}

__global__ void ar_reduce_kernel(float* __restrict__ dst, const float* __restrict__ staging,
                                 long n, long slot_f, int world,
                                 const unsigned* __restrict__ stamps, unsigned round,
                                 const unsigned long long* __restrict__ peer_reduced, int rank,
                                 unsigned* __restrict__ ctr2, int do_mark) {
    // every block waits: the stamps are in this rank's own memory (the peers
    // wrote them through their peer access) and stay in L2, so the spin is cheap
    if (threadIdx.x == 0) {
        for (int p = 0; p < world; ++p) {
            const volatile unsigned* s = stamps + p;
            while (*s < round) {
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < world; ++p) {
            acc += staging[(size_t)p * slot_f + i];
        }
        dst[i] = acc;
    }
    if (!do_mark || ctr2 == nullptr) return;
    // Same same-threads rule as the store: announce "round reduced" from inside
    // the kernel that did the reading, once every block is done.
    __threadfence();
    __shared__ bool is_last2;
    if (threadIdx.x == 0) {
        const unsigned prev = atomicAdd(ctr2, 1u);
        is_last2 = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (is_last2 && threadIdx.x == 0) {
        __threadfence_system();
        for (int p = 0; p < world; ++p) {
            unsigned* d = (unsigned*)(peer_reduced[p] + (size_t)rank * sizeof(unsigned));
            *d = round;
        }
        __threadfence_system();
        *ctr2 = 0;
    }
}

}  // namespace

// The decode-step n-gram hash on the device: removes the LAST per-step H2D on
// the decode path (the host used to run NgramHashState::forward_row and upload
// the ids) and makes the whole step graph-capturable. Faithful port of
// forward_row at seqlen=1 (the only shape step_impl calls): update the
// compressed-token cache at the DEVICE-side position counter, look back
// max_ngram tokens with the blocked rule (a position below zero or a DEAD entry
// pads every further lookback), then the rolling XOR hash per (layer, ngram,
// head): out = rolling.rem_euclid(lm) + off.
//
// Was ONE thread walking all n_layers*n_cols = 48 columns serially, each a
// 64-bit modulo (a software routine on sm_103a) plus two table reads: 17.6 us
// of pure serial latency. Every column is now an independent lane (the hash is
// integer and each eng_ids element is written by exactly one thread, so the
// result is bit-identical); the token gather is recomputed per lane, which is
// <= max_ngram (<= 8) L1/L2 cache reads.
__global__ void engram_hash_step_kernel(const long long* __restrict__ map, long long* __restrict__ cache,
                                   const long long* __restrict__ mults,
                                   const unsigned long long* __restrict__ lms,
                                   const unsigned long long* __restrict__ offs,
                                   long long* __restrict__ eng_ids, const int* __restrict__ token,
                                   const int* __restrict__ pos_ctr, long long map_len, int n_layers,
                                   int max_ngram, int n_heads, long long pad_id) {
    const int p = *pos_ctr;
    // One lane refreshes cache[p]; the barrier publishes it to the columns
    // (only pos_ctr itself was advanced by the previous kernel).
    if (threadIdx.x == 0) {
        const long long t = (long long)token[0];
        cache[p] = ((unsigned long long)t < (unsigned long long)map_len) ? map[t] : 0;
    }
    __syncthreads();
    const int n_cols = (max_ngram - 1) * n_heads;
    const int total = n_layers * n_cols;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int li = idx / n_cols;
        const int col = idx - li * n_cols;
        const int shift = col / n_heads + 1;  // col = (shift-1)*n_heads + head
        // tokens[0..shift] with the same cumulative blocked rule the serial
        // walk used (blocked[sh] covers every lookback up to sh).
        long long tokens[8];
        bool blocked = false;
        for (int sh = 0; sh <= shift; ++sh) {
            const long long q = p - (long long)sh;
            const long long src = (q >= 0) ? cache[q] : 0;
            blocked = blocked || (q < 0) || (src == -1);  // -1 == DEAD
            tokens[sh] = blocked ? pad_id : src;
        }
        const long long* m = mults + (size_t)li * 4;
        long long rolling = tokens[0] * m[0];
        for (int i = 1; i <= shift; ++i) rolling ^= tokens[i] * m[i];
        const long long lm = (long long)lms[(size_t)li * n_cols + col];
        const long long off = (long long)offs[(size_t)li * n_cols + col];
        long long v = rolling % lm;
        if (v < 0) v += lm;  // rem_euclid
        eng_ids[(size_t)li * n_cols + col] = v + off;
    }
    // NOTE: the counter is NOT advanced here any more - the argmax, the
    // LAST kernel of the step, advances it, so every kernel in between
    // (the window indices, the compressor, the rope) reads a stable
    // current position.
}

extern "C" int dsv41_engram_hash_step(const long long* map, long long* cache, const long long* mults,
                                 const unsigned long long* lms, const unsigned long long* offs,
                                 long long* eng_ids, const int* token, const int* pos_ctr,
                                 long long map_len, int n_layers, int max_ngram, int n_heads,
                                 long long pad_id, cudaStream_t s) {
    engram_hash_step_kernel<<<1, 128, 0, s>>>(map, cache, mults, lms, offs, eng_ids, token,
                                        pos_ctr, map_len, n_layers, max_ngram, n_heads, pad_id);
    return (int)cudaGetLastError();
}

// The decode-step window indices, on the device: replaces the host's
// ops::window_topk_idxs + a per-layer H2D upload. This is the decode branch
// (seqlen=1) of that function, VERBATIM - the trailing window in RING-SLOT
// order, where `oldest` is where the ring wraps - plus its start_pos == 0
// special case. Reads the position from the DEVICE counter (stable during the
// step: the argmax, the last kernel, is what advances it).
__global__ void window_idxs_kernel(int32_t* __restrict__ idxs, const int* __restrict__ pos_ctr,
                                   int window) {
    const int c = threadIdx.x + (int)blockIdx.x * blockDim.x;
    if (c >= window) return;
    const int start_pos = *pos_ctr;
    if (start_pos == 0) {
        idxs[c] = (c == 0) ? 0 : -1;
        return;
    }
    const int oldest = (start_pos % window) + 1;
    long long idx = ((long long)c < (long long)window - oldest)
                        ? (long long)oldest + c
                        : (long long)c - ((long long)window - oldest);
    if (idx > (long long)start_pos) idx = -1;
    idxs[c] = (int)idx;
}

// The recency placeholder for the compressed rows (the safety net when the
// group's owner has no indexer): idxs[win + j] = win + clen - take + j. Also
// removes a per-layer upload; clen is still a host value this round.
__global__ void comp_placeholder_kernel(int32_t* __restrict__ idxs,
                                        const int* __restrict__ clen, int window, int index_topk) {
    // `take` is derived on the DEVICE now (it changes per step and a captured graph
    // freezes launch arguments); the block count uses the cap and the guard is here.
    const int c = *clen;
    const int take = (c < index_topk) ? c : index_topk;
    const int j = threadIdx.x + (int)blockIdx.x * blockDim.x;
    if (j >= take) return;
    idxs[window + j] = window + c - take + j;
}

extern "C" int dsv41_window_idxs(int32_t* idxs, const int* pos_ctr, int window, cudaStream_t s) {
    if (window <= 0) return (int)cudaSuccess;
    window_idxs_kernel<<<(unsigned)((window + 127) / 128), 128, 0, s>>>(idxs, pos_ctr, window);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_comp_placeholder(int32_t* idxs, const int* clen, int window, int index_topk,
                                      cudaStream_t s) {
    comp_placeholder_kernel<<<(unsigned)((index_topk + 127) / 128), 128, 0, s>>>(idxs, clen, window,
                                                                                 index_topk);
    return (int)cudaGetLastError();
}

// The compressor commit, fused: reads out_rows ON THE DEVICE (the host used to
// download it, branch on it, run apply_rope, then cudaMemcpyD2D - a sync D2H per
// layer per step, and a host branch a CUDA graph cannot record), ropes the latent
// at the group's first token position, stores it into the ring at row
// window + *clen, and advances the DEVICE counter. No __syncthreads anywhere: the
// early-exit branch would make one UB (the failure mode of the older multi-token
// engram kernel), and each thread reads its own source pair and writes the
// destination, so no cross-thread ordering is needed at all.
// The rope math mirrors apply_rope_kernel exactly: the rotated region starts at
// hd - rope_dim, pairs (2i, 2i+1), the tables indexed as cos[t*half + i].
__global__ void compress_commit_kernel(const float* __restrict__ latent,
                                       const float* __restrict__ cos_t,
                                       const float* __restrict__ sin_t, float* __restrict__ ring,
                                       const int* __restrict__ out_rows, int* __restrict__ clen,
                                       int hd, int rope_dim, int half, int window, int ratio) {
    if (*out_rows <= 0) return;  // the device-side branch that replaces the download
    const int len = *clen;
    const int group_first = len * ratio;
    const int i0 = hd - rope_dim;
    const float* cs_row = cos_t + (size_t)group_first * half;
    const float* sn_row = sin_t + (size_t)group_first * half;
    float* dst = ring + (size_t)(window + len) * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float v = latent[c];
        if (c >= i0) {
            const int j = (c - i0) >> 1;
            const int base = i0 + (j << 1);
            const float x0 = latent[base];
            const float x1 = latent[base + 1];
            const float cv = cs_row[j];
            const float sv = sn_row[j];
            v = (c == base) ? (x0 * cv - x1 * sv) : (x0 * sv + x1 * cv);
        }
        dst[c] = v;
    }
    if (threadIdx.x == 0) *clen = len + 1;
}

extern "C" int dsv41_compress_commit(const float* latent, const float* cos_t, const float* sin_t,
                                     float* ring, const int* out_rows, int* clen, int hd,
                                     int rope_dim, int half, int window, int ratio,
                                     cudaStream_t s) {
    compress_commit_kernel<<<1, 128, 0, s>>>(latent, cos_t, sin_t, ring, out_rows, clen, hd,
                                             rope_dim, half, window, ratio);
    return (int)cudaGetLastError();
}

// Append this step's KV row into the window ring. The destination USED to be a
// host-computed address (slot = pos % window) baked into a captured cudaMemcpy
// node, so every replay of the graph wrote to the SAME slot - the classic
// "captured but not updated" bug, and exactly the cumulative degradation
// observed: the first tokens are right because they read the prefill's
// correctly written rows, and everything after the capture goes stale. The slot
// is derived from the DEVICE position counter inside the kernel now.
__global__ void ring_append_kernel(float* __restrict__ ring, const float* __restrict__ kv,
                                  const int* __restrict__ pos_ctr, int window, int hd) {
    const int slot = (*pos_ctr) % window;
    float* dst = ring + (size_t)slot * (size_t)hd;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < hd; i += gridDim.x * blockDim.x)
        dst[i] = kv[i];
}

// Publish this step's roped index key into the owner layer's group slot. The
// destination USED to be a host-computed address (index_k + (compress_len-1) *
// idx_hd), i.e. the same "host-computed address frozen by a graph capture" class
// as the window ring append: every replay would write the same group slot, so the
// indexer's compressed keys went stale as soon as the graph was used. The slot is
// derived from the DEVICE latent counter inside the kernel.
__global__ void index_k_publish_kernel(float* __restrict__ dst_base,
                                       const float* __restrict__ src,
                                       const int* __restrict__ clen, int idx_hd) {
    const int c = *clen;
    const int group = (c > 0) ? (c - 1) : 0;
    float* dst = dst_base + (size_t)group * (size_t)idx_hd;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < idx_hd; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

extern "C" int dsv41_index_k_publish(float* dst_base, const float* src, const int* clen,
                                     int idx_hd, cudaStream_t s) {
    if (idx_hd <= 0) return (int)cudaSuccess;
    index_k_publish_kernel<<<(unsigned)((idx_hd + 127) / 128), 128, 0, s>>>(dst_base, src, clen,
                                                                           idx_hd);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_ring_append(float* ring, const float* kv, const int* pos_ctr, int window,
                                 int hd, cudaStream_t s) {
    if (hd <= 0 || window <= 0) return (int)cudaSuccess;
    ring_append_kernel<<<(unsigned)((hd + 127) / 128), 128, 0, s>>>(ring, kv, pos_ctr, window, hd);
    return (int)cudaGetLastError();
}

// B2: ONE launch for the pair above and `window_idxs_kernel`. They are adjacent
// in the step, both one-block kernels gated on nothing but `*pos_ctr`, and
// neither consumes the other's output (the append writes the ring; the indices
// read only the counter), so nothing is reordered by fusing them. `ring == null`
// (a consumer layer that does not own its store) still performs the indices
// half, which lets the later standalone `window_idxs` launch disappear for
// EVERY layer, owner or not. Arithmetic is term for term the two kernels': the
// ring slot is (*pos_ctr % window), the indices reproduce the decode branch of
// ops::window_topk_idxs including the start_pos == 0 special case - so the
// fused launch is byte-identical to the pair it replaces.
__global__ void ring_win_fused_kernel(float* __restrict__ ring, const float* __restrict__ kv,
                                      const int* __restrict__ pos_ctr, int window, int hd,
                                      int32_t* __restrict__ idxs) {
    const int gid = threadIdx.x + (int)blockIdx.x * blockDim.x;
    const int start_pos = *pos_ctr;
    if (ring != nullptr && gid < hd) {
        const int slot = start_pos % window;
        ring[(size_t)slot * (size_t)hd + gid] = kv[gid];
    }
    if (gid >= window) return;
    const int c = gid;
    if (start_pos == 0) {
        idxs[c] = (c == 0) ? 0 : -1;
        return;
    }
    const int oldest = (start_pos % window) + 1;
    long long idx = ((long long)c < (long long)window - oldest)
                        ? (long long)oldest + c
                        : (long long)c - ((long long)window - oldest);
    if (idx > (long long)start_pos) idx = -1;
    idxs[c] = (int)idx;
}

extern "C" int dsv41_ring_win_fuse(float* ring, const float* kv, const int* pos_ctr, int window,
                                   int hd, int32_t* idxs, cudaStream_t s) {
    if (window <= 0) return (int)cudaSuccess;  // both halves are no-ops at window <= 0
    const int n = (window > hd) ? window : hd;
    ring_win_fused_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(ring, kv, pos_ctr, window, hd,
                                                                     idxs);
    return (int)cudaGetLastError();
}

// B3 (comp-placeholder fusion, `DSV41_COMP_PLACEHOLDER_FUSE`): the same launch
// ALSO writes `comp_placeholder_kernel`'s recency block
// (`idxs[window + j] = window + *clen - take + j`, `take = min(*clen,
// index_topk)`) into the SAME `idxs` buffer, one block past the window block
// `ring_win_fused_kernel` fills. That removes the standalone
// `dsv41_comp_placeholder` launch (30/step) and its graph node.
//
// Bit-identical by construction: the two blocks are disjoint
// ([0, window) vs [window, window + take)), the kernel writes the placeholder
// entries with the very expressions `comp_placeholder_kernel` uses (including
// the DEVICE-derived `take` - a captured graph freezes launch arguments, so it
// must not be a host value), and the `idxs` allocation is
// `window + index_topk + 8` (`chain_dev.rs`), which is why the launcher's grid
// is `max(window, hd, index_topk)` - the standalone launch used
// `ceil(index_topk / 128)` blocks, so every entry it could write is covered.
//
// `clen == nullptr` disables the placeholder half and reproduces
// `ring_win_fused_kernel` byte for byte, so a caller can route layers that must
// NOT take it (an index-source layer overwrites [window, ..) with its indexer
// afterwards; a compress-source layer's `*clen` is being advanced by this
// step's own compressor) through the same entry point.
__global__ void ring_win_fuse_ph_kernel(float* __restrict__ ring, const float* __restrict__ kv,
                                        const int* __restrict__ pos_ctr, int window, int hd,
                                        int32_t* __restrict__ idxs, const int* __restrict__ clen,
                                        int index_topk) {
    const int gid = threadIdx.x + (int)blockIdx.x * blockDim.x;
    // ---- half 3: the comp_placeholder recency entries ----
    if (clen != nullptr && gid < index_topk) {
        const int c = *clen;
        const int take = (c < index_topk) ? c : index_topk;
        if (gid < take) idxs[window + gid] = window + c - take + gid;
    }
    if (window <= 0) return;  // the two window halves are no-ops at window <= 0
    const int start_pos = *pos_ctr;
    if (ring != nullptr && gid < hd) {
        const int slot = start_pos % window;
        ring[(size_t)slot * (size_t)hd + gid] = kv[gid];
    }
    if (gid >= window) return;
    const int c = gid;
    if (start_pos == 0) {
        idxs[c] = (c == 0) ? 0 : -1;
        return;
    }
    const int oldest = (start_pos % window) + 1;
    long long idx = ((long long)c < (long long)window - oldest)
                        ? (long long)oldest + c
                        : (long long)c - ((long long)window - oldest);
    if (idx > (long long)start_pos) idx = -1;
    idxs[c] = (int)idx;
}

extern "C" int dsv41_ring_win_fuse_ph(float* ring, const float* kv, const int* pos_ctr, int window,
                                      int hd, int32_t* idxs, const int* clen, int index_topk,
                                      cudaStream_t s) {
    // NOT `window <= 0 -> return`: the placeholder half does not depend on
    // `window` and the standalone kernel it replaces ran unconditionally.
    int n = (window > hd) ? window : hd;
    if (clen != nullptr && index_topk > n) n = index_topk;
    if (n <= 0) return (int)cudaSuccess;
    ring_win_fuse_ph_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(ring, kv, pos_ctr, window,
                                                                       hd, idxs, clen, index_topk);
    return (int)cudaGetLastError();
}


// AR v5 launchers removed: DSV41 now calls the shared ferrite_p2p_ar_v5
// (ferrite_kernels.cu). The three DSV41 entry points (dsv41_ar_v5_store /
// _publish / _reduce) and their kernels lived here.

// ============================================================================
// extern "C" entry points -- the ABI in crates/ferrite-dsv41/src/kernels.rs
// ============================================================================

extern "C" int dsv41_engram_apply(float* x, const float* kv, const float* q_weight,
                                  const float* k_weight, const uint8_t* token_mask, int rows,
                                  int hc, int dim, float eps, cudaStream_t s) {
    if (rows <= 0 || hc <= 0 || dim <= 0) return (int)cudaSuccess;
    // 256 = 8 warps, the most block_sum3's __shared__ red[3][8] can combine
    // (the gated_rmsnorm lesson: never widen past a hardcoded warp count).
    const dim3 grid((unsigned)rows, (unsigned)hc);
    engram_apply_kernel<<<grid, 256, 0, s>>>(x, kv, q_weight, k_weight, token_mask, rows, hc, dim,
                                             eps);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_swiglu_limit(float* gate_up, int rows, int inter, float limit,
                                  cudaStream_t s) {
    if (rows <= 0 || inter <= 0) return (int)cudaSuccess;
    const size_t total = (size_t)rows * inter;
    const unsigned blocks = (unsigned)((total + 255) / 256);
    swiglu_limit_kernel<<<blocks, 256, 0, s>>>(gate_up, rows, inter, limit);
    return (int)cudaGetLastError();
}

// A4: swiglu + the fp8 pair the following GEMV consumes, in one launch. Returns
// 1 when the shape cannot use it (a caller that cannot meet the inter % 32 warp
// alignment keeps the swiglu_limit + quant1 pair), any other value is the usual
// cudaError_t status.
//
// ⚠️ LEGACY decline contract: 1 == cudaErrorInvalidValue, so a genuine error 1
// (incl. the trailing cudaGetLastError()) is indistinguishable from the decline
// and `Device::swiglu_limit_q_on` (rc == 1) swallows it as a fallback. Kept as-is
// per the rounds 37-41 freeze; migrate to decline == 2 if re-touched. M=1 row is the shared expert's down path.
extern "C" int dsv41_swiglu_limit_q(float* gate_up, int rows, int inter, float limit,
                                    uint8_t* xq, float* xsc, cudaStream_t s) {
    if (rows <= 0 || inter <= 0 || (inter & 31) || xq == nullptr || xsc == nullptr) return 1;
    const int wpb = 8;   // 256 threads = 8 warps, one 32-element scale block each
    const unsigned blocks =
        (unsigned)(((size_t)rows * (size_t)(inter >> 5) + wpb - 1) / wpb);
    swiglu_limit_q_kernel<<<blocks, wpb * 32, 0, s>>>(gate_up, rows, inter, limit, xq, xsc);
    return (int)cudaGetLastError();
}

// Batched form (DSV41_MOE_BATCH, default OFF): grid.y = the top-k slot, one
// launch per layer instead of one per (layer, slot). Identical arithmetic per
// slot - the slot only shifts the base pointer, so the result is bit-for-bit
// the sequential loop's. The batched gate/up writes each slot's [2*inter] block
// at a disjoint offset `slot * slot_stride`, and this kernel then rewrites the
// first `inter` floats of every block in place.
//
// MULTI-ROW (grid.z = the activation row): `rows` is now the ACTIVATION-ROW
// dimension, matching the expert gate/up and down launchers - the buffer is
// [rows][slot][2*inter], each row laid out exactly as the rows == 1 call laid
// out its single row, so this kernel's `gate_up` pointer and `slot_stride` mean
// the same thing for every row and only the row base moves:
//     base = gate_up + arow*gridDim.y*slot_stride + slot*slot_stride
// It therefore rewrites exactly the (row, slot) blocks that the rows=m gate/up
// launch wrote (its `out` row pitch is slots*out_slot_stride for the same
// reason). rows == 1 leaves blockIdx.z == 0, i.e. the pre-existing arithmetic.
//
// ⚠️ The former IN-SLOT row loop (`rows` rows inside one slot, pitch 2*inter)
// is GONE: it made `rows` mean two different things at once. It was only ever
// called with rows == 1 (chain_dev.rs:8030 and dspark_dev.rs:1396), where the
// two interpretations coincide, so every existing call is bit-identical.
__global__ void swiglu_limit_batched_kernel(float* __restrict__ gate_up, int rows, int inter,
                                            float limit, long slot_stride) {
    float* base = gate_up + (size_t)blockIdx.z * (size_t)gridDim.y * (size_t)slot_stride +
                  (size_t)blockIdx.y * (size_t)slot_stride;
    const size_t total = (size_t)inter;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)t;
        float* row = base;
        float g = row[i];
        float u = row[inter + i];
        if (limit > 0.f) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        row[i] = (g / (1.f + expf(-g))) * u;
    }
}

extern "C" int dsv41_swiglu_limit_batched(float* gate_up, int rows, int inter, float limit,
                                          long slot_stride, int slots, cudaStream_t s) {
    if (rows <= 0 || inter <= 0 || slots <= 0) return (int)cudaSuccess;
    const size_t total = (size_t)inter;
    dim3 grid((unsigned)((total + 255) / 256), (unsigned)slots, (unsigned)rows);
    swiglu_limit_batched_kernel<<<grid, 256, 0, s>>>(gate_up, rows, inter, limit, slot_stride);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_gather_rows(const float* src, const int32_t* idx, float* out, int n, int dim,
                                 cudaStream_t s) {
    if (n <= 0 || dim <= 0) return (int)cudaSuccess;
    const size_t total = (size_t)n * dim;
    const unsigned blocks = (unsigned)((total + 255) / 256);
    gather_rows_kernel<<<blocks, 256, 0, s>>>(src, idx, out, n, dim);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_scatter_add_rows(const float* src, const int32_t* idx, const float* weight,
                                      float* dst, int n, int dim, cudaStream_t s) {
    if (n <= 0 || dim <= 0) return (int)cudaSuccess;
    const size_t total = (size_t)n * dim;
    const unsigned blocks = (unsigned)((total + 255) / 256);
    scatter_add_rows_kernel<<<blocks, 256, 0, s>>>(src, idx, weight, dst, n, dim);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_hc_collapse(const float* x, const float* pre, float* out, int rows, int hc,
                                 int dim, cudaStream_t s) {
    if (rows <= 0 || hc <= 0 || dim <= 0) return (int)cudaSuccess;
    const size_t total = (size_t)rows * dim;
    const unsigned blocks = (unsigned)((total + 255) / 256);
    hc_collapse_kernel<<<blocks, 256, 0, s>>>(x, pre, out, rows, hc, dim);
    return (int)cudaGetLastError();
}

// Device-side all-reduce completion entry points (protocol in ar_stamp_kernel /
// ar_reduce_kernel above).
extern "C" int dsv41_ar_stamp(const unsigned long long* peer_stamps, int world, int rank,
                              unsigned round, cudaStream_t s) {
    if (world <= 0) return (int)cudaSuccess;
    ar_stamp_kernel<<<1, 32, 0, s>>>(peer_stamps, world, rank, round);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_ar_reduce2(float* dst, const float* staging, long n, long slot_f, int world,
                                const unsigned* stamps, unsigned round,
                                const unsigned long long* peer_reduced, int rank,
                                unsigned* ctr2, int do_mark, cudaStream_t s) {
    if (n <= 0 || world <= 0) return (int)cudaSuccess;
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 512) blocks = 512;
    ar_reduce_kernel<<<blocks, 256, 0, s>>>(dst, staging, n, slot_f, world, stamps, round,
                                            peer_reduced, rank, ctr2, do_mark);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_ar_store(const unsigned long long* peer_slots, int world, int rank,
                              const float* src, long n, long slot_f, cudaStream_t s) {
    if (n <= 0 || world <= 0) return (int)cudaSuccess;
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 512) blocks = 512;
    // legacy entry point: no parity, no credit wait (round 0 skips the wait)
    ar_store_kernel<<<blocks, 256, 0, s>>>(peer_slots, world, rank, src, n, slot_f, 0, nullptr, 0,
                                           nullptr, nullptr);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_gemv_bf16(const void* w, const float* x, float* out, int n, int k,
                               cudaStream_t s) {
    if (n <= 0 || k <= 0) return (int)cudaSuccess;
    unsigned blocks = (unsigned)((n + 7) / 8);
    if (blocks > 4096) blocks = 4096;
    // The kernel stages the activation row (k floats) in DYNAMIC shared memory;
    // passing 0 here makes that staging write past the allocation, which is what
    // the err 700 (illegal access) on this entry point was.
    const size_t smem = (size_t)k * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gemv_bf16_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             dsv41_smem_ceiling(nullptr));  // runtime ceiling
        if (e != cudaSuccess) return (int)e;
    }
    gemv_bf16_kernel<<<blocks, 256, smem, s>>>((const __nv_bfloat16*)w, x, out, n, k);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_gemv_f32(const float* w, const float* x, float* out, int n, int k,
                              cudaStream_t s) {
    if (n <= 0 || k <= 0) return (int)cudaSuccess;
    unsigned blocks = (unsigned)((n + 7) / 8);
    if (blocks > 4096) blocks = 4096;
    gemv_f32_kernel<<<blocks, 256, 0, s>>>(w, x, out, n, k);
    return (int)cudaGetLastError();
}

// Multi-row head GEMV: the verify block's head, one pass over the weight for all
// `m` rows (kernel + the bit-identity argument above). `m` selects a
// compile-time M so every row's accumulator is a register; m > 8 is refused —
// the caller then keeps its per-row loop, so an out-of-range call degrades to
// the old path instead of reading garbage. The verify block's VERIFY_ROWS is 6,
// so 8 leaves headroom without a general-m fallback kernel.
// `k % 8 != 0` is refused for the same reason (`ferrite_gemv_bf16_v2`'s host
// falls back to v1 on it): the vector body's uint4/float4 loads need 16B
// alignment, which a row stride of k bf16 only has when k is a multiple of 8 —
// the head's k is dim (5120), and the Rust caller declines before the launch so
// the verify keeps its per-row path instead of erroring.
// Launch shape copies dsv41_gemv_bf16's exactly (ceil(n/8) blocks capped at
// 4096, 256 threads = 8 warps) EXCEPT the dynamic smem: v1's launcher still
// reserves k*4 bytes for the activation staging that its 2026-09-11 revert
// removed, and this kernel declares no `extern __shared__` at all, so
// reserving it would only cap occupancy.
// Returns 0 (cudaSuccess) on a launch; the numeric payload is `out`.
extern "C" int dsv41_head_gemv_bf16_mrows(const void* w, const float* x, float* out, int m, int n,
                                          int k, cudaStream_t s) {
    if (m <= 0 || n <= 0 || k <= 0) return (int)cudaSuccess;
    if (m > 8) return (int)cudaErrorInvalidValue;
    if (k & 7) return (int)cudaErrorInvalidValue;
    unsigned blocks = (unsigned)((n + 7) / 8);
    if (blocks > 4096) blocks = 4096;
    const __nv_bfloat16* wb = (const __nv_bfloat16*)w;
    switch (m) {
        case 1: head_gemv_bf16_mrows_kernel<1><<<blocks, 256, 0, s>>>(wb, x, out, n, k); break;
        case 2: head_gemv_bf16_mrows_kernel<2><<<blocks, 256, 0, s>>>(wb, x, out, n, k); break;
        case 3: head_gemv_bf16_mrows_kernel<3><<<blocks, 256, 0, s>>>(wb, x, out, n, k); break;
        case 4: head_gemv_bf16_mrows_kernel<4><<<blocks, 256, 0, s>>>(wb, x, out, n, k); break;
        case 5: head_gemv_bf16_mrows_kernel<5><<<blocks, 256, 0, s>>>(wb, x, out, n, k); break;
        case 6: head_gemv_bf16_mrows_kernel<6><<<blocks, 256, 0, s>>>(wb, x, out, n, k); break;
        case 7: head_gemv_bf16_mrows_kernel<7><<<blocks, 256, 0, s>>>(wb, x, out, n, k); break;
        case 8: head_gemv_bf16_mrows_kernel<8><<<blocks, 256, 0, s>>>(wb, x, out, n, k); break;
        default: return (int)cudaErrorInvalidValue;
    }
    return (int)cudaGetLastError();
}

// ============================================================
// gemv_f32 v2 (vectorized float4 + K-split) — the f32 twin of
// gemv_bf16_v2_kernel (ferrite_kernels.cu:2731). The compressor's kvp/scp
// projections (n = head_dim = 128 rows, k = dim = 5120) are the only
// f32-weight M=1 GEMVs in the decode step (chain_dev.rs::lin_f32). v1 above
// launches ceil(128/8)=16 blocks x 8 warps = 128 warps = 11% of the 148 SMs
// with 160 SERIAL 4B loads per warp → 16.4us for a 2.6MB weight read
// (0.34us HBM floor at 7.6TB/s) = 48x off. Two diagnosed bottlenecks, the
// same pair the bf16 gate had before gemv_bf16_v2:
//   (a) scalar 4B loads   → float4 (16B) per lane-step (4x fewer LDG, 4x
//       fewer loop iterations). NOTE: unlike bf16 (uint4 = 8 values) an
//       f32 float4 carries only 4 values, so the loop is 2x longer per
//       byte — K-split is what actually buys the latency hiding, and the
//       two combine to ~4x the in-flight bytes of v1.
//   (b) latency-bound medium matrices → K-split WPR warps per row, folded
//       in smem (no atomics, no second kernel): n=128, WPR=8 → 1024 warps
//       (~7/SM) vs v1's 128 (~0.9/SM).
// Numerics: each multiply-accumulate is pinned to the fused rounding of v1
// (v1's `acc += w*x` contracts to a single FFMA under fmad=on) via
// __fmaf_rn — never let --use_fast_math reassociate this chain (build.sh
// documents a 1-ULP/layer drift from an unpinned plain-operator rewrite).
// The only difference vs v1 is the cross-K-slice fold order (~1e-6 f32),
// far below the compressor's tolerance (see gemv_bf16_v2's note).
// Grid: ceil(n/rpb) blocks, 256 threads = 8 warps; WPR ∈ {1,2,4,8}.
// ============================================================
template <int WPR>
__global__ void gemv_f32_v2_kernel(const float* __restrict__ w,
                                   const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int n, int k) {
    const int warps = blockDim.x >> 5;
    const int rpb = warps / WPR;               // rows per block
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * rpb + warp / WPR;
    const int kw = warp % WPR;                 // K-slice id
    float acc = 0.f;
    if (row < n) {
        const float* wr = w + (size_t)row * k;
        // Slice size rounded to 4: float4 loads need 16B alignment.
        // k % 4 == 0 is guaranteed by the host fallback to v1, so k1-k0 is
        // 4-aligned and the per-lane `c + 3 < k1` guard covers every element
        // of [k0, k1) (lanes sweep the slice in 128-element blocks).
        int kper = ((k + WPR - 1) / WPR + 3) & ~3;
        int k0 = kw * kper;
        int k1 = min(k0 + kper, k);
#pragma unroll 2
        for (int c = k0 + lane * 4; c + 3 < k1; c += 32 * 4) {
            float4 wv = *reinterpret_cast<const float4*>(wr + c);
            float4 xv = *reinterpret_cast<const float4*>(x + c);
            acc = __fmaf_rn(wv.x, xv.x, acc);
            acc = __fmaf_rn(wv.y, xv.y, acc);
            acc = __fmaf_rn(wv.z, xv.z, acc);
            acc = __fmaf_rn(wv.w, xv.w, acc);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (WPR == 1) {
        if (lane == 0 && row < n) out[row] = acc;
    } else {
        __shared__ float part[16];
        if (lane == 0) part[warp] = acc;
        __syncthreads();
        if (kw == 0 && lane == 0 && row < n) {
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < WPR; j++) sum += part[(warp / WPR) * WPR + j];
            out[row] = sum;
        }
    }
}

extern "C" int dsv41_gemv_f32_v2(const float* w, const float* x, float* out, int n, int k,
                                 cudaStream_t s) {
    if (n <= 0 || k <= 0) return (int)cudaSuccess;
    // float4 needs k % 4 == 0; the compressor k=5120 qualifies. Anything else
    // (no such f32 shape exists today) keeps v1 rather than taking a slow
    // scalar tail.
    if (k & 3) return dsv41_gemv_f32(w, x, out, n, k, s);
    // WPR heuristic mirrored from ferrite_gemv_bf16_v2: enough warps to cover
    // HBM latency (n*WPR/8 warps total; the compressor's n=128 → WPR=8 →
    // 1024 warps ≈ 7/SM).
    int wpr = n >= 16384 ? 1 : (n >= 4096 ? 2 : (n >= 1024 ? 4 : 8));
    int rpb = 8 / wpr;
    dim3 grid((n + rpb - 1) / rpb);
    dim3 block(256);
    switch (wpr) {
        case 1: gemv_f32_v2_kernel<1><<<grid, block, 0, s>>>(w, x, out, n, k); break;
        case 2: gemv_f32_v2_kernel<2><<<grid, block, 0, s>>>(w, x, out, n, k); break;
        case 4: gemv_f32_v2_kernel<4><<<grid, block, 0, s>>>(w, x, out, n, k); break;
        default: gemv_f32_v2_kernel<8><<<grid, block, 0, s>>>(w, x, out, n, k); break;
    }
    return (int)cudaGetLastError();
}

extern "C" int dsv41_ar_store2(const unsigned long long* peer_slots, int world, int rank,
                               const float* src, long n, long slot_f, long parity_off,
                               const unsigned* reduced, unsigned round,
                               const unsigned long long* peer_stamps, unsigned* ctr,
                               cudaStream_t s) {
    if (n <= 0 || world <= 0) return (int)cudaSuccess;
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 512) blocks = 512;
    ar_store_kernel<<<blocks, 256, 0, s>>>(peer_slots, world, rank, src, n, slot_f, parity_off,
                                           reduced, round, peer_stamps, ctr);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_ar_mark(const unsigned long long* peer_reduced, int world, int rank,
                             unsigned round, cudaStream_t s) {
    if (world <= 0) return (int)cudaSuccess;
    ar_mark_kernel<<<1, 32, 0, s>>>(peer_reduced, world, rank, round);
    return (int)cudaGetLastError();
}

// ===========================================================================
// DSpark Markov head  (dspark.rs::forward_head, the sequential sampling loop)
// ===========================================================================
//
// The draft head's last stage. For draft row `step` (0-based), exactly as the
// host reference does it:
//
//   1. read the ALREADY SAMPLED token `ids[step]` and its Markov embedding row
//      `er = markov_embed[ids[step]]`                                    [mr]
//   2. bias that row's logits IN PLACE, one GEMV over the vocabulary:
//        logits[step][v] += <markov_head[v], er>                         [vocab]
//   3. sample the next token `ids[step+1]` from the biased row
//   4. score the row's confidence from the collapsed hidden and `er`
//
// One launch covers ONE step; `draft_forward` issues `bs` of them in order,
// because `er` depends on the previous step's sample and nothing on the device
// can break that chain (a grid-wide sync needs a cooperative launch the
// runtime does not expose, and a "one block, loop over the vocabulary" form
// would read the 128 MiB Markov head at one SM's bandwidth: measured single-SM
// streaming is ~50 GB/s, i.e. ~2.6 ms per step against a ~1 ms whole-draft
// budget). The many-block form below reads the head ONCE per step at full HBM
// bandwidth (~35 us for 132 MiB at 3.8 TB/s effective) and is therefore the
// only form that fits the budget.
//
// ---- sampling numerics (the discipline the caller must preserve) ----
// `ops::gumbel_argmax` is `argmax(logits)` when `temperature == 0`, and
// otherwise `argmax(softmax(logits/t) / max(u, 1e-30))`. The chain calls it
// with `u == 1.0` for EVERY entry (chain.rs uploads `vec![1.0; bs * vocab]`),
// for which the ratio is `softmax(...)` = `exp((l-mx)/t) / den`. `den` is the
// SAME positive constant for every `v`, `expf` and division by a positive
// constant are both monotone in IEEE-754, so the winner is exactly
// `argmax(logits)` with ties going to the LOWEST index (the host's ascending
// strict-`>` scan). The kernel therefore reduces to one stable argmax of the
// biased row, which is bit-identical to the host at every temperature and
// needs a single pass. A future NON-constant `u` (per-entry Gumbel noise) would
// need `den` and hence a second pass - it is deliberately not implemented here.
//
// Tie rule: the packed key is (monotone(value) << 32) | (0xFFFFFFFF - v), the
// same construction `argmax_kernel` (dsv41_kernels.cu) uses, so a plain
// unsigned max over the keys picks the largest value and, among equals, the
// lowest index.
__device__ __forceinline__ unsigned dspark_markov_f2key(float f) {
    const unsigned b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Warps per block (the block reduction holds up to 32).
#define DSPARK_MARKOV_WARPS 8
// Vocab rows each warp covers before the grid strides: 129280 rows / (8*8)
// = 2020 blocks, i.e. one partial per block with a cheap last-block fold.
#define DSPARK_MARKOV_ROWS_PER_WARP 8
// Hard cap on the grid == the size of the caller's `partial` scratch.
#define DSPARK_MARKOV_MAX_BLOCKS 2048

__global__ void dspark_markov_head_kernel(
    float* __restrict__ logits,                 // [bs, vocab], row `step` biased in place
    const float* __restrict__ h,                // [bs, dim] collapsed (PRE-norm) hidden
    const float* __restrict__ markov_embed,     // [vocab, mr]
    const float* __restrict__ markov_head,      // [vocab, mr]
    const float* __restrict__ confidence_proj,  // [dim + mr] or null
    int* __restrict__ ids,                      // [bs + 1]; ids[step] in, ids[step+1] out
    float* __restrict__ confidence,             // [bs] or null
    int dim, int vocab, int mr, int step,
    unsigned long long* __restrict__ partial,   // [gridDim.x] scratch
    unsigned* __restrict__ ctr) {               // 1 u32, starts at 0, self-resets
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarp = (int)blockDim.x >> 5;
    const int tok = ids[step];
    const float* __restrict__ er = markov_embed + (size_t)tok * (size_t)mr;
    float* __restrict__ lrow = logits + (size_t)step * (size_t)vocab;

    // ---- per-warp: one vocab row per iteration, all 32 lanes on its [mr] dot ----
    // A whole warp per row keeps the markov_head read fully coalesced (lanes 0..31
    // walk consecutive 4-float groups of one row); a thread-per-row layout would
    // stride by `mr` and waste 7/8 of every 128 B line.
    unsigned long long best = 0ull;
    // EVERY lane of the warp walks the same `v` sequence (the bounds do not
    // depend on the lane), so the shuffle reductions below are always executed
    // with a fully converged warp - the mask is 0xFFFFFFFF throughout.
    for (int v = blockIdx.x * nwarp + wid; v < vocab; v += gridDim.x * nwarp) {
        const float* __restrict__ wr = markov_head + (size_t)v * (size_t)mr;
        float acc = 0.f;
        if ((mr & 3) == 0) {
            // Rows are 16 B aligned whenever mr % 4 == 0 (the tensor base is a
            // cudaMalloc pointer), so the float4 form is safe.
            const float4* w4 = reinterpret_cast<const float4*>(wr);
            const float4* e4 = reinterpret_cast<const float4*>(er);
            const int n4 = mr >> 2;
            for (int c = lane; c < n4; c += 32) {
                const float4 wv = w4[c], ev = e4[c];
                acc = __fmaf_rn(wv.x, ev.x, acc);
                acc = __fmaf_rn(wv.y, ev.y, acc);
                acc = __fmaf_rn(wv.z, ev.z, acc);
                acc = __fmaf_rn(wv.w, ev.w, acc);
            }
        } else {
            for (int c = lane; c < mr; c += 32) acc = __fmaf_rn(wr[c], er[c], acc);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) {
            const float l = lrow[v] + acc;
            lrow[v] = l;   // the bias is applied ONCE per row (row `step` only)
            const unsigned long long k =
                ((unsigned long long)dspark_markov_f2key(l) << 32) |
                (unsigned long long)(0xFFFFFFFFu - (unsigned)v);
            if (k > best) best = k;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, best, off);
        if (o > best) best = o;
    }

    // ---- block reduction ----
    __shared__ unsigned long long sb[32];
    if (lane == 0) sb[wid] = best;
    __syncthreads();
    if (wid == 0) {
        unsigned long long b2 = (lane < nwarp) ? sb[lane] : 0ull;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, b2, off);
            if (o > b2) b2 = o;
        }
        if (lane == 0) partial[blockIdx.x] = b2;
    }
    // The `sb` slots are reused by the last-block fold below; this barrier is
    // what stops a late warp from clobbering them before warp 0 has read them.
    __syncthreads();

    // ---- confidence (independent of the argmax; block 0's first warp) ----
    if (confidence != nullptr && confidence_proj != nullptr && blockIdx.x == 0 && wid == 0) {
        float acc = 0.f;
        const float* __restrict__ hr = h + (size_t)step * (size_t)dim;
        for (int c = lane; c < dim; c += 32) acc = __fmaf_rn(confidence_proj[c], hr[c], acc);
        for (int c = lane; c < mr; c += 32)
            acc = __fmaf_rn(confidence_proj[dim + c], er[c], acc);
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) confidence[step] = acc;
    }

    // ---- last-block election: fold the per-block winners and publish the id ----
    // Standard counting barrier: every block publishes its partial, fences, then
    // takes a ticket; the block that takes the LAST ticket sees every partial.
    __threadfence();
    __shared__ unsigned is_last;
    if (threadIdx.x == 0) {
        const unsigned old = atomicAdd(ctr, 1u);
        is_last = (old == gridDim.x - 1) ? 1u : 0u;
    }
    __syncthreads();
    if (is_last) {
        __threadfence();
        unsigned long long g = 0ull;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            const unsigned long long k = partial[i];
            if (k > g) g = k;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, g, off);
            if (o > g) g = o;
        }
        if (lane == 0) sb[wid] = g;
        __syncthreads();
        if (threadIdx.x == 0) {
            g = 0ull;
            for (int w = 0; w < nwarp; w++) {
                if (sb[w] > g) g = sb[w];
            }
            ids[step + 1] = (int)(0xFFFFFFFFu - (unsigned)(g & 0xFFFFFFFFu));
            // Self-reset: the NEXT step's launch (same stream) must start at zero.
            // Every block has already taken its ticket, so nothing can race this.
            *ctr = 0u;
        }
    }
}

extern "C" int dsv41_dspark_markov_head(float* logits, const float* h, const float* markov_embed,
                                        const float* markov_head, const float* confidence_proj,
                                        int* ids, float* confidence, int dim, int vocab, int mr,
                                        int step, unsigned long long* partial, unsigned* ctr,
                                        cudaStream_t s) {
    if (vocab <= 0 || mr <= 0 || dim <= 0 || step < 0) return (int)cudaErrorInvalidValue;
    if (partial == nullptr || ctr == nullptr) return (int)cudaErrorInvalidValue;
    int blocks = (vocab + DSPARK_MARKOV_WARPS * DSPARK_MARKOV_ROWS_PER_WARP - 1) /
                 (DSPARK_MARKOV_WARPS * DSPARK_MARKOV_ROWS_PER_WARP);
    if (blocks < 1) blocks = 1;
    if (blocks > DSPARK_MARKOV_MAX_BLOCKS) blocks = DSPARK_MARKOV_MAX_BLOCKS;
    dspark_markov_head_kernel<<<blocks, DSPARK_MARKOV_WARPS * 32, 0, s>>>(
        logits, h, markov_embed, markov_head, confidence_proj, ids, confidence, dim, vocab, mr,
        step, partial, ctr);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// DSpark verify: the m-row block append + per-row CAUSAL window indices, in
// one launch (the verify twin of `ring_win_fused_kernel`).
//
// The verify block's rows sit at positions base..base+m-1 where base =
// *pos_ctr (the anchor's position - the counter is advanced by the argmax at
// the END of a step, so during the verify forward it still holds the anchor).
// Row r's query must attend everything up to and including position base+r;
// its window enumerates positions [base+r-window+1 .. base+r], whose slots are
// exactly the block's own rows 0..r plus the surviving history - the ring
// geometry gives the intra-block causal order for free (window > m always).
//
// The append half writes all m kv rows into their slots ((base+j) % window,
// mutually distinct because m <= window). The indices half fills
// idxs[r * window + c] with the same arithmetic as `window_idxs_kernel`'s
// decode branch, with start_pos = base + r per row.
__global__ void verify_ring_win_kernel(float* __restrict__ ring, const float* __restrict__ kv,
                                       const int* __restrict__ pos_ctr, int window, int hd, int m,
                                       int32_t* __restrict__ idxs) {
    const int base = *pos_ctr;
    // ---- append half: m*hd elements, strided grid-stride loop ----
    const int total = m * hd;
    for (int e = threadIdx.x + (int)blockIdx.x * blockDim.x; e < total;
         e += gridDim.x * blockDim.x) {
        const int j = e / hd, i = e % hd;
        const int slot = (base + j) % window;
        ring[(size_t)slot * (size_t)hd + (size_t)i] = kv[(size_t)e];
    }
    // ---- indices half: m*window entries, first block only (it is tiny) ----
    if (blockIdx.x != 0) return;
    for (int e = threadIdx.x; e < m * window; e += blockDim.x) {
        const int r = e / window, c = e % window;
        const int start_pos = base + r;
        int idx;
        if (start_pos == 0) {
            idx = (c == 0) ? 0 : -1;
        } else {
            const int oldest = (start_pos % window) + 1;
            long long v = ((long long)c < (long long)window - oldest)
                              ? (long long)oldest + c
                              : (long long)c - ((long long)window - oldest);
            if (v > (long long)start_pos) v = -1;
            idx = (int)v;
        }
        idxs[(size_t)e] = idx;
    }
}

extern "C" int dsv41_verify_ring_win(float* ring, const float* kv, const int* pos_ctr,
                                     int window, int hd, int m, int32_t* idxs,
                                     cudaStream_t s) {
    if (window <= 0 || m <= 0) return (int)cudaSuccess;
    const int n = (m * hd > m * window) ? m * hd : m * window;
    const unsigned blocks = (unsigned)((n + 127) / 128);
    verify_ring_win_kernel<<<blocks, 128, 0, s>>>(ring, kv, pos_ctr, window, hd, m, idxs);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// DSpark verify: the snapshot / rollback (save / restore) pair for the write
// set of a speculative block.
//
// `DevChain::dspark_snapshot` / `dspark_rollback_keep` used to move that write
// set one `cudaMemcpyAsync` at a time. The ring half is `m` slots per owner
// layer, and the slots `(pos + 1 + j) % window` wrap at the ring's end, so a
// single contiguous copy cannot serve them; the compressor half is three
// segments (`state_kv`, `state_score`, `latent`) plus two 4-byte counters per
// source layer. That is ~520 stream submissions for the production geometry
// (40 owners x 6 slots, both directions), and at ~5.4us of launch overhead
// apiece they were the bulk of the 2.7ms a DSpark step spent in save/restore
// (dspark-verify-perf-plan P0).
//
// These kernels collapse the per-slot copies into ONE launch per layer per
// direction. The slot arithmetic is still exactly `(pos + 1 + j) % window`;
// the only change is that it is evaluated on the DEVICE. That is the P0 red
// line: a host-computed slot is a capture-time constant, which is exactly what
// made the old pair impossible to put inside a CUDA graph.
//
// Determinism: save and restore are pure element moves. Every destination
// element is produced by exactly one thread from exactly one source element,
// so the bytes are bit-identical to the `cudaMemcpyAsync` sequence they
// replace -- "the snapshot is the same snapshot" is a property of the layout,
// not of the emission order.
//
// `pos_base` is the caller's `pos + 1` (the first row's slot base). `keep` is
// `dspark_rollback_keep`'s accepted prefix: rows `0..keep` STAY in the ring
// (they are the accepted prefix's KV) and only rows `keep..m` come back.

__global__ void dspark_ring_save_kernel(float* __restrict__ snap, const float* __restrict__ ring,
                                        int pos_base, int win, int hd, int m) {
    const int total = m * hd;
    for (int e = (int)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (int)gridDim.x * blockDim.x) {
        const int j = e / hd;
        const int i = e - j * hd;
        const int slot = (pos_base + j) % win;
        snap[(size_t)e] = ring[(size_t)slot * (size_t)hd + (size_t)i];
    }
}

__global__ void dspark_ring_restore_kernel(float* __restrict__ ring, const float* __restrict__ snap,
                                           int pos_base, int win, int hd, int m, int keep) {
    const int rows = m - keep;
    const int total = rows * hd;
    for (int e = (int)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (int)gridDim.x * blockDim.x) {
        const int lr = e / hd;  // row within the restored `keep..m` range
        const int i = e - lr * hd;
        const int j = keep + lr;  // the SNAPSHOT row index (the two layouts match)
        const int slot = (pos_base + j) % win;
        ring[(size_t)slot * (size_t)hd + (size_t)i] = snap[(size_t)j * (size_t)hd + (size_t)i];
    }
}

// The compressor carry of ONE source layer, all of it in one launch. Element
// index space:
//   [0, n)       -> state_kv            (n = ratio * hd)
//   [n, 2n)      -> state_score
//   [2n, 2n + hd)-> latent
// and thread 0 of block 0 moves the two 4-byte counters, so every buffer
// `dspark_snapshot` saves for this layer is covered. `snap_state` is this
// layer's `2 * max_ratio * hd` slice: `state_kv` at [0, ratio * hd) and
// `state_score` at [max_ratio * hd, max_ratio * hd + ratio * hd) -- the
// buffers are max-sized because `ratio` is per layer (1 or 2 here).
__global__ void dspark_comp_save_kernel(const float* __restrict__ state_kv,
                                        const float* __restrict__ state_score,
                                        const float* __restrict__ latent,
                                        const int* __restrict__ clen,
                                        const int* __restrict__ out_rows,
                                        float* __restrict__ snap_state,
                                        float* __restrict__ snap_latent,
                                        int* __restrict__ snap_clen,
                                        int* __restrict__ snap_out_rows, int ratio, int max_ratio,
                                        int hd) {
    const int n = ratio * hd;
    const int total = 2 * n + hd;
    for (int e = (int)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (int)gridDim.x * blockDim.x) {
        if (e < n) {
            snap_state[e] = state_kv[e];
        } else if (e < 2 * n) {
            const int c = e - n;
            snap_state[(size_t)max_ratio * hd + c] = state_score[c];
        } else {
            const int c = e - 2 * n;
            snap_latent[c] = latent[c];
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *snap_clen = *clen;
        *snap_out_rows = *out_rows;
    }
}

__global__ void dspark_comp_restore_kernel(float* __restrict__ state_kv,
                                           float* __restrict__ state_score,
                                           float* __restrict__ latent,
                                           int* __restrict__ clen, int* __restrict__ out_rows,
                                           const float* __restrict__ snap_state,
                                           const float* __restrict__ snap_latent,
                                           const int* __restrict__ snap_clen,
                                           const int* __restrict__ snap_out_rows, int ratio,
                                           int max_ratio, int hd) {
    const int n = ratio * hd;
    const int total = 2 * n + hd;
    for (int e = (int)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (int)gridDim.x * blockDim.x) {
        if (e < n) {
            state_kv[e] = snap_state[e];
        } else if (e < 2 * n) {
            const int c = e - n;
            state_score[c] = snap_state[(size_t)max_ratio * hd + c];
        } else {
            const int c = e - 2 * n;
            latent[c] = snap_latent[c];
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *clen = *snap_clen;
        *out_rows = *snap_out_rows;
    }
}

// The op count is `m * hd` (ring) or `2 * ratio * hd + hd` (compressor), i.e.
// a few thousand elements; 256 threads with one block per 256 elements is the
// same geometry `dsv41_dspark_markov_head` uses and keeps the grid in a single
// wave at these sizes.
extern "C" int dsv41_dspark_ring_save(float* snap, const float* ring, int pos_base, int win, int hd,
                                      int m, cudaStream_t s) {
    if (snap == nullptr || ring == nullptr || win <= 0 || hd <= 0 || m <= 0)
        return (int)cudaErrorInvalidValue;
    // perf-review finding 1: `slot = (pos_base + j) % win` is injective only for
    // `m <= win`. Past that two rows of the SAME layer map to ONE slot, so the
    // grid's blocks race on that destination and the snapshot stops being the
    // deterministic element move the header above promises. Decline (2) -- never
    // 1, which is cudaErrorInvalidValue and must stay a real device failure --
    // so the caller falls back to the per-slot host loop, which serialises the
    // colliding slots in j order.
    if (m > win) return 2;
    // perf-review finding 3: the Rust bound on `m` is `dspark_snapshot`'s bare
    // `debug_assert!(m <= VERIFY_ROWS)`, which release builds compile away.
    // `snap` here is ONE layer's `VERIFY_ROWS`-row slice (chain_dev.rs sizes
    // `dspark_snap_ring` as `n_layers * VERIFY_ROWS * head_dim`), so a longer
    // block would run past the last layer's slice. Enforce the same capacity at
    // the single entry every call goes through. VERIFY_ROWS is 6 today; this
    // constant has to follow it.
    if (m > 6) return 2;
    const unsigned blocks = (unsigned)((m * hd + 255) / 256);
    dspark_ring_save_kernel<<<blocks, 256, 0, s>>>(snap, ring, pos_base, win, hd, m);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_dspark_ring_restore(float* ring, const float* snap, int pos_base, int win,
                                         int hd, int m, int keep, cudaStream_t s) {
    if (ring == nullptr || snap == nullptr || win <= 0 || hd <= 0 || m <= 0)
        return (int)cudaErrorInvalidValue;
    // perf-review finding 1: same slot-map aliasing as `dsv41_dspark_ring_save`
    // above -- for `m > win` several rows resolve to one `(pos_base + j) % win`
    // slot and the restore's reads/writes race. Decline (2) so the caller keeps
    // its per-slot host loop.
    if (m > win) return 2;
    // perf-review finding 3: `m <= VERIFY_ROWS` is only debug-asserted on the
    // Rust side; `snap` is one layer's `VERIFY_ROWS`-row slice, so cap it here
    // too (see `dsv41_dspark_ring_save`).
    if (m > 6) return 2;
    if (keep < 0) keep = 0;
    const int rows = m - keep;
    // `keep == m` (the whole block was accepted) restores nothing. Returning
    // success here -- instead of launching a 0-block grid -- matches the host
    // loop this replaces, which simply ran no iterations.
    if (rows <= 0) return (int)cudaSuccess;
    const unsigned blocks = (unsigned)((rows * hd + 255) / 256);
    dspark_ring_restore_kernel<<<blocks, 256, 0, s>>>(ring, snap, pos_base, win, hd, m, keep);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_dspark_comp_save(const float* state_kv, const float* state_score,
                                      const float* latent, const int* clen, const int* out_rows,
                                      float* snap_state, float* snap_latent, int* snap_clen,
                                      int* snap_out_rows, int ratio, int max_ratio, int hd,
                                      cudaStream_t s) {
    if (state_kv == nullptr || state_score == nullptr || latent == nullptr || clen == nullptr ||
        out_rows == nullptr || snap_state == nullptr || snap_latent == nullptr ||
        snap_clen == nullptr || snap_out_rows == nullptr)
        return (int)cudaErrorInvalidValue;
    if (hd <= 0 || ratio <= 0 || max_ratio <= 0)
        return (int)cudaErrorInvalidValue;
    // perf-review finding 2: `ratio > max_ratio` is a SHAPE this kernel cannot
    // serve -- `snap_state` is only `2 * max_ratio * hd` wide and the
    // `state_score` segment is written at `+max_ratio * hd`, so the copy would
    // land past the layer's slice. That is a decline, not a device failure:
    // returning cudaErrorInvalidValue (1) made it indistinguishable from a real
    // launch error. 2 is the file's fallback code, the same one the fused
    // kernels use for a shape they decline.
    if (ratio > max_ratio) return 2;
    const unsigned blocks = (unsigned)((2 * ratio * hd + hd + 255) / 256);
    dspark_comp_save_kernel<<<blocks, 256, 0, s>>>(state_kv, state_score, latent, clen, out_rows,
                                                   snap_state, snap_latent, snap_clen, snap_out_rows,
                                                   ratio, max_ratio, hd);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_dspark_comp_restore(float* state_kv, float* state_score, float* latent,
                                         int* clen, int* out_rows, const float* snap_state,
                                         const float* snap_latent, const int* snap_clen,
                                         const int* snap_out_rows, int ratio, int max_ratio, int hd,
                                         cudaStream_t s) {
    if (state_kv == nullptr || state_score == nullptr || latent == nullptr || clen == nullptr ||
        out_rows == nullptr || snap_state == nullptr || snap_latent == nullptr ||
        snap_clen == nullptr || snap_out_rows == nullptr)
        return (int)cudaErrorInvalidValue;
    if (hd <= 0 || ratio <= 0 || max_ratio <= 0)
        return (int)cudaErrorInvalidValue;
    // perf-review finding 2: the `ratio > max_ratio` shape decline, mirroring
    // `dsv41_dspark_comp_save` above -- 2 (fallback), not 1 (a real failure).
    if (ratio > max_ratio) return 2;
    const unsigned blocks = (unsigned)((2 * ratio * hd + hd + 255) / 256);
    dspark_comp_restore_kernel<<<blocks, 256, 0, s>>>(state_kv, state_score, latent, clen, out_rows,
                                                      snap_state, snap_latent, snap_clen,
                                                      snap_out_rows, ratio, max_ratio, hd);
    return (int)cudaGetLastError();
}
