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
    const float* h = x + ((size_t)r * hc + i) * dim;             // read + written back
    const float* k = kv + (size_t)r * span + (size_t)i * dim;
    const float* value = kv + (size_t)r * span + (size_t)hc * dim;
    const float* qw = q_weight + (size_t)i * dim;
    const float* kw = k_weight + (size_t)i * dim;

    float hss = 0.f, kss = 0.f, dot = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float hv = h[c];
        const float kval = k[c];
        hss += hv * hv;
        kss += kval * kval;
        dot += hv * qw[c] * kw[c] * kval;
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

    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        x[((size_t)r * hc + i) * dim + c] = h[c] + gate * value[c];
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
// kernels (ar_v5_store/publish/reduce) and their extern "C" launchers were
// deleted — one protocol, one implementation.

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
    // Baseline single-chain loop: the four-way manual unroll that lived here
    // since a4053cd turned out to be the second-round regression. With
    // --use_fast_math, four INDEPENDENT `acc += w*x` expressions are
    // reassociable (the single dependency chain is not), and the drift
    // accumulates over 40 layers into a degenerate model. The isolation
    // comparison had missed it because it compared single-chain vs fmaf,
    // not unrolled vs single-chain.
    for (int row = blockIdx.x * nwarp + wid; row < n; row += gridDim.x * nwarp) {
        const __nv_bfloat16* wr = w + (size_t)row * (size_t)k;
        float acc = 0.f;
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
    // Baseline single-chain loop (same as gemv_bf16 - see the note there).
    for (int row = blockIdx.x * nwarp + wid; row < n; row += gridDim.x * nwarp) {
        const float* wr = w + (size_t)row * (size_t)k;
        float acc = 0.f;
        for (int c = lane; c < k; c += 32) acc += wr[c] * x[c];
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) out[row] = acc;
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
// head): out = rolling.rem_euclid(lm) + off. Single thread on purpose: a few
// hundred integer ops once per step, and serial execution keeps it bit-identical
// to the host reference (which is also serial).
__global__ void engram_hash_step_kernel(const long long* __restrict__ map, long long* __restrict__ cache,
                                   const long long* __restrict__ mults,
                                   const unsigned long long* __restrict__ lms,
                                   const unsigned long long* __restrict__ offs,
                                   long long* __restrict__ eng_ids, const int* __restrict__ token,
                                   const int* __restrict__ pos_ctr, long long map_len, int n_layers,
                                   int max_ngram, int n_heads, long long pad_id) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int p = *pos_ctr;
    const long long t = (long long)token[0];
    cache[p] = ((unsigned long long)t < (unsigned long long)map_len) ? map[t] : 0;
    long long tokens[8];
    bool blocked = false;
    for (int shift = 0; shift < max_ngram; ++shift) {
        const long long q = p - (long long)shift;
        const long long src = (q >= 0) ? cache[q] : 0;
        blocked = blocked || (q < 0) || (src == -1);  // -1 == DEAD
        tokens[shift] = blocked ? pad_id : src;
    }
    const int n_cols = (max_ngram - 1) * n_heads;
    for (int li = 0; li < n_layers; ++li) {
        const long long* m = mults + (size_t)li * 4;
        long long rolling = tokens[0] * m[0];
        for (int i = 1; i < max_ngram; ++i) {
            rolling ^= tokens[i] * m[i];
            for (int h = 0; h < n_heads; ++h) {
                const int col = (i - 1) * n_heads + h;
                const long long lm = (long long)lms[(size_t)li * n_cols + col];
                const long long off = (long long)offs[(size_t)li * n_cols + col];
                long long v = rolling % lm;
                if (v < 0) v += lm;  // rem_euclid
                eng_ids[(size_t)li * n_cols + col] = v + off;
            }
        }
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
    engram_hash_step_kernel<<<1, 32, 0, s>>>(map, cache, mults, lms, offs, eng_ids, token, pos_ctr,
                                        map_len, n_layers, max_ngram, n_heads, pad_id);
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
    const dim3 grid((unsigned)rows, (unsigned)hc);
    engram_apply_kernel<<<grid, 128, 0, s>>>(x, kv, q_weight, k_weight, token_mask, rows, hc, dim,
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

// Batched form (DSV41_MOE_BATCH, default OFF): grid.y = the top-k slot, one
// launch per layer instead of one per (layer, slot). Identical arithmetic per
// slot - the slot only shifts the base pointer, so the result is bit-for-bit
// the sequential loop's. The batched gate/up writes each slot's [2*inter] block
// at a disjoint offset `slot * slot_stride`, and this kernel then rewrites the
// first `inter` floats of every block in place.
__global__ void swiglu_limit_batched_kernel(float* __restrict__ gate_up, int rows, int inter,
                                            float limit, long slot_stride) {
    float* base = gate_up + (size_t)blockIdx.y * (size_t)slot_stride;
    const size_t total = (size_t)rows * inter;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(t / (size_t)inter);
        const int i = (int)(t % (size_t)inter);
        float* row = base + (size_t)r * 2 * inter;
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
    const size_t total = (size_t)rows * inter;
    dim3 grid((unsigned)((total + 255) / 256), (unsigned)slots);
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
    // The kernel stages the (shared) activation row in shared memory: k floats.
    // The size is the caller's to pass - launching with 0, as the first cut of
    // this staging did, points s_x at an empty allocation and the staging writes
    // walk off the end (faults=4 with empty outputs on the first deployment).
    const size_t smem = (size_t)k * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gemv_bf16_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
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
    // Same as gemv_bf16: the kernel stages the activation row in shared memory,
    // so the size is the caller's to pass. Launching with 0 made that staging
    // walk off an empty allocation on the sibling kernel.
    const size_t smem = (size_t)k * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gemv_f32_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
        if (e != cudaSuccess) return (int)e;
    }
    gemv_f32_kernel<<<blocks, 256, smem, s>>>(w, x, out, n, k);
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
