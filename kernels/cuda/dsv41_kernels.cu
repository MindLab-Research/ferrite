// DeepSeek-V4.1-Flash kernels.
//
// ============================================================================
// PERFORMANCE CONTRACT (non-negotiable)
// ============================================================================
//  1. No weight is ever dequantised into a bf16/f32 buffer. fp8/fp4 weights
//     stay packed in device memory for the whole run.
//  2. Every large matmul is a tensor-core MMA over the native format.
//  3. ue8m0 block scales are applied per k-block in the epilogue with a
//     separate accumulator -- the reference fp8_gemm_kernel scheme:
//         acc += dot(a_k, b_k) * scale_a[row, kblk] * scale_b[nblk, kblk]
//
// Verified instruction availability on sm_103a (B300, CUDA 13.2, ptxas, probed
// with an explicit -gencode arch=compute_103a,code=sm_103a and a confirmed
// `.target sm_103a`):
//
//   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32      -> OK   (used here)
//   mma.sync ... kind::f8f6f4 / any fp4 e2m1 mma.sync        -> REJECTED
//        "Instruction 'mma with FP6/FP4 floating point type' not supported
//         on .target 'sm_103a'"
//   tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
//        -> the ONLY fp4 entry point on Blackwell (tmem accumulator, smem
//           operand + tmem block-scale descriptors, mbarrier completion).
//
// Therefore:
//   * DENSE fp8 GEMMs run the native e4m3 m16n8k32 MMA here, with the 32x32
//     ue8m0 block scales applied per k-block (exactly the reference scheme).
//   * The fp4 EXPERT GEMMs are NOT in this file. They live in
//     kernels/cuda/dsv41_experts_mxf4.cu, which implements
//     `dsv41_expert_gate_up_fp4` / `dsv41_expert_down_fp4` with the tcgen05
//     `kind::mxf4.block_scale.scale_vec::2X` instruction (the only fp4
//     tensor-core entry on this part; its scale type is ue8m0, exactly the
//     checkpoint's expert scale format). There is deliberately no fp8 expert
//     entry point anywhere in this ABI.
//
// The layout conventions match the release: A is [m, k] row-major (activations,
// fp8 e4m3 with per-row 32-wide scales), B is [n, k] row-major (weights, fp8
// e4m3 with per-(32-row, 32-col) ue8m0 scales).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float ue8m0_to_f(uint8_t b) {
    // float8_e8m0fnu: a pure power of two, 2^(b-127); 0xFF is NaN.
    return __int_as_float((int)((b == 0xFFu ? 0x7FC00000u : ((uint32_t)b) << 23)));
}

__device__ __forceinline__ float e4m3_to_f(uint8_t b) {
    // sign(1) exp(4) mantissa(3), bias 7; subnormals (exp 0) are m * 2^-9.
    const uint32_t s = (b & 0x80u) ? 0x80000000u : 0u;
    const uint32_t e = (b >> 3) & 0x0Fu;
    const uint32_t m = b & 0x07u;
    if (e == 0) {
        const float v = (float)m * (1.0f / 512.0f);
        return s ? -v : v;
    }
    const float v = (1.0f + (float)m * 0.125f) * exp2f((float)((int)e - 7));
    return s ? -v : v;
}

// The e2m1 code table (convert.py FP4_TABLE).
__device__ __constant__ float kFp4Table[16] = {
    0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
    0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float e2m1_to_f(uint8_t code) {
    return kFp4Table[code & 0x0Fu];
}

// The reference's power-of-two scale: 2^ceil(log2(amax / maxv)).
__device__ __forceinline__ float fast_round_scale(float amax, float max_inv) {
    const uint32_t bits = __float_as_uint(amax * max_inv);
    const int exp = (int)((bits >> 23) & 0xFFu);
    const uint32_t man = bits & 0x7FFFFFu;
    const int e = exp - 127 + (man != 0 ? 1 : 0);
    return __int_as_float((e + 127) << 23);
}

// ---------------------------------------------------------------------- quant

// Block-wise activation quantisation. block = 128 (window KV) or 32/16
// (compressed KV / indexer). `round_scale` picks the power-of-two scale.
template <int FP4>
__global__ void quant_kernel(const float* __restrict__ x, uint8_t* __restrict__ y,
                             float* __restrict__ scale, int rows, int cols, int block,
                             int round_scale) {
    const int nb = cols / block;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per block
    if (idx >= rows * nb) return;
    const int r = idx / nb, b = idx % nb;
    const float* src = x + (size_t)r * cols + (size_t)b * block;
    // amax over the block (the reference uses a whole block per group)
    float amax = 0.f;
    for (int i = threadIdx.y; i < block; i += blockDim.y) amax = fmaxf(amax, fabsf(src[i]));
    // warp/block reduction
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    __shared__ float samax;
    if (threadIdx.y == 0 && (threadIdx.x & 31) == 0) samax = amax;
    __syncthreads();
    amax = samax;
    const float maxv = FP4 ? 6.0f : 448.0f;
    const float sc = round_scale ? fmaxf(fast_round_scale(amax, 1.0f / maxv), 1e-30f)
                                 : fmaxf(amax / maxv, 1e-30f);
    if (threadIdx.x == 0 && threadIdx.y == 0) scale[idx] = sc;
    const float inv = 1.0f / sc;
    for (int i = threadIdx.y; i < block; i += blockDim.y) {
        float v = src[i] * inv;
        if (FP4) {
            v = fminf(fmaxf(v, -6.0f), 6.0f);
            // nearest code in the e2m1 table (magnitudes are the low 8 codes)
            uint8_t best = 0;
            float bd = 1e30f;
            const float a = fabsf(v);
            const float mags[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                const float d = fabsf(a - mags[c]);
                if (d < bd) { bd = d; best = (uint8_t)c; }
            }
            best |= (v < 0.f) ? 0x8u : 0u;
            y[(size_t)r * cols + (size_t)b * block + i] = best;  // nibble (packed by the caller)
        } else {
            const __nv_fp8_e4m3 f = __nv_fp8_e4m3(fminf(fmaxf(v, -448.0f), 448.0f));
            y[(size_t)r * cols + (size_t)b * block + i] = *(const uint8_t*)&f;
        }
    }
}

// Pack two e2m1 nibbles per byte (low nibble = even element, as in convert.py).
__global__ void fp4_pack_kernel(const uint8_t* __restrict__ nib, uint8_t* __restrict__ packed,
                                size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 2 >= n) return;
    const uint8_t lo = nib[i * 2];
    const uint8_t hi = (i * 2 + 1 < n) ? nib[i * 2 + 1] : 0u;
    packed[i] = (uint8_t)((lo & 0x0Fu) | (hi << 4));
}

// ------------------------------------------------------- dense fp8 MMA GEMM

// out[m, n] = a[m, k] . w[n, k]^T, fp8 e4m3 both sides.
//
// One warp per 16x16 output tile (two n-tiles of 8). Each step consumes exactly
// one 32-wide k block, which is also exactly one scale block, so the scales are
// applied per k-block into a running fp32 accumulator -- the reference scheme.
// NOTE on scale types: the ACTIVATION scales are f32 power-of-two values (the
// reference's `scales_a` is fp32; `act_quant` with round_scale returns
// 2^ceil(log2(amax/448)) as a float), while the WEIGHT scales are ue8m0 bytes
// straight out of the checkpoint. Mixing the two up silently zeroes the output,
// which is what tests_dsv41_gemm_fp8.cu guards against.
__global__ void gemm_fp8_kernel(const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
                                const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_scale,
                                const float* __restrict__ bias, float* __restrict__ out, int m,
                                int n, int k) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // block tile: 16 rows x (4 warps * 16) columns
    const int m0 = blockIdx.y * 16;
    const int n0 = blockIdx.x * 64 + warp * 16;
    if (m0 >= m) return;
    const int gid = lane >> 2, tg = lane & 3;  // row group, thread in group

    extern __shared__ uint8_t sa[];  // [16][k] e4m3 tile
    for (int i = threadIdx.x; i < 16 * k / 4; i += blockDim.x) {
        const int r = i / (k / 4), c = (i % (k / 4)) * 4;
        if (m0 + r < m) *(int*)&sa[r * k + c] = *(const int*)&a[(size_t)(m0 + r) * k + c];
    }
    __syncthreads();

    // NOTE: the warp owns 16 columns = TWO 8-wide n-tiles, so each n-tile needs
    // its own accumulator (a single acc[4] would sum the two tiles together and
    // the second tile's columns would never be written).
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const int nb_k = k >> 5;
    for (int kb = 0; kb < nb_k; kb++) {
        // A fragment: 4 regs, 4 e4m3 each
        uint32_t af[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = gid + 8 * (i & 1);
            const int c = tg * 4 + 16 * (i >> 1);
            af[i] = *(const uint32_t*)&sa[r * k + kb * 32 + c];
        }
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            // Clamp to the last valid row/column: when `n` is not a multiple of
            // the 64-column block tile the tail warp/n-tile computes garbage
            // columns (dropped in the epilogue) and must not read out of
            // bounds. Real model shapes are multiples of 64, but relying on
            // that is a bug waiting to happen (found by
            // tests_dsv41_gemm_fp8.cu with n=96).
            const int ncol = min(n0 + nt * 8 + gid, n - 1);       // B row (read)
            const int scol = min(n0 + nt * 8 + tg * 2, n - 1);    // C columns (scale)
            uint32_t bf[2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int kk = kb * 32 + tg * 4 + 16 * i;
                bf[i] = *(const uint32_t*)&w[(size_t)ncol * k + kk];
            }
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
            // per-k-block scales: activation per row, weight per (n/32, k/32)
            const int mr0 = min(m0 + gid, m - 1), mr1 = min(m0 + gid + 8, m - 1);
            const float sar0 = a_scale[(size_t)mr0 * nb_k + kb];
            const float sar1 = a_scale[(size_t)mr1 * nb_k + kb];
            const float sb0 = ue8m0_to_f(w_scale[(size_t)(scol >> 5) * nb_k + kb]);
            const float sb1 = ue8m0_to_f(w_scale[(size_t)(min(scol + 1, n - 1) >> 5) * nb_k + kb]);
            acc[nt][0] += d[0] * sar0 * sb0;  // (gid, 2*tg)     of n-tile nt
            acc[nt][1] += d[1] * sar0 * sb1;  // (gid, 2*tg+1)
            acc[nt][2] += d[2] * sar1 * sb0;  // (gid+8, 2*tg)
            acc[nt][3] += d[3] * sar1 * sb1;
        }
    }
    // epilogue: both n-tiles, C fragment rows gid / gid+8, columns 2*tg / 2*tg+1
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        const int col = n0 + nt * 8 + tg * 2;
        const float b0 = (bias && col < n) ? bias[col] : 0.f;
        const float b1 = (bias && col + 1 < n) ? bias[col + 1] : 0.f;
        if (m0 + gid < m) {
            if (col < n) out[(size_t)(m0 + gid) * n + col] = acc[nt][0] + b0;
            if (col + 1 < n) out[(size_t)(m0 + gid) * n + col + 1] = acc[nt][1] + b1;
        }
        if (m0 + gid + 8 < m) {
            if (col < n) out[(size_t)(m0 + gid + 8) * n + col] = acc[nt][2] + b0;
            if (col + 1 < n) out[(size_t)(m0 + gid + 8) * n + col + 1] = acc[nt][3] + b1;
        }
    }
}

// ------------------------------------------------------------------ engram

__global__ void engram_hash_kernel(const int32_t* __restrict__ token_map,
                                   int64_t* __restrict__ cache, const int64_t* __restrict__ primes,
                                   const int64_t* __restrict__ offsets,
                                   const int64_t* __restrict__ multipliers,
                                   const int32_t* __restrict__ input_ids,
                                   const uint8_t* __restrict__ mask, int64_t* __restrict__ out,
                                   int batch_row, int seqlen, int max_seq, int start_pos,
                                   int n_layers, int max_ngram, int n_heads, int64_t pad_id) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seqlen) return;
    const int pos = start_pos + s;
    // refresh the compressed-token cache (image spans -> DEAD)
    int64_t comp = token_map[input_ids[s]];
    if (mask && !mask[s]) comp = -1;
    cache[(size_t)batch_row * max_seq + pos] = comp;
    __syncthreads();
    const int n_cols = (max_ngram - 1) * n_heads;
    const int64_t base = 0;  // this rank's table base (row offset added by the loader)
    for (int li = 0; li < n_layers; li++) {
        const int64_t* mult = multipliers + (size_t)li * max_ngram;
        int64_t tokens[8];
        bool blocked = false;
        for (int sh = 0; sh < max_ngram; sh++) {
            const int p = max(pos - sh, 0);
            const int64_t src = cache[(size_t)batch_row * max_seq + p];
            blocked = blocked || (pos < sh) || (src == -1);
            tokens[sh] = blocked ? pad_id : src;
        }
        int64_t rolling = tokens[0] * mult[0];
        for (int i = 1; i < max_ngram; i++) {
            rolling ^= tokens[i] * mult[i];
            for (int h = 0; h < n_heads; h++) {
                const int col = (i - 1) * n_heads + h;
                const int64_t lm = primes[(size_t)li * n_cols + col];
                const int64_t off = offsets[(size_t)li * n_cols + col];
                int64_t v = rolling % lm;
                if (v < 0) v += lm;
                out[((size_t)s * n_layers + li) * n_cols + col] = v + off + base;
            }
        }
    }
}

// Gather `n_cols` rows per token from the rank's fp8 table shard, dequantise
// with the per-row 32-wide ue8m0 scales and write [rows, n_cols*head_dim].
__global__ void engram_gather_kernel(const uint8_t* __restrict__ table,
                                     const uint8_t* __restrict__ table_scale,
                                     const int64_t* __restrict__ hash_ids, float* __restrict__ out,
                                     int rows, int n_cols, int head_dim, int64_t part_start,
                                     int64_t part_rows) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)rows * n_cols * head_dim;
    if (i >= total) return;
    const int j = (int)(i % head_dim);
    const size_t row_id = i / head_dim;  // (token, col)
    const int64_t id = hash_ids[row_id];
    if (id < part_start || id >= part_start + part_rows) {
        out[i] = 0.f;  // owned by another rank; the caller all-reduces
        return;
    }
    const int64_t local = id - part_start;
    out[i] = e4m3_to_f(table[(size_t)local * head_dim + j]) *
             ue8m0_to_f(table_scale[(size_t)local * (head_dim / 32) + j / 32]);
}

// --------------------------------------------------------- sparse attention

// q[b,m,h,d] x kv[b,n,d] (ONE KV head) with idxs[b,m,topk]; online softmax with
// the sink folded into the denominator after the loop.
#define kMaxPer 8   // d <= 512 with blockDim >= 64; the launcher uses 128 (per = 4)
__global__ void sparse_attn_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                                   const float* __restrict__ sink, const int32_t* __restrict__ idxs,
                                   float* __restrict__ out, int b, int m, int h, int d,
                                   const int* __restrict__ clen, int window, int index_topk,
                                   float scale) {
    // n and topk used to be host arguments derived from this layer's compress_len;
    // they change per step, so a captured graph would freeze them. The counter now
    // lives on the device (the compressor's commit kernel advances it).
    const int n = window + *clen;
    const int topk = window + ((*clen < index_topk) ? *clen : index_topk);
    const int row = blockIdx.x;  // flattened (b, m)
    if (row >= b * m) return;
    const int bb = row / m, mm = row % m;
    for (int hh = blockIdx.y; hh < h; hh += gridDim.y) {
        const float* qr = q + ((size_t)(bb * m + mm) * h + hh) * d;
        // acc is a COMPILE-TIME-sized array (d <= 512 and blockDim is 128 here, so
        // per-thread <= 4) indexed by the loop counter, NOT by the element index.
        // The element-to-thread mapping and the per-thread summation order are
        // unchanged - thread tid still sums elements tid, tid+blockDim, ... in that
        // order - so the partials are identical and the output stays bit-identical.
        // The old `float acc[512]` indexed by `c` was dynamically indexed and spilled
        // to local memory, which is the largest recoverable cost in this kernel.
        const int per = (d + (int)blockDim.x - 1) / (int)blockDim.x;
        float acc[kMaxPer];
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i) acc[i] = 0.f;
        float smax = -1e30f, se = 0.f;
        for (int t = 0; t < topk; t++) {
            const int idx = idxs[(size_t)(bb * m + mm) * topk + t];
            if (idx < 0) continue;
            const float* kr = kv + ((size_t)bb * n + idx) * d;
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPer; ++i) {
                const int c = threadIdx.x + i * (int)blockDim.x;
                if (c < d) dot += qr[c] * kr[c];
            }
            // Two-stage dot reduction. The warp shuffle only covers 32 lanes, so
            // on its own it drops every warp but the first: with blockDim=128 and
            // d=512 each thread sums 4 elements, and taking only warp 0's partial
            // made the score ~1/4 of its true value — which collapsed the softmax
            // weight (0.95 -> 0.67 for a single visible key) and therefore the
            // whole attention output. The per-warp sums must be combined across
            // the block. (Same class of bug as the hc_mixes ss reduction.)
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            __shared__ float sdot;
            __shared__ float wpart[32];
            const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
            if (lane == 0) wpart[wid] = dot;
            __syncthreads();
            if (threadIdx.x == 0) {
                const int nw = (blockDim.x + 31) >> 5;
                float s = 0.f;
                for (int w = 0; w < nw; w++) s += wpart[w];
                sdot = s;
            }
            __syncthreads();
            dot = sdot * scale;
            const float nm = fmaxf(smax, dot);
            const float corr = expf(smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPer; ++i) {
                const int c = threadIdx.x + i * (int)blockDim.x;
                if (c < d) acc[i] = acc[i] * corr + e * kr[c];
            }
            se = se * corr + e;
            smax = nm;
            __syncthreads();
        }
        se += expf(sink[hh] - smax);
        float* orow = out + ((size_t)(bb * m + mm) * h + hh) * d;
        for (int c = threadIdx.x; c < d; c += blockDim.x) {
            // c == threadIdx.x + i * blockDim.x, so the accumulator slot is
            // (c - threadIdx.x) / blockDim.x (NOT c / blockDim.x: that is only
            // correct for thread 0).
            const int i = (c - (int)threadIdx.x) / (int)blockDim.x;
            orow[c] = (se > 0.f) ? acc[i] / se : 0.f;
        }
    }
}

// Flash-decode split of sparse attention: each WARP owns a subset of the topk
// slots (t = wid, wid + nwarp, ...), covers the whole head dim with its 32 lanes,
// and runs its own online softmax. The per-slot dot is therefore a pure warp
// shuffle - ZERO barriers inside the loop, where the sequential version paid two
// __syncthreads per slot (~1000 over topk=512) at an occupancy of eight blocks,
// which the isolated repro measured at 330 us, flat in n. The block's partials are
// merged once at the end (the only barriers). Sizing: the launcher pins blockDim
// to 128 (4 warps), so sh_acc is [4][512] = 8 KB.
#define kMaxPerW 16   // d <= 512 over 32 lanes
__global__ void sparse_attn_warp_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                                        const float* __restrict__ sink,
                                        const int32_t* __restrict__ idxs, float* __restrict__ out,
                                        int b, int m, int h, int d,
                                        const int* __restrict__ clen, int window, int index_topk,
                                        float scale) {
    const int n = window + *clen;
    const int topk = window + ((*clen < index_topk) ? *clen : index_topk);
    const int row = blockIdx.x;  // flattened (b, m)
    if (row >= b * m) return;
    const int bb = row / m, mm = row % m;
    for (int hh = blockIdx.y; hh < h; hh += gridDim.y) {
        const float* qr = q + ((size_t)(bb * m + mm) * h + hh) * d;
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        const int nwarp = (int)blockDim.x >> 5;
        float my_acc[kMaxPerW];
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) my_acc[i] = 0.f;
        float my_smax = -1e30f, my_se = 0.f;
        for (int t = wid; t < topk; t += nwarp) {
            const int idx = idxs[(size_t)(bb * m + mm) * topk + t];
            if (idx < 0) continue;
            const float* kr = kv + ((size_t)bb * n + idx) * d;
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) dot += qr[c] * kr[c];
            }
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            dot *= scale;
            const float nm = fmaxf(my_smax, dot);
            const float corr = expf(my_smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) my_acc[i] = my_acc[i] * corr + e * kr[c];
            }
            my_se = my_se * corr + e;
            my_smax = nm;
        }
        // ---- the ONLY barriers in this kernel: merge the warps' partials ----
        __shared__ float sh_smax[32], sh_se[32];
        __shared__ float sh_acc[4][512];
        if (lane == 0) {
            sh_smax[wid] = my_smax;
            sh_se[wid] = my_se;
        }
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) sh_acc[wid][c] = my_acc[i];
        }
        __syncthreads();
        // smax over slots only (the sink is folded into se below, not into smax,
        // matching the sequential version's post-loop `se += expf(sink - smax)`).
        float smax = -1e30f;
        for (int w = 0; w < nwarp; ++w) smax = fmaxf(smax, sh_smax[w]);
        float wsc[4];
        for (int w = 0; w < nwarp && w < 4; ++w) wsc[w] = expf(sh_smax[w] - smax);
        float se = 0.f;
        for (int w = 0; w < nwarp; ++w) se += sh_se[w] * wsc[w < 4 ? w : 0];
        se += expf(sink[hh] - smax);
        float* orow = out + ((size_t)(bb * m + mm) * h + hh) * d;
        for (int c = threadIdx.x; c < d; c += blockDim.x) {
            float a = 0.f;
            for (int w = 0; w < nwarp && w < 4; ++w) a += sh_acc[w][c] * wsc[w];
            orow[c] = (se > 0.f) ? a / se : 0.f;
        }
        __syncthreads();  // sh_* reuse safety across the hh loop
    }
}

// ---------------------------------------------------------------------------
// Split sparse attention (DSV41_ATTN_SPLIT). The warp version runs one block per
// (token, head) - eight blocks on this model's decode - which leaves 95 percent
// of the machine idle and pays for it in latency: 26 us per layer, forty layers,
// one millisecond a step. This shape splits the KEY range across `C` blocks per
// head, each computing an online-softmax partial (max, sum, d-wide PV
// accumulator) into a fixed device scratch, and a second kernel merges the
// partials in ascending chunk order - a deterministic combine, so the result
// differs from the single-block version only in the last bits of the summation
// grouping. The sink fold and the normalisation live in the merge, exactly where
// the warp version had them.
#define kAttnMaxC 16
#define kAttnMaxBM 8
#define kAttnMaxH 64
#define kAttnStride (2 + 512)
__device__ float g_attn_part[kAttnMaxBM][kAttnMaxH][kAttnMaxC][kAttnStride];

// grid (C, b*m, h): block (ck, row, hh) owns keys [topk*ck/C, topk*(ck+1)/C).
__global__ void sparse_attn_split_kernel(const float* __restrict__ q,
                                         const float* __restrict__ kv,
                                         const int32_t* __restrict__ idxs, int b, int m, int h,
                                         int d, const int* __restrict__ clen, int window,
                                         int index_topk, float scale, int C) {
    const int topk = window + ((*clen < index_topk) ? *clen : index_topk);
    const int ck = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= b * m || ck >= C) return;
    const int hh = blockIdx.z;
    if (hh >= h) return;
    const int bb = row / m, mm = row % m;
    const float* qr = q + ((size_t)(bb * m + mm) * h + hh) * d;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nwarp = (int)blockDim.x >> 5;
    const int lo = (int)((long long)topk * ck / C);
    const int hi = (int)((long long)topk * (ck + 1) / C);
    const int32_t* row_idx = idxs + (size_t)(bb * m + mm) * topk;
    float my_acc[kMaxPerW];
#pragma unroll
    for (int i = 0; i < kMaxPerW; ++i) my_acc[i] = 0.f;
    float my_smax = -1e30f, my_se = 0.f;
    for (int t = lo + wid; t < hi; t += nwarp) {
        const int idx = row_idx[t];
        if (idx < 0) continue;
        const float* kr = kv + ((size_t)bb * (window + *clen) + idx) * d;
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) dot += qr[c] * kr[c];
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        dot *= scale;
        const float nm = fmaxf(my_smax, dot);
        const float corr = expf(my_smax - nm);
        const float e = expf(dot - nm);
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) my_acc[i] = my_acc[i] * corr + e * kr[c];
        }
        my_se = my_se * corr + e;
        my_smax = nm;
    }
    // merge the block's warps exactly as the warp version does, then publish the
    // chunk partial (the sink fold is the merge kernel's job)
    __shared__ float sh_smax[32], sh_se[32];
    __shared__ float sh_acc[4][512];
    if (lane == 0) {
        sh_smax[wid] = my_smax;
        sh_se[wid] = my_se;
    }
#pragma unroll
    for (int i = 0; i < kMaxPerW; ++i) {
        const int c = lane + i * 32;
        if (c < d) sh_acc[wid][c] = my_acc[i];
    }
    __syncthreads();
    float smax = -1e30f;
    for (int w = 0; w < nwarp; ++w) smax = fmaxf(smax, sh_smax[w]);
    float wsc[4];
    for (int w = 0; w < nwarp && w < 4; ++w) wsc[w] = expf(sh_smax[w] - smax);
    float* P = &g_attn_part[row][hh][ck][0];
    if (threadIdx.x == 0) {
        float se = 0.f;
        for (int w = 0; w < nwarp; ++w) se += sh_se[w] * wsc[w < 4 ? w : 0];
        P[0] = smax;
        P[1] = se;
    }
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float a = 0.f;
        for (int w = 0; w < nwarp && w < 4; ++w) a += sh_acc[w][c] * wsc[w];
        P[2 + c] = a;
    }
}

// grid (b*m, h): combine the C chunk partials in ascending chunk order, fold the
// sink into se (not into smax, as the warp version does) and normalise.
__global__ void sparse_attn_merge_kernel(const float* __restrict__ sink, float* __restrict__ out,
                                         int b, int m, int h, int d, int C) {
    const int row = blockIdx.x;
    if (row >= b * m) return;
    const int hh = blockIdx.y;
    if (hh >= h) return;
    const float* P = &g_attn_part[row][hh][0][0];
    float smax = -1e30f;
    for (int ck = 0; ck < C; ++ck) smax = fmaxf(smax, P[(size_t)ck * kAttnStride]);
    float se = 0.f;
    for (int ck = 0; ck < C; ++ck)
        se += P[(size_t)ck * kAttnStride + 1] * expf(P[(size_t)ck * kAttnStride] - smax);
    se += expf(sink[hh] - smax);
    float* orow = out + ((size_t)row * h + hh) * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float a = 0.f;
        for (int ck = 0; ck < C; ++ck)
            a += P[(size_t)ck * kAttnStride + 2 + c] * expf(P[(size_t)ck * kAttnStride] - smax);
        orow[c] = (se > 0.f) ? a / se : 0.f;
    }
}

// ------------------------------------------------------------ indexer / rope

__global__ void candidate_blocks_kernel(const float* __restrict__ logits,
                                        const int32_t* __restrict__ compress_lens,
                                        uint8_t* __restrict__ mask, int rows, int n_pos,
                                        int topk_blocks, int block_size) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int nb = (n_pos + block_size - 1) / block_size;
    __shared__ float sc[512];
    for (int blk = threadIdx.x; blk < nb; blk += blockDim.x) {
        float mx = -1e30f;
        for (int i = 0; i < block_size; i++) {
            const int p = blk * block_size + i;
            if (p < n_pos) mx = fmaxf(mx, logits[(size_t)r * n_pos + p]);
        }
        sc[blk] = mx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int last = (compress_lens[r] - 1) / block_size;
        if (last >= 0 && last < nb) sc[last] = 1e30f;  // pin the newest block
        for (int k = 0; k < topk_blocks; k++) {
            int best = -1;
            float bv = -1e30f;
            for (int blk = 0; blk < nb; blk++) {
                if (sc[blk] > bv) { bv = sc[blk]; best = blk; }
            }
            if (best < 0 || !(bv > -1e30f)) break;
            for (int i = 0; i < block_size; i++) {
                const int p = best * block_size + i;
                if (p < n_pos) mask[(size_t)r * n_pos + p] = 1;
            }
            sc[best] = -1e30f;
        }
    }
}

__global__ void rope_precompute_kernel(float* __restrict__ cos, float* __restrict__ sin, int dim,
                                       int seqlen, int original_seq_len, float base, float factor,
                                       float beta_fast, float beta_slow) {
    const int half = dim / 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < half; i += gridDim.x * blockDim.x) {
        float f = powf(base, -2.0f * (float)i / (float)dim);
        if (original_seq_len > 0) {
            // YaRN: the rotation band between beta_fast and beta_slow is divided
            // by `factor`, faded in with a linear ramp.
            const float cd = (float)dim * logf((float)original_seq_len /
                                               (32.0f * 2.0f * 3.14159265358979f)) /
                             (2.0f * logf(base));
            const float cd2 = (float)dim * logf((float)original_seq_len /
                                                (1.0f * 2.0f * 3.14159265358979f)) /
                              (2.0f * logf(base));
            const float lo = fmaxf(cd, 0.f), hi = fminf(cd2, (float)(dim - 1));
            const float t = fminf(fmaxf(((float)i - lo) / fmaxf(hi - lo, 1e-3f), 0.f), 1.f);
            const float smooth = 1.0f - t;
            f = f / factor * (1.0f - smooth) + f * smooth;
        }
        (void)beta_fast;
        (void)beta_slow;
        for (int tt = 0; tt < seqlen; tt++) {
            const float a = (float)tt * f;
            cos[(size_t)tt * half + i] = cosf(a);
            sin[(size_t)tt * half + i] = sinf(a);
        }
    }
}

// The position is now read from DEVICE memory: (*base) * mul + off. That covers
// both call shapes without a host value - the plain rope (base = the position
// counter, mul = 1, off = 0) and the compressor-group rope (base = this layer's
// latent count, mul = ratio, off = -ratio, i.e. (clen - 1) * ratio). This is what
// makes the call capturable in a graph: the counter advances on the device (the
// argmax does it) and nothing about the launch arguments changes per step.
__global__ void apply_rope_kernel(float* __restrict__ x, const float* __restrict__ cos,
                                  const float* __restrict__ sin, int rows, int row_len, int dim,
                                  int half, const int* __restrict__ base, int mul, int off,
                                  int step, int inverse) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = (*base) * mul + off + r * step;
    float* row = x + (size_t)r * row_len + (row_len - dim);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float c = cos[(size_t)t * half + i];
        const float s = sin[(size_t)t * half + i] * (inverse ? -1.f : 1.f);
        const float x0 = row[2 * i], x1 = row[2 * i + 1];
        row[2 * i] = x0 * c - x1 * s;
        row[2 * i + 1] = x0 * s + x1 * c;
    }
}

// ------------------------------------------------------------------ hc / moe

// hc_mixes: one projection per token of the flattened hc*dim stream, then the
// sigmoid/sinkhorn split. Grid: one block per token.
__global__ void hc_mixes_kernel(const float* __restrict__ x, const float* __restrict__ hc_fn,
                                const float* __restrict__ hc_scale,
                                const float* __restrict__ hc_base, float* __restrict__ pre,
                                float* __restrict__ post, float* __restrict__ comb, int rows,
                                int hc_dim, int hc, int sinkhorn_iters, float eps,
                                bool hc_mixes_acc4) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int mix = hc * (2 + hc);
    extern __shared__ float sm[];
    float* mixes = sm;             // [mix]
    float* cm = sm + mix;          // [hc*hc]
    const float* xr = x + (size_t)r * hc_dim;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hc_dim; c += blockDim.x) ss += xr[c] * xr[c];
    // warp reduction, then ACROSS WARPS. The earlier version stopped at the
    // warp level and stored only warp 0's partial sum, so `ss` was 1/nwarps of
    // the true sum of squares and `inv` was sqrt(nwarps) = 2.83x too large at
    // blockDim=256. That scaled every hc coefficient wrong, which mis-mixed the
    // whole residual stream — the model's output had no relation to its input.
    // (Same class as the documented GLM '8-warp reduce' bug.)
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
    __shared__ float sss;
    __shared__ float wpart[32];
    const int nwarp = (blockDim.x + 31) >> 5;
    if ((threadIdx.x & 31) == 0) wpart[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < nwarp) ? wpart[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (threadIdx.x == 0) sss = v;
    }
    __syncthreads();
    const float inv = rsqrtf(sss / (float)hc_dim + eps);
    // One WARP per projection row with a shuffle reduction. The previous shape
    // handed row m to thread m, so only `mix` (=24) lanes had work and each ran
    // a serial 20480-iteration dependent-load loop: measured 1.06ms per call,
    // 51% of the whole decode's GPU time (nsys cuda_gpu_kern_sum), at ~1% of
    // memory bandwidth. Coalescing across a warp and splitting the dot by lane
    // is the same fix that cured gdn_chunk and sparse_attn.
    {
        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        const int nwarp = (blockDim.x + 31) >> 5;
        for (int m = wid; m < mix; m += nwarp) {
            const float* wr = hc_fn + (size_t)m * hc_dim;
            // Single accumulate chain = the original, verified behaviour. A
            // four-accumulator unroll was tried together with the wide block above
            // and must be retested alone (HC_MIXES_ACC4=1) before being believed.
            float acc = 0.f;
            if (hc_mixes_acc4) {
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                int c = lane;
                for (; c + 96 < hc_dim; c += 128) {
                    a0 += wr[c] * xr[c];
                    a1 += wr[c + 32] * xr[c + 32];
                    a2 += wr[c + 64] * xr[c + 64];
                    a3 += wr[c + 96] * xr[c + 96];
                }
                for (; c < hc_dim; c += 32) a0 += wr[c] * xr[c];
                acc = (a0 + a1) + (a2 + a3);
            } else {
                // 16 bytes per thread per iteration instead of 4. A phase-by-phase
                // shutdown sweep puts 78 percent of this kernel (39.7 of 50.8 us) in
                // this dot product, and it moves 1.5 MB of weights at only 38 GB/s -
                // one warp per projection row, about 1.7 GB/s each, which is the
                // signature of too few bytes in flight per warp rather than of an
                // arithmetic or SM-count limit (spreading the 24 rows over 24 or even
                // 192 blocks changed nothing). Widening the loads is the cheapest
                // multiplier of in-flight bytes, the same move that fixed gdn_chunk,
                // sparse_attn and gemv_fp8. Four accumulators alone are NOT the answer:
                // the existing acc4 branch measures 52.1 against 49.3 us, because four
                // separate 4-byte streams add addresses without adding bytes per load.
                // The summation order changes, so this is not bit-identical - validate
                // with the four prompts and DSV41_TOKTRACE.
                const float4* wr4 = reinterpret_cast<const float4*>(wr);
                const float4* xr4 = reinterpret_cast<const float4*>(xr);
                const int n4 = hc_dim >> 2;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f;
                int c = lane;
                for (; c + 64 < n4; c += 96) {
                    const float4 w0 = wr4[c], w1 = wr4[c + 32], w2 = wr4[c + 64];
                    const float4 v0 = xr4[c], v1 = xr4[c + 32], v2 = xr4[c + 64];
                    a0 += w0.x * v0.x + w0.y * v0.y + w0.z * v0.z + w0.w * v0.w;
                    a1 += w1.x * v1.x + w1.y * v1.y + w1.z * v1.z + w1.w * v1.w;
                    a2 += w2.x * v2.x + w2.y * v2.y + w2.z * v2.z + w2.w * v2.w;
                }
                for (; c < n4; c += 32) {
                    const float4 w = wr4[c], v = xr4[c];
                    a0 += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
                }
                // scalar tail for shapes whose hc_dim is not a multiple of four
                for (int k = (n4 << 2) + lane; k < hc_dim; k += 32) a0 += wr[k] * xr[k];
                acc = (a0 + a1) + a2;
            }
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
            }
            if (lane == 0) mixes[m] = acc * inv;
        }
    }
    __syncthreads();
    if (threadIdx.x < (unsigned)hc) {
        const int j = threadIdx.x;
        pre[(size_t)r * hc + j] =
            (1.f / (1.f + expf(-(mixes[j] * hc_scale[0] + hc_base[j])))) + eps;
        post[(size_t)r * hc + j] =
            2.f / (1.f + expf(-(mixes[hc + j] * hc_scale[1] + hc_base[hc + j])));
    }
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
        const int j = jk / hc, k = jk % hc;
        cm[jk] = mixes[2 * hc + j * hc + k] * hc_scale[2] + hc_base[2 * hc + j * hc + k];
    }
    __syncthreads();
    // Sinkhorn normalisation to doubly-stochastic: each row AND column of comb
    // sums to ~1, which is what keeps the residual's energy from growing through
    // the hc_post expansion. Without it comb is unbounded (mixes*scale+base) and
    // the h stream grows exponentially — rms 0.5 at L0, 3e11 at L5, inf/NaN by L20.
    // Matches hc_split_sinkhorn in the reference (kernel.py:407).
    // step 1: row softmax + eps
    // The sinkhorn lives in ONE WARP's registers. hc is 4, so comb is sixteen
    // values - lanes 0..15, lane l holding cm[l] with l = j*hc + k - and the
    // row/column reductions are xor butterflies (offsets 1,2 within a row group;
    // 4,8 across rows). Measured motivation: an isolated repro shows this kernel
    // costs a FLAT 58 us per call from rows=1 to rows=64, i.e. it is all fixed
    // overhead, and it runs 803 times per decode step (28.1 percent of the GPU
    // time). The only structure that can absorb a fixed 58 us is the twenty-pass
    // normalisation: block-wide barriers in the original, and in the single-thread
    // version dynamically indexed local arrays that spill. Registers plus shuffles
    // need neither.
    __syncthreads();
    {
        const int hh = hc * hc;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (warp == 0) {
            float c = (lane < hh) ? cm[lane] : 0.f;
            // row softmax: max then sum over the hc lanes of this row group
            float mx = c;
            for (int off = 1; off < hc; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
            c = expf(c - mx);
            float rs = c;
            for (int off = 1; off < hc; off <<= 1) rs += __shfl_xor_sync(0xFFFFFFFFu, rs, off);
            c = c / rs + eps;
            for (int it = 0; it < sinkhorn_iters; ++it) {
                if (it > 0) {
                    float s = c;
                    for (int off = 1; off < hc; off <<= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
                    c = c / (s + eps);
                }
                float t = c;
                for (int off = hc; off < hh; off <<= 1) t += __shfl_xor_sync(0xFFFFFFFFu, t, off);
                c = c / (t + eps);
            }
            if (lane < hh) cm[lane] = c;
        }
    }
    __syncthreads();
    {
        const int hh = hc * hc;
        for (int jk = threadIdx.x; jk < hh; jk += blockDim.x) comb[(size_t)r * hh + jk] = cm[jk];
    }
}

__global__ void moe_route_kernel(const float* __restrict__ x, const uint8_t* __restrict__ gate_w,
                                 const uint8_t* __restrict__ gate_w_scale,
                                 const float* __restrict__ gate_bias, float* __restrict__ weights,
                                 int32_t* __restrict__ indices, int32_t* __restrict__ hist,
                                 int rows, int dim, int n_experts, int topk, float gate_temp,
                                 int norm_topk_prob, float route_scale, int score_func) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sc[];
    const int nb_k = dim >> 5;
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        float acc = 0.f;
        for (int kb = 0; kb < nb_k; kb++) {
            for (int c = 0; c < 32; c += blockDim.x) {
                const int kk = kb * 32 + c;
                if (kk < dim)
                    acc += e4m3_to_f(gate_w[(size_t)e * dim + kk]) *
                           ue8m0_to_f(gate_w_scale[(size_t)(e >> 5) * nb_k + kb]) *
                           x[(size_t)r * dim + kk];
            }
        }
        acc /= gate_temp;
        sc[e] = (score_func == 2) ? sqrtf(log1pf(expf(acc))) : (1.f / (1.f + expf(-acc)));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int t = 0; t < topk; t++) {
            int best = -1;
            float bv = -1e30f;
            for (int e = 0; e < n_experts; e++) {
                const float s = sc[e] + gate_bias[e];
                if (s > bv) { bv = s; best = e; }
            }
            if (best < 0) break;
            weights[(size_t)r * topk + t] = sc[best];
            indices[(size_t)r * topk + t] = best;
            if (hist) atomicAdd(&hist[best], 1);
            sc[best] = -1e30f;
            // keep the raw score for the normalisation below
            sc[n_experts + t] = 0.f;
        }
        float sum = 0.f;
        for (int t = 0; t < topk; t++) sum += weights[(size_t)r * topk + t];
        for (int t = 0; t < topk; t++) {
            float wv = weights[(size_t)r * topk + t];
            if (norm_topk_prob && topk > 1) wv /= (sum + 1e-20f);
            weights[(size_t)r * topk + t] = wv * route_scale;
        }
    }
}


// --------------------------------------------------------- indexer (level 2)
// One CTA per (b, m) row of `indexer_topk` (ops.rs golden):
//   score[p] = sum_h relu(q[b,m,h,:] . k[b,p,:]) * weights[b,m,h]
//              * softmax_scale * head_scale
//   p >= compress_lens[m] -> -inf ; candidate-masked -> -inf
// then top-`cols` with the golden's stable tie rule (equal scores keep the
// lower position), the picked positions re-sorted ascending, and `p + offset`
// written for reachable positions / -1 otherwise.
//
// The candidates are visited in fixed-size CHUNKS (kIndexerChunk), so the dynamic
// shared memory is a function of compile-time constants only - never of the
// candidate count. That count is a PER-STEP value and a CUDA graph capture freezes
// the launch (its arguments AND the launcher's shared-memory size), so a score
// array sized from it either degraded silently or indexed past a smaller
// allocation on every replay. Each chunk's own stable top-k is merged into a
// running top-`cols` set, which leaves the scan bound free to be the live device
// counter for any value.
constexpr int kIndexerChunk = 4096;  // candidates per chunk (shared memory is sized from this)

// Selection key: higher score first, and on an EQUAL score the lower position
// first (the golden's stable rule). Returns true when (av, ap) is the worse key.
__device__ __forceinline__ bool idx_key_worse(float av, int ap, float bv, int bp) {
    return (av < bv) || (av == bv && ap > bp);
}

__global__ void indexer_topk_kernel(const float* __restrict__ q, const float* __restrict__ ik,
                                    const float* __restrict__ w, const uint8_t* __restrict__ cand,
                                    const int32_t* __restrict__ lens, int32_t* __restrict__ out,
                                    int m, int nh, int hd, int n_pos, int topk, int offset,
                                    float softmax_scale, float head_scale, int uses_cand) {
    // n_pos arrives as a launch argument, and it is a PER-STEP value (the number of
    // committed latents). A CUDA graph capture freezes launch arguments, so every
    // replay would apply the capture step's bound and the compressed-slot
    // retrieval would silently degrade as the generation grows. The device counter
    // is already passed in as `lens`; prefer it whenever it is available and
    // non-zero (falling back keeps the pre-prefill / uninitialised case working).
    // This is safe for ANY bound only because the shared memory below is a function
    // of the chunk constant alone - it does not grow with the candidate count.
    if (lens != nullptr && *lens > 0) n_pos = *lens;
    extern __shared__ float smem[];
    const int cols = topk < n_pos ? topk : n_pos;  // out slots (sparse_attn's stride)
    const int cap = topk;                          // array capacity: cols <= topk
    // The chunk's candidates live as (score, position) pairs, sorted in place by a
    // bitonic network: the value at [2i], the position as float bits at [2i+1].
    // The iterative block argmax this replaces ran one full-chunk scan, two
    // barriers and a serial cross-warp merge PER PICK - hundreds of passes per
    // call, the better part of 150 us. The sort produces the same descending
    // (score, position-ascending) total order the argmax picked in, so the
    // selection itself is unchanged.
    float* s_pair = smem;                             // [2*kIndexerChunk] (score, position)
    float* s_rv = s_pair + 2 * kIndexerChunk;         // [cap] running values (key order)
    int* s_rp = (int*)(s_rv + cap);                   // [cap] running positions
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int mm = blockIdx.x, bb = blockIdx.y;
    const size_t row = (size_t)bb * m + mm;
    __shared__ int s_nr;  // running-set size (thread 0 owns it; the output reads it)

    int cl = n_pos;
    if (lens != nullptr) cl = lens[mm];
    if (cl > n_pos) cl = n_pos;
    const float* qrow = q + row * (size_t)nh * hd;
    if (tid == 0) s_nr = 0;

    for (int base = 0; base < n_pos; base += kIndexerChunk) {
        const int len = (n_pos - base < kIndexerChunk) ? (n_pos - base) : kIndexerChunk;
        // Pad to the network's power of two. The padding pairs carry -INFINITY
        // with a position above the chunk's real ones, so they sort behind every
        // real entry and the first `k <= len` slots of the sorted array are always
        // real candidates.
        int P = 1;
        while (P < len) P <<= 1;
        for (int i = len + tid; i < P; i += nthr) {
            s_pair[2 * i] = -INFINITY;
            s_pair[2 * i + 1] = __int_as_float(base + i);
        }
        __syncthreads();

        // ---- scores for this chunk
        for (int i = tid; i < len; i += nthr) {
            const int p = base + i;
            const float* krow = ik + ((size_t)bb * n_pos + p) * hd;
            float acc = 0.f;
            for (int h = 0; h < nh; ++h) {
                const float* qh = qrow + (size_t)h * hd;
                float dot = 0.f;
                for (int c = 0; c < hd; ++c) dot += qh[c] * krow[c];
                acc += fmaxf(dot, 0.f) * w[row * (size_t)nh + h];
            }
            float sv = acc * softmax_scale * head_scale;
            if (p >= cl) sv = -INFINITY;
            if (uses_cand && cand != nullptr && !cand[row * (size_t)n_pos + p]) sv = -INFINITY;
            s_pair[2 * i] = sv;
            s_pair[2 * i + 1] = __int_as_float(p);
        }
        __syncthreads();

        // ---- this chunk's stable top-`k`: bitonic sort of the (score, position)
        // pairs. The order it produces - score descending, position ascending on
        // ties - is the same total order the iterative argmax used to pick in, so
        // the selection is identical; the sort is one pass with log^2(P) barriers
        // against k passes with 2k barriers and a serial cross-warp merge each.
        // After the sort, s_pair[2*j] is the j-th best score and
        // __float_as_int(s_pair[2*j+1]) its position, for j < k <= len.
        const int k = cols < len ? cols : len;
        for (int k2 = 2; k2 <= P; k2 <<= 1) {
            for (int j2 = k2 >> 1; j2 > 0; j2 >>= 1) {
                __syncthreads();
                for (int i = tid; i < P; i += nthr) {
                    const int l = i ^ j2;
                    if (l <= i) continue;
                    // the final pass sorts the whole array best-first (descending)
                    const bool up = (i & k2) == 0;
                    const float va = s_pair[2 * i], vb = s_pair[2 * l];
                    const int ia = __float_as_int(s_pair[2 * i + 1]);
                    const int ib = __float_as_int(s_pair[2 * l + 1]);
                    // "a first" = a has the higher score, or an equal score at the
                    // lower position - idx_key_worse's rule
                    const bool a_first = (va > vb) || (va == vb && ia < ib);
                    if (up ? !a_first : a_first) {
                        s_pair[2 * i] = vb;
                        s_pair[2 * i + 1] = __int_as_float(ib);
                        s_pair[2 * l] = va;
                        s_pair[2 * l + 1] = __int_as_float(ia);
                    }
                }
            }
        }
        __syncthreads();

        // ---- merge this chunk's picks into the running top-`cols` (thread 0)
        if (tid == 0) {
            // Both lists are in key order (best first), so a backwards two-pointer
            // merge over the WORST ends fills the new running set from its tail, and
            // the elements past `cols` are dropped first. A pick takes a slot only on
            // a STRICTLY better key, so an equal score keeps the incumbent - which is
            // the earlier, lower position: chunks run in ascending position order, and
            // the argmax above already yields equal scores in ascending index order.
            // The merge is in place: the write index d equals i + j + 1 with both read
            // indices included, so d >= i (the running index) and a slot is only
            // overwritten after it has been read.
            const int t = s_nr + k;
            const int keep = t < cols ? t : cols;
            int i = s_nr - 1, j = k - 1;
            for (int drop = t - keep; drop > 0; --drop) {
                if (i < 0) --j;
                else if (j < 0) --i;
                else if (idx_key_worse(s_rv[i], s_rp[i], s_pair[2 * j], __float_as_int(s_pair[2 * j + 1]))) --i;
                else --j;
            }
            for (int d = keep - 1; d >= 0; --d) {
                if (i < 0) {
                    const int p = __float_as_int(s_pair[2 * j + 1]);
                    s_rv[d] = s_pair[2 * j];
                    s_rp[d] = p;
                    --j;
                } else if (j < 0) {
                    s_rv[d] = s_rv[i];
                    s_rp[d] = s_rp[i];
                    --i;
                } else if (idx_key_worse(s_rv[i], s_rp[i], s_pair[2 * j], __float_as_int(s_pair[2 * j + 1]))) {
                    s_rv[d] = s_rv[i];
                    s_rp[d] = s_rp[i];
                    --i;
                } else {
                    const int p = __float_as_int(s_pair[2 * j + 1]);
                    s_rv[d] = s_pair[2 * j];
                    s_rp[d] = p;
                    --j;
                }
            }
            s_nr = keep;
        }
        __syncthreads();
    }

    // ---- output: the golden sorts the SELECTION by position ascending, so rank the
    // running picks by position (they are unique) and write `p + offset` for
    // reachable positions / -1 otherwise. The chunk loop covers all of `*lens`, so
    // every candidate the direct path used to see is still here.
    for (int i = tid; i < cols; i += nthr) {
        if (i >= s_nr) {
            out[row * (size_t)cols + i] = -1;
            continue;
        }
        const int pi = s_rp[i];
        int rank = 0;
        for (int j = 0; j < s_nr; ++j)
            if (s_rp[j] < pi) ++rank;
        out[row * (size_t)cols + rank] = (pi < cl) ? (pi + offset) : -1;
    }
}

// ---------------------------------------------------------------- compressor
// f32 -> e4m3 with one power-of-two ue8m0 scale per (row, 32-column block):
// the activation layout the fp8 GEMM path (gemm_fp8_kernel) consumes.
// One warp per 32-element block, one scale byte per block.
__global__ void quant_e4m3_pow2_kernel(const float* __restrict__ x, uint8_t* __restrict__ y,
                                       float* __restrict__ scale, int rows, int nb) {
    const int unit = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (unit >= rows * nb) return;
    const float* src = x + (size_t)unit * 32;
    float amax = fabsf(src[lane]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float sc = fmaxf(fast_round_scale(amax, 1.0f / 448.0f), 1e-30f);
    if (lane == 0) scale[unit] = sc;   // f32, matching the GEMM's `scales_a`
    const float q = fminf(fmaxf(src[lane] / sc, -448.f), 448.f);
    y[unit * 32 + lane] = (uint8_t)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

// Carries the current step's projections into the compressor state:
//   start_pos == 0 (prefill): the trailing `seqlen % ratio` rows -> slots 0..rem-1
//   start_pos  > 0 (decode) : the single row -> slot (start_pos % ratio)
__global__ void compressor_state_kernel(const float* __restrict__ kvp,
                                        const float* __restrict__ scp,
                                        float* __restrict__ state_kv,
                                        float* __restrict__ state_score, int b, int seqlen,
                                        int hd, int ratio, const int* __restrict__ pos_ctr) {
    const int start_pos = *pos_ctr;  // device-side: graph-capturable
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    if (start_pos == 0) {
        const int rem = seqlen % ratio;
        if (rem <= 0) return;
        const int cut = seqlen - rem;
        const size_t total = (size_t)b * rem * hd;
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
            const int c = (int)(i % hd);
            const size_t rt = i / hd;
            const int bbi = (int)(rt / rem), t = (int)(rt % rem);
            const size_t src = ((size_t)bbi * seqlen + cut + t) * hd + c;
            const size_t dst = ((size_t)bbi * ratio + t) * hd + c;
            state_kv[dst] = kvp[src];
            state_score[dst] = scp[src];
        }
    } else {
        const int slot = start_pos % ratio;
        const size_t total = (size_t)b * hd;
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
            const int c = (int)(i % hd);
            const int bbi = (int)(i / hd);
            const size_t src = (size_t)bbi * hd + c;
            const size_t dst = ((size_t)bbi * ratio + slot) * hd + c;
            state_kv[dst] = kvp[src];
            state_score[dst] = scp[src];
        }
    }
}

// Pooling + RMSNorm epilogue of the compressor (ops.rs compressor_forward):
//   mode 0 (ratio == 1)      : latents[row] = rmsnorm(kvp[row]) for all rows
//   mode 1 (ratio > 1, prefill): one latent per completed group of `ratio` rows,
//                              pooled with a per-channel softmax over the group
//   mode 2 (ratio > 1, decode) : pool the `ratio` state slots (only when the step
//                              completes a group; `out_rows_val` carries that)
// `*out_rows` is written unconditionally by block 0 so the caller always sees
// the decision (0 = nothing written).
__global__ void compressor_pool_kernel(const float* __restrict__ kvp,
                                       const float* __restrict__ scp,
                                       const float* __restrict__ norm_w,
                                       const float* __restrict__ state_kv,
                                       const float* __restrict__ state_score,
                                       float* __restrict__ latents, int32_t* __restrict__ out_rows,
                                       int mode, int grid_n, int b, int seqlen, int hd, int ratio,
                                       const int* __restrict__ pos_ctr, float eps) {
    // out_rows is derived from the DEVICE position counter, not passed in: it
    // changes per step (one latent per `ratio` positions) and a captured graph
    // freezes launch arguments, so the decision has to live on the device.
    const int out_rows_val = ((*pos_ctr + 1) % ratio == 0) ? 1 : 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) *out_rows = out_rows_val;
    if ((int)blockIdx.x >= grid_n) return;
    // an unfinished decode group updates the state but writes no latent
    if (mode == 2 && out_rows_val == 0) return;
    const int tid = threadIdx.x, nthr = blockDim.x;
    __shared__ float s_red[32];

    // channels this thread owns (strided; inactive threads contribute 0 to ss)
    float yv[16];
    int cn[16];
    int nown = 0;
    for (int c = tid; c < hd; c += nthr) {
        if (nown >= 16) break;
        cn[nown++] = c;
    }

    int out_row = -1;
    int bb = 0;
    int ngroups = 0;

    if (mode == 0) {  // ratio == 1: plain projection row
        out_row = (int)blockIdx.x;
        for (int i = 0; i < nown; ++i) yv[i] = kvp[(size_t)out_row * hd + cn[i]];
    } else if (mode == 1) {  // prefill: pool group g of batch bb
        ngroups = grid_n / b;
        bb = (int)blockIdx.x / ngroups;
        const int g = (int)blockIdx.x % ngroups;
        out_row = bb * ngroups + g;
        for (int i = 0; i < nown; ++i) {
            const int c = cn[i];
            float mx = -INFINITY;
            float sv[32];
            float vv[32];
            const int rr = ratio < 32 ? ratio : 32;
            for (int r = 0; r < rr; ++r) {
                const size_t rw = ((size_t)bb * seqlen + (size_t)g * ratio + r) * hd + c;
                sv[r] = scp[rw];
                vv[r] = kvp[rw];
                mx = fmaxf(mx, sv[r]);
            }
            float den = 0.f, acc = 0.f;
            for (int r = 0; r < rr; ++r) {
                const float e = expf(sv[r] - mx);
                den += e;
                acc += e * vv[r];
            }
            yv[i] = den > 0.f ? acc / den : 0.f;
        }
    } else {  // decode: pool the carried state slots
        for (int i = 0; i < nown; ++i) {
            const int c = cn[i];
            float mx = -INFINITY;
            float sv[32];
            float vv[32];
            const int rr = ratio < 32 ? ratio : 32;
            for (int r = 0; r < rr; ++r) {
                sv[r] = state_score[((size_t)bb * ratio + r) * hd + c];
                vv[r] = state_kv[((size_t)bb * ratio + r) * hd + c];
                mx = fmaxf(mx, sv[r]);
            }
            float den = 0.f, acc = 0.f;
            for (int r = 0; r < rr; ++r) {
                const float e = expf(sv[r] - mx);
                den += e;
                acc += e * vv[r];
            }
            yv[i] = den > 0.f ? acc / den : 0.f;
        }
        out_row = bb;
    }

    // RMSNorm over the pooled row (weights + eps, exactly as the golden)
    {
        float ss = 0.f;
        for (int i = 0; i < nown; ++i) ss += yv[i] * yv[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xffffffffu, ss, off);
        const int warp = tid >> 5, lane = tid & 31, nw = nthr >> 5;
        if (lane == 0) s_red[warp] = ss;
        __syncthreads();
        float total = 0.f;
        for (int widx = 0; widx < nw; ++widx) total += s_red[widx];
        const float inv = rsqrtf(total / (float)hd + eps);
        for (int i = 0; i < nown; ++i) {
            const int c = cn[i];
            latents[(size_t)out_row * hd + c] = yv[i] * inv * norm_w[c];
        }
    }
    (void)pos_ctr;
}

}  // namespace

#define DSV41_LAUNCH_CHECK()                     \
    do {                                         \
        cudaError_t e = cudaGetLastError();      \
        if (e != cudaSuccess) return (int)e;     \
    } while (0)

extern "C" int dsv41_quant_fp8(const float* x, uint8_t* y, float* scale, int rows, int cols,
                               int block, int round_scale, cudaStream_t s) {
    if (rows <= 0 || cols % block != 0) return (int)cudaErrorInvalidValue;
    dim3 grid((rows * (cols / block) + 255) / 256), blk(1, 256);
    // one thread-block per (row, scale-block) group; the y dim carries the lanes
    const int nb = cols / block;
    blk = dim3(1, (block < 256 ? block : 256));
    grid = dim3(rows * nb);
    quant_kernel<0><<<grid, blk, 0, s>>>(x, y, scale, rows, cols, block, round_scale);
    return (int)cudaGetLastError();
}

// Per-device fp4 quantise scratch (see the comment in dsv41_quant_fp4).
static uint8_t* g_q4nib[64] = {nullptr};
static size_t   g_q4nib_cap[64] = {0};

extern "C" int dsv41_quant_fp4(const float* x, uint8_t* y, float* scale, int rows, int cols,
                               int block, int round_scale, cudaStream_t s) {
    if (rows <= 0 || cols % block != 0) return (int)cudaErrorInvalidValue;
    const int nb = cols / block;
    // Cached per-device scratch: a TP8 process has one context per rank thread,
    // so the cache is indexed by device. Growing it is a synchronising
    // cudaMalloc, which is (a) illegal inside a stream capture and (b) a hot-path
    // hazard because quant_fp4 runs several times per layer. The warm-up step
    // sizes every entry, so the capture path never allocates.
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return (int)cudaErrorInvalidDevice;
    const size_t need = (size_t)rows * cols;
    if (g_q4nib_cap[dev] < need) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone)
            return (int)cudaErrorStreamCaptureUnsupported;
        if (g_q4nib[dev]) {
            cudaFree(g_q4nib[dev]);
            g_q4nib[dev] = nullptr;
            g_q4nib_cap[dev] = 0;
        }
        if (cudaMalloc(&g_q4nib[dev], need) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
        g_q4nib_cap[dev] = need;
    }
    uint8_t* nib = g_q4nib[dev];
    dim3 blk(1, (block < 256 ? block : 256));
    quant_kernel<1><<<dim3(rows * nb), blk, 0, s>>>(x, nib, scale, rows, cols, block, round_scale);
    const size_t n = (size_t)rows * cols;
    fp4_pack_kernel<<<(unsigned)((n / 2 + 255) / 256), 256, 0, s>>>(nib, y, n);
    return (int)cudaGetLastError();
}

// M=1 fp8 GEMV. gemm_fp8_kernel's tile is 16 rows x (4 warps * 16) columns with
// smem = 16*k bytes, so at decode's M=1 a typical n=4096 projection launches
// grid = (64, 1) with 128 threads: 64 warps over 148 SMs, about 0.43 per SM, and
// 15/16 of the A tile is wasted (64.8 us per call, 19 percent of decode, nsys).
// This is the same shape of problem the expert fp4 GEMM had, and it takes the same
// cure: one warp per output row, no tile, no tmem.
// Formats read off gemm_fp8_kernel and matched exactly:
//   a       [1, k]        e4m3, one byte per value
//   a_scale [1, k/32]     f32
//   w       [n, k]        e4m3, one byte per value
//   w_scale [n/32, k/32]  e8m0  (block 32x32)
//   out     [1, n]        = a @ w^T + bias
// Read once: this launcher runs ~290 times per step, and a per-call getenv on
// the hot path is the same slip the other gates avoid.
static const int g_gemv_fp8_mode = [] {
    const char* e = getenv("DSV41_GEMV_FP8_MODE");
    if (e != nullptr) return atoi(e);              // 0 scalar, 1 vectorised, 3 staged+ordered
    const char* v = getenv("DSV41_GEMV_FP8_VEC");
    if (v != nullptr && v[0] == '0') return 0;     // the earlier opt-out still honoured
    // The staged modes are the default because they won on both axes at once:
    // mode 3 measured 20.04 against the vectorised 21.52 ms in one session, same
    // binary, and it is bit-identical to the scalar path so the stray token the
    // reordering used to flip is gone; mode 4 additionally stages the activation
    // once per block instead of letting all eight warps re-read it, which
    // measured 19.24 against mode 3's 19.96 ms with the same clean text. The wide
    // loads are what made the vectorised branch fast, and staging keeps them
    // without paying its order change.
    return 4;
}();

__global__ void gemm_fp8_gemv_kernel(const uint8_t* __restrict__ a,
                                     const float* __restrict__ a_scale,
                                     const uint8_t* __restrict__ w,
                                     const uint8_t* __restrict__ w_scale,
                                     const float* __restrict__ bias,
                                     float* __restrict__ out, int n, int k, int vec,
                                     // Second family: rows [n1, n1+n2) of the same input.
                                     // A single-family launch passes n1 == n and leaves
                                     // the family-2 pointers untouched, so every row maps
                                     // to family 1 and the behaviour is exactly the old
                                     // kernel's. Both families share the staged activation,
                                     // which is the point: two projections that read the
                                     // same vector become one launch and one staging.
                                     const uint8_t* __restrict__ w2 = nullptr,
                                     const uint8_t* __restrict__ w2_scale = nullptr,
                                     const float* __restrict__ bias2 = nullptr,
                                     float* __restrict__ out2 = nullptr, int n1 = 0) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = (blockDim.x + 31) >> 5;
    const int nb_k = k >> 5;   // k-blocks of 32
    // Four fp8 per lane per iteration instead of one, behind DSV41_GEMV_FP8_VEC
    // until the text check passes. The warp still covers the row, but 32 lanes *
    // 4 bytes = 128 elements = four 32-element scale blocks, so one uint32 load
    // and one scale pair replace four byte loads and four scale lookups. Because
    // 4*lane is 4-aligned inside a 32-byte scale block each lane's four bytes
    // always sit inside ONE block: lanes 0-7 cover block 0, lanes 8-15 block 1,
    // which is what (lane >> 3) selects. The per-element product keeps the old
    // shape; only the order in which a lane visits its elements changes, so the
    // sum differs in the last bits and the text must be re-checked.
    const int n_vec = nb_k >> 2;            // full 128-element iterations
    const int tail0 = n_vec << 2;           // first block of the scalar tail
    const int blk_off = lane >> 3;          // which of the four blocks this lane owns
    const int byte_off = lane << 2;         // byte offset inside the 128-element group
    // The activation row is the same for every output row, so it is staged once per
    // block rather than re-read by each of the `nwarps` warps - eight byte-loads per
    // element where one will do, and sixteen-byte loads instead of one byte. The
    // barrier sits outside the row loop on purpose: that loop advances by warp, so a
    // barrier inside it would be reached a different number of times per warp.
    extern __shared__ uint8_t s_w[];
    uint8_t* s_a = s_w + (size_t)nwarps * (size_t)k;
    if (vec == 4) {
        const int n16a = k >> 4;
        for (int i = threadIdx.x; i < n16a; i += blockDim.x) {
            *reinterpret_cast<uint4*>(s_a + (i << 4)) =
                *reinterpret_cast<const uint4*>(a + (i << 4));
        }
        for (int i = (n16a << 4) + threadIdx.x; i < k; i += blockDim.x) s_a[i] = a[i];
        __syncthreads();
    }
    for (int row = blockIdx.x * nwarps + warp; row < n; row += gridDim.x * nwarps) {
        // Family dispatch: rows below n1 belong to the first projection, the rest
        // to the second. Each row is still one warp walking the same lane order,
        // so both dots are bit-identical to two separate launches.
        int rrow;
        const uint8_t* wr;
        const uint8_t* wsr;
        const float* bias_;
        float* out_;
        if (row < n1) {
            rrow = row;
            wr = w + (size_t)rrow * k;
            wsr = w_scale + (size_t)(rrow >> 5) * nb_k;
            bias_ = bias;
            out_ = out;
        } else {
            rrow = row - n1;
            wr = w2 + (size_t)rrow * k;
            wsr = w2_scale + (size_t)(rrow >> 5) * nb_k;
            bias_ = bias2;
            out_ = out2;
        }
        float acc = 0.f;
        if (vec >= 3) {
            // Order-preserving staging. The row is fetched with sixteen-byte loads
            // into this warp's slice of shared memory and then consumed exactly the
            // way the scalar loop consumes it - element kb*32 + lane, kb ascending -
            // so the summation order, and therefore the sum, is bit-identical to the
            // scalar path. That is the whole point: the vectorised branch reorders a
            // lane's elements from a thirty-two stride to four consecutive values,
            // correct arithmetic that nonetheless flips near-boundary logits.
            // Mode 4 additionally reads the activation out of the block-wide copy
            // staged above rather than from global memory.
            const uint8_t* ap = (vec == 4) ? s_a : a;
            uint8_t* row_s = s_w + (size_t)warp * (size_t)k;
            // Sixteen bytes per lane per iteration. The first cut counted
            // thirty-two-byte k-blocks while copying one uint4 each, so only the
            // first half of every block was staged, the odd halves stayed
            // uninitialised, and the result was a five percent "speed-up" on a page
            // of garbage. Count sixteen-byte units, and cover a k that is not a
            // multiple of sixteen with a byte tail.
            const int n16 = k >> 4;
            for (int i = lane; i < n16; i += 32) {
                *reinterpret_cast<uint4*>(row_s + (i << 4)) =
                    *reinterpret_cast<const uint4*>(wr + (i << 4));
            }
            for (int i = (n16 << 4) + lane; i < k; i += 32) row_s[i] = wr[i];
            __syncwarp();
            for (int kb = 0; kb < nb_k; ++kb) {
                const float sb = ue8m0_to_f(wsr[kb]);
                const float sa = a_scale[kb];    // m == 1
                const int j = kb * 32 + lane;
                acc += e4m3_to_f(ap[j]) * sa * (e4m3_to_f(row_s[j]) * sb);
            }
            __syncwarp();
        } else if (vec) {
            for (int g = 0; g < n_vec; ++g) {
                const int kb = (g << 2) + blk_off;
                const float sa = a_scale[kb];    // m == 1
                const float sb = ue8m0_to_f(wsr[kb]);
                const int j = (g << 7) + byte_off;
                const uint32_t av = *reinterpret_cast<const uint32_t*>(a + j);
                const uint32_t wv = *reinterpret_cast<const uint32_t*>(wr + j);
                acc += e4m3_to_f((uint8_t)(av & 0xFFu)) * sa * (e4m3_to_f((uint8_t)(wv & 0xFFu)) * sb);
                acc += e4m3_to_f((uint8_t)((av >> 8) & 0xFFu)) * sa * (e4m3_to_f((uint8_t)((wv >> 8) & 0xFFu)) * sb);
                acc += e4m3_to_f((uint8_t)((av >> 16) & 0xFFu)) * sa * (e4m3_to_f((uint8_t)((wv >> 16) & 0xFFu)) * sb);
                acc += e4m3_to_f((uint8_t)(av >> 24)) * sa * (e4m3_to_f((uint8_t)(wv >> 24)) * sb);
            }
            for (int kb = tail0; kb < nb_k; ++kb) {
                const float sb = ue8m0_to_f(wsr[kb]);
                const float sa = a_scale[kb];    // m == 1
                const int j = kb * 32 + lane;
                acc += e4m3_to_f(a[j]) * sa * (e4m3_to_f(wr[j]) * sb);
            }
        } else {
            for (int kb = 0; kb < nb_k; ++kb) {
                const float sb = ue8m0_to_f(wsr[kb]);
                const float sa = a_scale[kb];    // m == 1
                const int j = kb * 32 + lane;
                acc += e4m3_to_f(a[j]) * sa * (e4m3_to_f(wr[j]) * sb);
            }
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) out_[rrow] = acc + (bias_ ? bias_[rrow] : 0.f);
    }
}

extern "C" int dsv41_gemm_fp8_mx(const uint8_t* a, const float* a_scale, const uint8_t* w,
                                 const uint8_t* w_scale, const float* bias, float* out, int m,
                                 int n, int k, cudaStream_t s) {
    if (m <= 0 || n <= 0 || k <= 0 || (k & 31) || (k & 3)) return (int)cudaErrorInvalidValue;
    // The A tile lives in shared memory: 16 rows x k bytes. At the model's real
    // k (5120) that is 80 KB, well past the 48 KB static limit, so the kernel
    // needs the opt-in dynamic size (Blackwell allows ~227 KB/block). Setting
    // the attribute every call is cheap and avoids the per-device pitfall (the
    // attribute is per-context, and a TP8 process has one context per rank).
    const int smem = 16 * k;
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
        if (e != cudaSuccess) return (int)e;
    }
    // M=1 (decode): skip the 16-row tile entirely - it wastes 15/16 of itself and
    // its 16*k bytes of shared memory cap the occupancy. One warp per output row.
    if (m == 1 && getenv("DSV41_NO_GEMV_FP8") == nullptr) {
        // Eight rows per block was the first shape. With k = 1536 that is 12 KB of
        // weights per block and 2048 short-lived blocks for the q_b projection, so
        // the per-block setup is a real share of the call. The count is adjustable
        // to test that against the opposite risk (fewer, larger blocks and less
        // occupancy); the per-row accumulation does not depend on it.
        static const int g_warps = [] {
            const char* e = getenv("DSV41_GEMV_FP8_WARPS");
            // Four rows per block measured 15.91 against 16.14 ms for eight, same
            // session, same binary, text unchanged: halving the rows doubles the
            // block count and the warps in flight, which is what this
            // latency-bound family actually needs. Two was no better than four and
            // stages the activation twice as often, so four it is.
            if (e == nullptr) return 4;
            const int v = atoi(e);
            return (v >= 1 && v <= 32) ? v : 4;
        }();
        const int warps = g_warps;
        const int blocks = (n + warps - 1) / warps;
        const size_t gsmem = (g_gemv_fp8_mode == 3)   ? (size_t)warps * (size_t)k
                             : (g_gemv_fp8_mode == 4) ? (size_t)(warps + 1) * (size_t)k
                                                      : (size_t)0;
        if (gsmem > 48 * 1024) {
            cudaError_t e = cudaFuncSetAttribute(
                gemm_fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
            if (e != cudaSuccess) return (int)e;
        }
        gemm_fp8_gemv_kernel<<<blocks, warps * 32, gsmem, s>>>(a, a_scale, w, w_scale, bias, out, n,
                                                              k, g_gemv_fp8_mode, nullptr, nullptr,
                                                              nullptr, nullptr, n);
        return (int)cudaGetLastError();
    }
    dim3 grid((n + 63) / 64, (m + 15) / 16);
    gemm_fp8_kernel<<<grid, 128, smem, s>>>(a, a_scale, w, w_scale, bias, out, m, n, k);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_engram_hash(const int32_t* token_map, int64_t* cache, const int64_t* primes,
                                 const int64_t* offsets, const int64_t* multipliers,
                                 const int32_t* input_ids, const uint8_t* mask, int64_t* out,
                                 int batch_row, int seqlen, int max_seq, int start_pos,
                                 int n_layers, int max_ngram, int n_heads, int64_t pad_id,
                                 cudaStream_t s) {
    engram_hash_kernel<<<(unsigned)((seqlen + 127) / 128), 128, 0, s>>>(
        token_map, cache, primes, offsets, multipliers, input_ids, mask, out, batch_row, seqlen,
        max_seq, start_pos, n_layers, max_ngram, n_heads, pad_id);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_engram_gather(const uint8_t* table, const uint8_t* table_scale,
                                   const int64_t* hash_ids, float* out, int rows, int n_cols,
                                   int head_dim, int64_t part_start, int64_t part_rows,
                                   cudaStream_t s) {
    const size_t total = (size_t)rows * n_cols * head_dim;
    engram_gather_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(
        table, table_scale, hash_ids, out, rows, n_cols, head_dim, part_start, part_rows);
    return (int)cudaGetLastError();
}

// Stable argmax over f32 logits: ties resolve to the LOWEST index, matching the
// host loop's strictly-greater scan. f32 becomes a sortable u32 key (negatives
// inverted, positives get the sign bit), packed as (key << 32) | (0xFFFFFFFF - idx)
// so among equal keys the SMALLER index has the LARGER packed value; one block of
// 1024 threads reduces with shuffles and thread 0 writes the winner. No atomics, no
// scratch, fully deterministic. One block is deliberate: 129280 elements is ~127
// per thread, all coalesced, and it removes any cross-block combine. This is the
// GLM HEAD_DEV pattern - the sampled token never leaves the device except as the
// single 4-byte read the host needs for EOS and printing.
__global__ void argmax_kernel(const float* __restrict__ v, int* __restrict__ out, int n,
                              int* __restrict__ pos_ctr) {
    unsigned long long my = 0ull;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned int bits = __float_as_uint(v[i]);
        const unsigned int key = (bits >> 31) ? ~bits : (bits | 0x80000000u);
        const unsigned long long pk =
            ((unsigned long long)key << 32) | (0xFFFFFFFFu - (unsigned)i);
        if (pk > my) my = pk;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_down_sync(0xFFFFFFFFu, my, off);
        if (o > my) my = o;
    }
    __shared__ unsigned long long wb[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (int)blockDim.x >> 5;
    if (lane == 0) wb[wid] = my;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long m = 0ull;
        for (int w = 0; w < nw; ++w)
            if (wb[w] > m) m = wb[w];
        *out = (int)(0xFFFFFFFFu - (unsigned)(m & 0xFFFFFFFFu));
        // the argmax is the LAST kernel of the step: this is where the
        // device position counter advances, so every kernel of the NEXT
        // step (the engram hash, the window indices, the compressor) sees
        // pos + 1 while every kernel of THIS step saw a stable position.
        if (pos_ctr != nullptr) *pos_ctr = *pos_ctr + 1;
    }
}

extern "C" int dsv41_argmax(const float* v, int* out, int n, int* pos_ctr, cudaStream_t s) {
    if (n <= 0) return (int)cudaErrorInvalidValue;
    argmax_kernel<<<1, 1024, 0, s>>>(v, out, n, pos_ctr);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_sparse_attn(const float* q, const float* kv, const float* sink,
                                 const int32_t* idxs, float* out, int b, int m, int h, int d,
                                 const int* clen, int window, int index_topk, float scale,
                                 cudaStream_t s) {
    if (d > 512) return (int)cudaErrorInvalidValue;  // the accumulator is d-wide per thread group
    // Flash-decode split by default; DSV41_ATTN_SEQ=1 restores the sequential
    // version for A/B. Cached in a static: this runs per attention call, and a
    // per-call getenv is exactly the hot-path slip this project has been bitten by.
    static const bool seq = [] { return getenv("DSV41_ATTN_SEQ") != nullptr; }();
    // Chunked key split (DSV41_ATTN_SPLIT=C): off until the A/B passes, then the
    // default becomes the winning chunk count.
    static const int g_attn_split = [] {
        const char* e = getenv("DSV41_ATTN_SPLIT");
        if (e == nullptr) return 0;
        return atoi(e);
    }();
    dim3 grid(b * m, h);
    if (!seq) {
        if (g_attn_split > 0 && g_attn_split <= kAttnMaxC && b * m <= kAttnMaxBM &&
            h <= kAttnMaxH) {
            sparse_attn_split_kernel<<<dim3((unsigned)g_attn_split, (unsigned)(b * m),
                                            (unsigned)h), 128, 0, s>>>(
                q, kv, idxs, b, m, h, d, clen, window, index_topk, scale, g_attn_split);
            cudaError_t e2 = cudaGetLastError();
            if (e2 != cudaSuccess) return (int)e2;
            sparse_attn_merge_kernel<<<dim3((unsigned)(b * m), (unsigned)h), 128, 0, s>>>(
                sink, out, b, m, h, d, g_attn_split);
            return (int)cudaGetLastError();
        }
        sparse_attn_warp_kernel<<<grid, 128, 0, s>>>(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk,
                                                     scale);
    } else {
        sparse_attn_kernel<<<grid, 128, 0, s>>>(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale);
    }
    return (int)cudaGetLastError();
}

extern "C" int dsv41_indexer_topk(const float* q, const float* index_k, const float* weights,
                                  const uint8_t* candidates, const int32_t* compress_lens,
                                  int32_t* out, int b, int m, int nh, int hd, int n_pos, int topk,
                                  int offset, float softmax_scale, float head_scale,
                                  int uses_candidates, cudaStream_t s) {
    if (b <= 0 || m <= 0 || nh <= 0 || hd <= 0 || topk <= 0) return (int)cudaErrorInvalidValue;
    if (n_pos <= 0) return (int)cudaSuccess;  // nothing to select from
    // Dynamic shared memory from COMPILE-TIME constants only: the sort's
    // (score, position) pairs - two floats per candidate - plus the running
    // top-`cols` set (a value and a position per slot). n_pos is deliberately
    // absent - it is a PER-STEP value and a CUDA graph capture freezes the launch
    // (arguments AND this size), so a size derived from it was applied unchanged
    // to every replay while the kernel scanned the live (growing) device count:
    // either a stale bound (silent candidate loss) or an index past a smaller
    // allocation. The kernel's chunk loop makes any scan bound safe, so it now
    // scans `*lens` for any value.
    const size_t smem = (size_t)kIndexerChunk * (2 * sizeof(float)) +
                        (size_t)topk * (sizeof(float) + sizeof(int)) + 64;
    // Stay inside the 47 KiB USABLE default (48 KiB nominal minus the 1 KiB the driver
    // reserves per block), so no cudaFuncSetAttribute opt-in is ever needed. A `topk`
    // too large for that budget fails the launch LOUDLY here - the previous constant
    // bound instead kept the launch valid and silently dropped the far candidates.
    if (smem > 47 * 1024) return (int)cudaErrorInvalidValue;
    dim3 grid((unsigned)m, (unsigned)b);
    indexer_topk_kernel<<<grid, 256, smem, s>>>(q, index_k, weights, candidates, compress_lens,
                                                out, m, nh, hd, n_pos, topk, offset, softmax_scale,
                                                head_scale, uses_candidates);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_candidate_blocks(const float* logits, const int32_t* compress_lens,
                                      uint8_t* mask, int rows, int n_pos, int topk_blocks,
                                      int block_size, cudaStream_t s) {
    candidate_blocks_kernel<<<rows, 128, 0, s>>>(logits, compress_lens, mask, rows, n_pos,
                                                 topk_blocks, block_size);
    return (int)cudaGetLastError();
}

// The compressor's projections use the existing fp8 MMA path: activations are
// quantised to e4m3 (per-(row, 32) power-of-two ue8m0 scales) and the fp8 GEMM
// applies the checkpoint's 32x32 weight scales per k-block. head_dim must be a
// multiple of 64 (the GEMM's column tile) and dim a multiple of 32.
static int compressor_alloc(void** p, size_t n, cudaStream_t s) {
    cudaError_t e = cudaMallocAsync(p, n, s);
    if (e == cudaSuccess) return 0;
    cudaGetLastError();
    return (int)cudaMalloc(p, n);
}
static void compressor_free(void* p, cudaStream_t s) {
    if (p == nullptr) return;
    if (cudaFreeAsync(p, s) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(p);
    }
}

// Compressor, pooling half only.
//
// The checkpoint stores the compressor's `wkv`/`wgate` in **bf16** (the release
// promotes them to fp32 at runtime for the pooling), so the projections run on
// the bf16 tensor-core path — cuBLAS from the chain — and this entry takes their
// outputs. It mirrors exactly the second half of `dsv41_compressor`: the state
// carry, the gated pool, the RMSNorm and the out_rows decision.
//
// `kvp`/`scp` are [rows, head_dim]; `out_rows` is a DEVICE pointer (a host
// pointer here is a device store to host memory — the trap the first draft of
// the chain fell into).
extern "C" int dsv41_compressor_pool(const float* kvp, const float* scp, const float* norm_w,
                                     float* state_kv, float* state_score, float* latents,
                                     int* out_rows, int b, int seqlen, int head_dim, int ratio,
                                     int start_pos, const int* pos_ctr, float eps, cudaStream_t s) {
    if (b <= 0 || seqlen <= 0 || head_dim <= 0 || ratio <= 0) return (int)cudaErrorInvalidValue;
    if (ratio > 1) {
        const size_t work = start_pos == 0 ? (size_t)b * (seqlen % ratio) * head_dim
                                           : (size_t)b * head_dim;
        const int blocks = (int)((work + 255) / 256);
        if (work > 0)
            compressor_state_kernel<<<(blocks > 0 ? blocks : 1), 256, 0, s>>>(
                kvp, scp, state_kv, state_score, b, seqlen, head_dim, ratio, pos_ctr);
    }
    // mode/grid_n come from the HOST's knowledge of prefill vs decode: they are
    // CONSTANT across every decode step (2 / b), which is what lets a graph bake
    // them in. out_rows_val is deliberately NOT passed - it flips once per `ratio`
    // steps and the kernel derives it from the device position counter.
    int mode, grid_n;
    if (ratio == 1) {
        mode = 0;
        grid_n = seqlen;
    } else if (start_pos == 0) {
        mode = 1;
        grid_n = b * (seqlen / ratio);
    } else {
        mode = 2;
        grid_n = b;
    }
    const unsigned launch = (unsigned)(grid_n > 0 ? grid_n : 1);
    compressor_pool_kernel<<<launch, 128, 0, s>>>(kvp, scp, norm_w, state_kv, state_score, latents,
                                                  out_rows, mode, grid_n, b, seqlen, head_dim, ratio,
                                                  pos_ctr, eps);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_compressor(const float* x, const uint8_t* wkv, const uint8_t* wkv_scale,
                                const uint8_t* wgate, const uint8_t* wgate_scale,
                                const float* norm_w, float* state_kv, float* state_score,
                                float* latents, int32_t* out_rows, int b, int seqlen, int dim,
                                int head_dim, int ratio, int start_pos, const int* pos_ctr,
                                float eps, cudaStream_t s) {
    if (b <= 0 || seqlen <= 0 || dim <= 0 || head_dim <= 0 || ratio <= 0)
        return (int)cudaErrorInvalidValue;
    if ((dim & 31) || (head_dim & 63)) return (int)cudaErrorInvalidValue;
    if (ratio > 1 && (wgate == nullptr || wgate_scale == nullptr || state_kv == nullptr ||
                      state_score == nullptr))
        return (int)cudaErrorInvalidValue;

    const int rows = b * seqlen;
    const int nb = dim >> 5;
    const size_t sz_xq = (size_t)rows * dim;
    const size_t sz_xs = (size_t)rows * nb * sizeof(float);   // f32 activation scales
    const size_t sz_p = (size_t)rows * head_dim * sizeof(float);
    void* xq = nullptr;
    void* xs = nullptr;
    void* kvp = nullptr;
    void* scp = nullptr;
    int rc = 0;
    rc |= compressor_alloc(&xq, sz_xq, s);
    rc |= compressor_alloc(&xs, sz_xs, s);
    rc |= compressor_alloc(&kvp, sz_p, s);
    if (ratio > 1) rc |= compressor_alloc(&scp, sz_p, s);
    if (rc != 0) {
        compressor_free(xq, s);
        compressor_free(xs, s);
        compressor_free(kvp, s);
        compressor_free(scp, s);
        return (int)cudaErrorMemoryAllocation;
    }

    // 1. activation quantisation (e4m3 + per-(row,32) power-of-two scales)
    {
        const int units = rows * nb;
        const int blocks = (units + 3) / 4;  // 4 warps per CTA, one block each
        quant_e4m3_pow2_kernel<<<blocks, 128, 0, s>>>(x, (uint8_t*)xq, (float*)xs, rows, nb);
    }
    // 2. kv / gate projections (the existing fp8 MMA path)
    {
        const int smem = 16 * dim;
        if (smem > 200 * 1024) {
            compressor_free(xq, s);
            compressor_free(xs, s);
            compressor_free(kvp, s);
            compressor_free(scp, s);
            return (int)cudaErrorInvalidValue;
        }
        cudaFuncSetAttribute(gemm_fp8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        dim3 grid((head_dim + 63) / 64, (rows + 15) / 16);
        gemm_fp8_kernel<<<grid, 128, smem, s>>>((const uint8_t*)xq, (const float*)xs, wkv,
                                                wkv_scale, nullptr, (float*)kvp, rows, head_dim,
                                                dim);
        if (ratio > 1)
            gemm_fp8_kernel<<<grid, 128, smem, s>>>((const uint8_t*)xq, (const float*)xs, wgate,
                                                    wgate_scale, nullptr, (float*)scp, rows,
                                                    head_dim, dim);
    }
    // 3. state update (trailing partial group on prefill / the current slot on decode)
    if (ratio > 1) {
        const size_t work = start_pos == 0 ? (size_t)b * (seqlen % ratio) * head_dim
                                           : (size_t)b * head_dim;
        const int blocks = (int)((work + 255) / 256);
        if (work > 0)
            compressor_state_kernel<<<(blocks > 0 ? blocks : 1), 256, 0, s>>>(
                (const float*)kvp, (const float*)scp, state_kv, state_score, b, seqlen, head_dim,
                ratio, pos_ctr);
    }
    // 4. pooling + RMSNorm + the out_rows decision
    // mode/grid_n are host-known (constant per phase, so a graph can bake them);
    // out_rows_val is NOT passed - the kernel derives it from the device counter
    int mode, grid_n;
    if (ratio == 1) {
        mode = 0;
        grid_n = rows;
    } else if (start_pos == 0) {
        mode = 1;
        grid_n = b * (seqlen / ratio);
    } else {
        mode = 2;
        grid_n = b;
    }
    {
        const unsigned launch = (unsigned)(grid_n > 0 ? grid_n : 1);
        compressor_pool_kernel<<<launch, 128, 0, s>>>(
            (const float*)kvp, (const float*)scp, norm_w, state_kv, state_score, latents, out_rows,
            mode, grid_n, b, seqlen, head_dim, ratio, pos_ctr, eps);
    }
    compressor_free(xq, s);
    compressor_free(xs, s);
    compressor_free(kvp, s);
    compressor_free(scp, s);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_rope_precompute(float* cos, float* sin, int dim, int seqlen,
                                     int original_seq_len, float base, float factor, float beta_fast,
                                     float beta_slow, cudaStream_t s) {
    rope_precompute_kernel<<<(unsigned)((dim / 2 + 127) / 128), 128, 0, s>>>(
        cos, sin, dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_apply_rope(float* x, const float* cos, const float* sin, int rows, int row_len,
                                int dim, int half, const int* base, int mul, int off, int step, int inverse,
                                cudaStream_t s) {
    apply_rope_kernel<<<rows, 128, 0, s>>>(x, cos, sin, rows, row_len, dim, half, base, mul, off, step,
                                           inverse);
    return (int)cudaGetLastError();
}

static const bool g_hc_acc4 = getenv("DSV41_HC_MIXES_ACC4") != nullptr;

// ---------------------------------------------------------------------------
// SPREAD variant of hc_mixes (DSV41_HC_MIXES_SPREAD), see the note by the kernels.
// ---------------------------------------------------------------------------
#define DSV41_HC_SPREAD_MAXR 2048
#define DSV41_HC_SPREAD_S 8          // K chunks; one block per (row, projection row, chunk)
// The split is configurable so the spread path can be tested at split=1, where a
// dot is one block's verbatim copy of the fused kernel's reduction and the result
// should be bit-identical; the shipped default stays at 8, where the partials are
// summed in a different order and the outputs no longer match bit for bit.
static const int g_hc_spread_s = [] {
    const char* e = getenv("DSV41_HC_SPREAD_S");
    if (e == nullptr) return DSV41_HC_SPREAD_S;
    const int v = atoi(e);
    return (v >= 1 && v <= DSV41_HC_SPREAD_S) ? v : DSV41_HC_SPREAD_S;
}();
__device__ float g_hc_inv[DSV41_HC_SPREAD_MAXR];
__device__ float g_hc_part[DSV41_HC_SPREAD_MAXR][64][DSV41_HC_SPREAD_S];

// Phase A: one block per token, the sum of squares -> g_hc_inv[r]. The two-stage
// reduction is copied verbatim from hc_mixes_kernel so the sum lands in the same order.
__global__ void hc_mixes_ss_kernel(const float* __restrict__ x, int rows, int hc_dim, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (size_t)r * hc_dim;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hc_dim; c += blockDim.x) ss += xr[c] * xr[c];
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
    __shared__ float wpart[32];
    const int nwarp = (blockDim.x + 31) >> 5;
    if ((threadIdx.x & 31) == 0) wpart[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < nwarp) ? wpart[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (threadIdx.x == 0) g_hc_inv[r] = rsqrtf(v / (float)hc_dim + eps);
    }
}

// Phase B: grid (mix, rows), one warp per (row, projection row). Each block reads one
// projection row's weights, which is what puts the 1.5 MB of weights across `mix` SMs
// instead of one. The accumulation is the same lane-strided single chain as the
// single-block kernel, so the dot product is bit-identical.
__global__ void hc_mixes_rows_kernel(const float* __restrict__ x, const float* __restrict__ hc_fn,
                                     int rows, int hc_dim, int mix, int split) {
    const int m = blockIdx.x;
    const int r = blockIdx.y;
    const int ck = blockIdx.z;
    if (m >= mix || r >= rows || ck >= split) return;
    const float* xr = x + (size_t)r * hc_dim;
    const float* wr = hc_fn + (size_t)m * hc_dim;
    // Contiguous K range per block, so the 64 KB of one row's weights is read by
    // `split` blocks on `split` different SMs. The per-chunk accumulation is the
    // same lane-strided single chain; the cross-chunk sum happens in phase C in a
    // fixed ascending order, so the result is deterministic.
    int lo = (int)((long long)hc_dim * ck / split);
    int hi = (int)((long long)hc_dim * (ck + 1) / split);
    float acc = 0.f;
    for (int c = lo + threadIdx.x; c < hi; c += blockDim.x) acc += wr[c] * xr[c];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    if (threadIdx.x == 0) g_hc_part[r][m][ck] = acc;
}

// Phase C: one block per token, the sigmoid/cm/sinkhorn/comb tail moved over verbatim
// from hc_mixes_kernel (the sinkhorn already lives in one warp's registers).
__global__ void hc_mixes_post_kernel(const float* __restrict__ hc_scale,
                                     const float* __restrict__ hc_base, float* __restrict__ pre,
                                     float* __restrict__ post, float* __restrict__ comb, int rows,
                                     int hc, int sinkhorn_iters, float eps, int split) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int mix = hc * (2 + hc);
    __shared__ float sm[64];
    __shared__ float cm[64];
    float* mixes = sm;
    for (int m = threadIdx.x; m < mix; m += blockDim.x) {
        float a = 0.f;
        for (int ck = 0; ck < split; ++ck) a += g_hc_part[r][m][ck];   // fixed order
        mixes[m] = a * g_hc_inv[r];
    }
    __syncthreads();
    if (threadIdx.x < (unsigned)hc) {
        const int j = threadIdx.x;
        pre[(size_t)r * hc + j] =
            (1.f / (1.f + expf(-(mixes[j] * hc_scale[0] + hc_base[j])))) + eps;
        post[(size_t)r * hc + j] =
            2.f / (1.f + expf(-(mixes[hc + j] * hc_scale[1] + hc_base[hc + j])));
    }
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
        const int j = jk / hc, k = jk % hc;
        cm[jk] = mixes[2 * hc + j * hc + k] * hc_scale[2] + hc_base[2 * hc + j * hc + k];
    }
    __syncthreads();
    {
        const int hh = hc * hc;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (warp == 0) {
            float c = (lane < hh) ? cm[lane] : 0.f;
            float mx = c;
            for (int off = 1; off < hc; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
            c = expf(c - mx);
            float rs = c;
            for (int off = 1; off < hc; off <<= 1) rs += __shfl_xor_sync(0xFFFFFFFFu, rs, off);
            c = c / rs + eps;
            for (int it = 0; it < sinkhorn_iters; ++it) {
                if (it > 0) {
                    float s = c;
                    for (int off = 1; off < hc; off <<= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
                    c = c / (s + eps);
                }
                float t = c;
                for (int off = hc; off < hh; off <<= 1) t += __shfl_xor_sync(0xFFFFFFFFu, t, off);
                c = c / (t + eps);
            }
            if (lane < hh) cm[lane] = c;
        }
    }
    __syncthreads();
    {
        const int hh = hc * hc;
        for (int jk = threadIdx.x; jk < hh; jk += blockDim.x) comb[(size_t)r * hh + jk] = cm[jk];
    }
}

// Read ONCE: a per-call getenv is a hot-path slip, and this launcher runs ~90 times a step.
static const bool g_hc_spread = [] {
    const char* e = getenv("DSV41_HC_MIXES_SPREAD");
    return e != nullptr && e[0] != '0';      // "0" must mean OFF, not "set"
}();

extern "C" int dsv41_hc_mixes(const float* x, const float* hc_fn, const float* hc_scale,
                              const float* hc_base, float* pre, float* post, float* comb, int rows,
                              int hc_dim, int hc, int sinkhorn_iters, float eps, cudaStream_t s) {
    const int mix = hc * (2 + hc);
    const int smem = (mix + hc * hc) * sizeof(float);
    // One warp per mix row — every row in flight at once. Isolated A/B, same
    // session, 32 tokens: 128 threads = 138.8 ms/token, `mix` warps (768) =
    // 80.4 ms/token, i.e. +72%. The kernel hands row m to warp m (m += nwarp), so
    // at 128 threads only 4 of the 24 rows ran and each warp walked six rows
    // serially; with the grid being just `rows`, the occupancy was there but the
    // work per warp was six dependent rows deep. Note this must NOT be combined
    // with the four-accumulator unroll (DSV41_HC_MIXES_ACC4): that pair measured
    // 348 ms/token, a 4.3x regression over this alone, and the unroll is neutral
    // by itself (138.4 vs 138.8).
    // NOTE: the thread count is mix*32 (one warp per mix row), NOT
    // ((mix+31)/32)*32 — the latter rounds the row count up to a warp count and
    // then uses it as a thread count, which for mix=24 yields 32 threads, i.e. a
    // single warp walking all 24 rows serially: measured 350 ms/token. One warp
    // per row is 80 ms/token on the same binary (+72%).
    int nthreads = mix * 32;
    if (nthreads < 32) nthreads = 32;
    if (nthreads > 1024) nthreads = 1024;
    if (const char* e = getenv("DSV41_HC_MIXES_THREADS")) {
        const int v = atoi(e);
        if (v >= 32 && v <= 1024) nthreads = v;
    }
    if (g_hc_spread && rows <= DSV41_HC_SPREAD_MAXR && mix <= 64) {
        const int split = g_hc_spread_s;
        // Spread variant: one block per (row, projection row), so each of the `mix`
        // 64 KB weight rows is read on its own SM instead of all 1.5 MB on one.
        // The arithmetic order was believed identical in every phase, but a
        // same-session, same-binary A/B does NOT reproduce bit-identical outputs:
        // 31.44 ms against 32.07 ms, yet the answers diverge - the 1+1 prompt turns
        // from a clean '2' into an off-topic sentence. The sinkhorn moving out of
        // the single warp's registers into hc_mixes_post is the likely cause. This
        // path therefore needs element-wise parity evidence (tests/hc_parity.rs)
        // before it may be defaulted; it stays opt-in.
        hc_mixes_ss_kernel<<<rows, 256, 0, s>>>(x, rows, hc_dim, eps);
        hc_mixes_rows_kernel<<<dim3(mix, rows, split), 64, 0, s>>>(x, hc_fn, rows, hc_dim, mix, split);
        hc_mixes_post_kernel<<<rows, 32, 0, s>>>(hc_scale, hc_base, pre, post, comb, rows, hc,
                                                 sinkhorn_iters, eps, split);
        return (int)cudaGetLastError();
    }
    hc_mixes_kernel<<<rows, nthreads, smem, s>>>(x, hc_fn, hc_scale, hc_base, pre, post, comb, rows,
                                            hc_dim, hc, sinkhorn_iters, eps, g_hc_acc4);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_moe_route(const float* x, const uint8_t* gate_w, const uint8_t* gate_w_scale,
                               const float* gate_bias, float* weights, int32_t* indices,
                               int32_t* hist, int rows, int dim, int n_experts, int topk,
                               float gate_temp, int norm_topk_prob, float route_scale,
                               int score_func, cudaStream_t s) {
    const int smem = (n_experts + topk) * sizeof(float);
    moe_route_kernel<<<rows, 128, smem, s>>>(x, gate_w, gate_w_scale, gate_bias, weights, indices,
                                             hist, rows, dim, n_experts, topk, gate_temp,
                                             norm_topk_prob, route_scale, score_func);
    return (int)cudaGetLastError();
}

// ============================================================================
// NOTE — the fp4 expert GEMMs (dsv41_expert_gate_up_fp4 / _down_fp4) are
// implemented in kernels/cuda/dsv41_experts_mxf4.cu with tcgen05.mma
// kind::mxf4.block_scale.scale_vec::2X (the only fp4 tensor-core entry on
// sm_103a; scale type ue8m0 == the checkpoint's per-row-per-32 e8m0 format).
// That file also carries the masked M=128 tile organisation, the canonical
// K-major smem descriptors and the TMEM scale-factor layout notes.
// ============================================================================

// ---------------------------------------------------------------------------
// Segment C fused: the hyper-connection post-mix evaluated IN PLACE on the
// residual stream, so the separate h2 staging buffer and its device-to-device
// copy both disappear.
//
//   out[i,j] = post[i]*x[j] + sum_k comb[k,i]*res[k,j]      (s == 1 in decode)
//
// One thread owns a four-float column and walks ALL n hyper-connection rows
// itself, which matters twice over: the residual is read once instead of once
// per output row, and the aliasing res == out becomes safe by construction
// because a thread's read set and write set are the same columns it alone
// touches. A per-(i,j) thread would need a barrier at best and would still race
// across blocks.
//
// The per-element accumulation keeps the original shape and the same ascending k
// order, so the result is bit-identical to hc_post + copy_h_back.
__global__ void dsv41_hc_post_inplace_kernel(float* __restrict__ res,
                                             const float* __restrict__ x,
                                             const float* __restrict__ post,
                                             const float* __restrict__ comb, int n, int h) {
    const int h4 = h >> 2;
    const int j4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (j4 >= h4) return;
    const int j = j4 << 2;
    float4 r[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= n) break;
        r[k] = *reinterpret_cast<const float4*>(res + (size_t)k * h + j);
    }
    const float4 xv = *reinterpret_cast<const float4*>(x + j);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i >= n) break;
        const float pv = post[i];
        float4 acc = xv;
        acc.x *= pv;
        acc.y *= pv;
        acc.z *= pv;
        acc.w *= pv;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= n) break;
            const float c = comb[(size_t)k * n + i];
            acc.x = __fmaf_rn(c, r[k].x, acc.x);
            acc.y = __fmaf_rn(c, r[k].y, acc.y);
            acc.z = __fmaf_rn(c, r[k].z, acc.z);
            acc.w = __fmaf_rn(c, r[k].w, acc.w);
        }
        *reinterpret_cast<float4*>(res + (size_t)i * h + j) = acc;
    }
}

extern "C" int dsv41_hc_post_inplace(float* res, const float* x, const float* post,
                                     const float* comb, int n, int h, cudaStream_t s) {
    // h % 4 == 0 keeps the float4 path valid; n <= 8 is the register-staging bound
    // above (hc is 4 in this model, 8 is headroom, not an envelope on the data).
    if (res == nullptr || x == nullptr || post == nullptr || comb == nullptr) return (int)cudaErrorInvalidValue;
    if ((h & 3) != 0 || n <= 0 || n > 8) return (int)cudaErrorInvalidValue;
    const int h4 = h >> 2;
    dsv41_hc_post_inplace_kernel<<<(unsigned)((h4 + 255) / 256), 256, 0, s>>>(res, x, post, comb, n, h);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Segment B, cluster 1: hc_collapse + rmsnorm(ffn_norm) as ONE kernel.
//
// The two are inherently a pair: the collapse produces the row the norm
// normalises, and at batch size one the norm's row-reduction already runs as a
// single block, so fusing costs no parallelism while removing one launch and one
// round trip of the collapsed row through global memory.
//
//   collapsed[c] = sum_i pre[i] * x[i*dim + c]          (i ascending, fmaf)
//   out[c]       = collapsed[c] * inv * w[c],  inv = rsqrt(mean(collapsed^2) + eps)
//
// The reduction tree and the summation orders are copied from the two kernels it
// replaces - hc_collapse_kernel's explicit fmaf, and rmsnorm_kernel's
// shfl_down tree plus its in-order cross-warp sum - because two kernels in
// different translation units already disagreed about contraction once and the
// result was a one-ulp shift that flipped preambles. Everything that can be
// pinned is pinned; anything left to the compiler must be settled by the parity
// test, not by reading the source.
__global__ void dsv41_hc_collapse_norm_kernel(const float* __restrict__ x,
                                              const float* __restrict__ pre,
                                              const float* __restrict__ w,
                                              float* __restrict__ out, int hc, int dim, float eps) {
    const int row = blockIdx.x;
    const float* pre_r = pre + (size_t)row * hc;
    const float* x_r = x + (size_t)row * hc * dim;
    float* o_r = out + (size_t)row * dim;

    // phase 1: collapse into the output buffer and accumulate the sum of squares
    float ss = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < hc; ++i) acc = fmaf(pre_r[i], x_r[(size_t)i * dim + c], acc);
        o_r[c] = acc;
        ss += acc * acc;
    }

    // the same reduction tree rmsnorm_kernel uses
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += red[i];
        red[0] = rsqrtf(t / dim + eps);
    }
    __syncthreads();
    const float inv = red[0];

    // phase 2: normalise in place
    for (int c = threadIdx.x; c < dim; c += blockDim.x) o_r[c] = o_r[c] * inv * w[c];
}

extern "C" int dsv41_hc_collapse_norm(float* x, const float* pre, const float* w, float* out, int rows,
                                      int hc, int dim, float eps, cudaStream_t s) {
    if (x == nullptr || pre == nullptr || w == nullptr || out == nullptr) return (int)cudaErrorInvalidValue;
    if (rows <= 0 || hc <= 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    dsv41_hc_collapse_norm_kernel<<<(unsigned)rows, 1024, 0, s>>>(x, pre, w, out, hc, dim, eps);
    return (int)cudaGetLastError();
}

// ============================================================================
// Hyper-connection front end, fused and spread (DSV41_HC_FRONT).
//
// hc_mixes ran as one block per token, which at rows == 1 pins the whole step's
// 1.5 MB of hyper-connection weights to a single SM and therefore to that SM's
// L1 bandwidth - about 39 us per call, 122 calls a step, a quarter of the decode.
// The bytes are not the problem: 1.5 MB is 0.2 us of DRAM. The one-SM ceiling is.
// Worse, one warp per projection row leaves only three float4 loads per lane in
// flight, so each warp sits on DRAM latency instead of on bandwidth.
//
// This shape gives every projection row its own block and stages both the token
// vector and the row in shared memory with cp.async, which needs no registers and
// keeps the whole row in flight, so the block streams at L2/DRAM speed rather
// than at latency. The dot product is the single-block kernel's float4
// three-accumulator chain, reproduced statement for statement, so the
// accumulation order - and therefore the sum - does not move. That matters here:
// an earlier spread shape changed the order and turned the 1+1 prompt from a
// clean '2' into an off-topic sentence.
//
// The tail merges what used to be hc_mixes_ss_kernel and hc_mixes_post_kernel:
// the sum of squares is walked with the same 768-element stride that
// hc_mixes_kernel used (blockDim here is 1024, so warps 24..31 contribute exact
// zeros and the cross-warp sum is unchanged), and the sigmoid/sinkhorn/comb tail
// is the single-block kernel's, verbatim. The whole front end is therefore
// bit-identical to the pair it replaces.
// ============================================================================
__device__ __forceinline__ void dsv41_cp_async16(void* smem, const void* gmem) {
    unsigned sp = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sp), "l"(gmem));
}
__device__ __forceinline__ void dsv41_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void dsv41_cp_wait_all() { asm volatile("cp.async.wait_all;\n"); }

// Phase A: one warp per (token, projection row). Both operands are staged with
// cp.async; the raw dot lands in g_hc_part[r][m][0] for the tail to scale.
__global__ void hc_mix_dots_kernel(const float* __restrict__ x, const float* __restrict__ hc_fn,
                                   int rows, int hc_dim, int mix) {
    const int m = blockIdx.x;
    const int r = blockIdx.y;
    if (m >= mix || r >= rows) return;
    extern __shared__ float hc_sm[];
    float* s_x = hc_sm;                 // hc_dim floats
    float* s_w = hc_sm + hc_dim;        // hc_dim floats, this projection row
    const int lane = threadIdx.x & 31;
    const int n4 = hc_dim >> 2;
    const float4* xg = reinterpret_cast<const float4*>(x + (size_t)r * hc_dim);
    const float4* wg = reinterpret_cast<const float4*>(hc_fn + (size_t)m * hc_dim);
    float4* sx = reinterpret_cast<float4*>(s_x);
    float4* sw = reinterpret_cast<float4*>(s_w);
    for (int i = lane; i < n4; i += 32) dsv41_cp_async16(&sx[i], &xg[i]);
    dsv41_cp_commit();
    for (int i = lane; i < n4; i += 32) dsv41_cp_async16(&sw[i], &wg[i]);
    dsv41_cp_commit();
    dsv41_cp_wait_all();
    __syncwarp();
    // hc_mixes_kernel's float4 three-accumulator chain, verbatim
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    int c = lane;
    for (; c + 64 < n4; c += 96) {
        const float4 w0 = sw[c], w1 = sw[c + 32], w2 = sw[c + 64];
        const float4 v0 = sx[c], v1 = sx[c + 32], v2 = sx[c + 64];
        a0 += w0.x * v0.x + w0.y * v0.y + w0.z * v0.z + w0.w * v0.w;
        a1 += w1.x * v1.x + w1.y * v1.y + w1.z * v1.z + w1.w * v1.w;
        a2 += w2.x * v2.x + w2.y * v2.y + w2.z * v2.z + w2.w * v2.w;
    }
    for (; c < n4; c += 32) {
        const float4 w = sw[c], v = sx[c];
        a0 += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
    }
    for (int k = (n4 << 2) + lane; k < hc_dim; k += 32) a0 += s_w[k] * s_x[k];
    float acc = (a0 + a1) + a2;
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) g_hc_part[r][m][0] = acc;
}

// Phase B: one block per token, 1024 threads. The sum of squares, the scale, the
// sigmoid split, the sinkhorn and the comb write - all as hc_mixes_kernel does
// them, with the walk over hc_dim striding by ss_stride (= mix*32) so the
// per-lane partial sums group exactly as they did at blockDim 768.
__global__ void hc_mixes_tail_kernel(const float* __restrict__ x,
                                     const float* __restrict__ hc_scale,
                                     const float* __restrict__ hc_base, float* __restrict__ pre,
                                     float* __restrict__ post, float* __restrict__ comb, int hc,
                                     int dim, int sinkhorn_iters, float eps, int ss_stride,
                                     const float* __restrict__ w_norm,
                                     const float* __restrict__ pre_collapse,
                                     float* __restrict__ out, float eps_norm) {
    const int r = blockIdx.x;
    const int mix = hc * (2 + hc);
    const int hc_dim = hc * dim;
    extern __shared__ float t_sm[];
    float* mixes = t_sm;                // mix
    float* cm = t_sm + 64;              // hc*hc
    __shared__ float sss;
    __shared__ float wpart[32];
    const float* xr = x + (size_t)r * hc_dim;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hc_dim; c += ss_stride) ss += xr[c] * xr[c];
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
    const int nwarp = ss_stride >> 5;
    if ((threadIdx.x & 31) == 0 && (int)(threadIdx.x >> 5) < nwarp)
        wpart[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < nwarp) ? wpart[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (threadIdx.x == 0) sss = v;
    }
    __syncthreads();
    const float inv = rsqrtf(sss / (float)hc_dim + eps);
    for (int m = threadIdx.x; m < mix; m += blockDim.x) mixes[m] = g_hc_part[r][m][0] * inv;
    __syncthreads();
    if (threadIdx.x < (unsigned)hc) {
        const int j = threadIdx.x;
        pre[(size_t)r * hc + j] =
            (1.f / (1.f + expf(-(mixes[j] * hc_scale[0] + hc_base[j])))) + eps;
        post[(size_t)r * hc + j] =
            2.f / (1.f + expf(-(mixes[hc + j] * hc_scale[1] + hc_base[hc + j])));
    }
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
        const int j = jk / hc, k = jk % hc;
        cm[jk] = mixes[2 * hc + j * hc + k] * hc_scale[2] + hc_base[2 * hc + j * hc + k];
    }
    __syncthreads();
    {
        const int hh = hc * hc;
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (warp == 0) {
            float c = (lane < hh) ? cm[lane] : 0.f;
            float mx = c;
            for (int off = 1; off < hc; off <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
            c = expf(c - mx);
            float rs = c;
            for (int off = 1; off < hc; off <<= 1) rs += __shfl_xor_sync(0xFFFFFFFFu, rs, off);
            c = c / rs + eps;
            for (int it = 0; it < sinkhorn_iters; ++it) {
                if (it > 0) {
                    float s = c;
                    for (int off = 1; off < hc; off <<= 1)
                        s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
                    c = c / (s + eps);
                }
                float t = c;
                for (int off = hc; off < hh; off <<= 1)
                    t += __shfl_xor_sync(0xFFFFFFFFu, t, off);
                c = c / (t + eps);
            }
            if (lane < hh) cm[lane] = c;
        }
    }
    __syncthreads();
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x)
        comb[(size_t)r * hc * hc + jk] = cm[jk];
    // Collapse + rmsnorm, the body of dsv41_hc_collapse_norm_kernel. It reads the
    // `pre` of the slot the caller names for the collapse, which is NOT the one the
    // mixes just wrote: the block walks the premix slots so the attention half
    // collapses with the previous layer's coefficients. Folding it in here is what
    // makes this the whole front end in two launches.
    if (w_norm != nullptr) {
        __syncthreads();
        float* o_r = out + (size_t)r * dim;
        float s2 = 0.f;
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            float acc = 0.f;
            for (int i = 0; i < hc; ++i)
                acc = fmaf(pre_collapse[(size_t)r * hc + i], xr[(size_t)i * dim + c], acc);
            o_r[c] = acc;
            s2 += acc * acc;
        }
        // the same reduction tree rmsnorm_kernel uses
        for (int off = 16; off > 0; off >>= 1) s2 += __shfl_down_sync(0xffffffffu, s2, off);
        __shared__ float red[32];
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s2;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.f;
            for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += red[i];
            red[0] = rsqrtf(t / dim + eps_norm);
        }
        __syncthreads();
        const float inv2 = red[0];
        for (int c = threadIdx.x; c < dim; c += blockDim.x) o_r[c] = o_r[c] * inv2 * w_norm[c];
    }
}

static const bool g_hc_front = [] {
    const char* e = getenv("DSV41_HC_FRONT");
    // Default on: 16.85 ms against 19.96 ms in one session, same binary, with the
    // four prompts character-for-character correct. "0" still opts out.
    if (e == nullptr) return true;
    return e[0] != '0';
}();

extern "C" int dsv41_hc_front(const float* x, const float* hc_fn, const float* hc_scale,
                              const float* hc_base, const float* w_norm, const float* pre_collapse,
                              float* pre, float* post, float* comb, float* out, int rows, int hc,
                              int dim, int sinkhorn_iters, float eps, float eps_norm,
                              cudaStream_t s) {
    if (x == nullptr || hc_fn == nullptr || hc_scale == nullptr || hc_base == nullptr ||
        pre == nullptr || post == nullptr || comb == nullptr)
        return (int)cudaErrorInvalidValue;
    if (rows <= 0 || hc <= 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    if (!g_hc_front) return (int)cudaErrorInvalidValue;   // caller keeps the old path
    if (rows > DSV41_HC_SPREAD_MAXR) return (int)cudaErrorInvalidValue;
    // The collapse half is optional (w_norm null skips it), but when it is asked
    // for it must have somewhere to read and write.
    if ((w_norm == nullptr) != (pre_collapse == nullptr)) return (int)cudaErrorInvalidValue;
    if (w_norm != nullptr && out == nullptr) return (int)cudaErrorInvalidValue;
    const int mix = hc * (2 + hc);
    const int hc_dim = hc * dim;
    const size_t smem = (size_t)2 * (size_t)hc_dim * sizeof(float);
    // Set it EVERY call. The attribute is per-context, and a TP8 process has one
    // context per rank, so a process-wide guard leaves seven of the eight ranks
    // without the 160 KB opt-in - the kernel then launches with shared memory it
    // was not granted and the cp.async waits never retire. The existing gemm_fp8
    // launcher carries the same warning for the same reason.
    {
        cudaError_t e = cudaFuncSetAttribute(hc_mix_dots_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
        if (e != cudaSuccess) {
            (void)cudaGetLastError();   // clear the sticky flag before reporting
            return (int)e;
        }
    }
    hc_mix_dots_kernel<<<dim3((unsigned)mix, (unsigned)rows), 32, smem, s>>>(x, hc_fn, rows, hc_dim,
                                                                             mix);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    hc_mixes_tail_kernel<<<(unsigned)rows, 1024, (64 + 64) * sizeof(float), s>>>(
        x, hc_scale, hc_base, pre, post, comb, hc, dim, sinkhorn_iters, eps, mix * 32, w_norm,
        pre_collapse, out, eps_norm);
    return (int)cudaGetLastError();
}
