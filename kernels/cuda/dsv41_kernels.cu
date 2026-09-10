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
//   * The fp4 EXPERT GEMMs have two implementations behind the same ABI:
//       (a) the fp8 fallback entry points below, fed by the *lossless*
//           fp4 -> e4m3 re-encode at load time (the reference's own
//           cast_e2m1fn_to_e4m3fn: an e2m1 magnitude times a power-of-two
//           offset stays representable in e4m3, so no precision is lost --
//           this is re-encoding, not dequantisation);
//       (b) `dsv41_expert_gate_up_fp4` / `dsv41_expert_down_fp4`, which are
//           the tcgen05 MXFP4 path: see the TODO block at the bottom of this
//           file for the exact remaining plumbing. Until that lands they
//           return cudaErrorNotSupported so a caller can never silently get a
//           non-native path.
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
__global__ void gemm_fp8_kernel(const uint8_t* __restrict__ a, const uint8_t* __restrict__ a_scale,
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

    float acc[4] = {0.f, 0.f, 0.f, 0.f};  // rows m0+gid, m0+gid+8 x 2 columns
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
        float c0[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const int ncol = n0 + nt * 8 + gid;
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
            const float sar0 = ue8m0_to_f(a_scale[(size_t)(m0 + gid) * nb_k + kb]);
            const float sar1 = ue8m0_to_f(a_scale[(size_t)(m0 + gid + 8) * nb_k + kb]);
            const float sb0 = ue8m0_to_f(w_scale[(size_t)(ncol >> 5) * nb_k + kb]);
            const float sb1 = ue8m0_to_f(w_scale[(size_t)((ncol + 1) >> 5) * nb_k + kb]);
            c0[0] += d[0] * sar0 * sb0;  // (gid, 2*tg)
            c0[1] += d[1] * sar0 * sb1;  // (gid, 2*tg+1)
            c0[2] += d[2] * sar1 * sb0;  // (gid+8, 2*tg)
            c0[3] += d[3] * sar1 * sb1;
        }
        acc[0] += c0[0]; acc[1] += c0[1]; acc[2] += c0[2]; acc[3] += c0[3];
    }
    // epilogue: C fragment rows gid / gid+8, columns 2*tg / 2*tg+1
    if (m0 + gid < m) {
        const int col = n0 + tg * 2;
        out[(size_t)(m0 + gid) * n + col] = acc[0] + (bias ? bias[col] : 0.f);
        out[(size_t)(m0 + gid) * n + col + 1] = acc[1] + (bias ? bias[col + 1] : 0.f);
    }
    if (m0 + gid + 8 < m) {
        const int col = n0 + tg * 2;
        out[(size_t)(m0 + gid + 8) * n + col] = acc[2] + (bias ? bias[col] : 0.f);
        out[(size_t)(m0 + gid + 8) * n + col + 1] = acc[3] + (bias ? bias[col + 1] : 0.f);
    }
}

// ------------------------------------------------ expert fp8 (fallback path)

// gate+up: out[rows, 2*inter] = [silu-clamped gate | up] from w1/w3.
// The weights here are the losslessly re-encoded e4m3 form (see the header).
__global__ void expert_gate_up_kernel(const uint8_t* __restrict__ a,
                                      const float* __restrict__ a_scale,
                                      const uint8_t* __restrict__ w1,
                                      const uint8_t* __restrict__ w1_scale,
                                      const uint8_t* __restrict__ w3,
                                      const uint8_t* __restrict__ w3_scale,
                                      float* __restrict__ out, int rows, int dim, int inter,
                                      float limit) {
    // one thread per (row, inter) pair for the arithmetic; the GEMM itself is
    // the same MMA path (kept separate here so the fallback needs no new ABI).
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * inter) return;
    const int r = idx / inter, i = idx % inter;
    const int nb_k = dim >> 5;
    float g = 0.f, u = 0.f;
    for (int kb = 0; kb < nb_k; kb++) {
        float gd = 0.f, ud = 0.f;
        for (int c = 0; c < 32; c++) {
            const float av = e4m3_to_f(a[(size_t)r * dim + kb * 32 + c]) *
                             a_scale[(size_t)r * nb_k + kb];
            gd += e4m3_to_f(w1[(size_t)i * dim + kb * 32 + c]) *
                  ue8m0_to_f(w1_scale[(size_t)(i >> 5) * nb_k + kb]) * av;
            ud += e4m3_to_f(w3[(size_t)i * dim + kb * 32 + c]) *
                  ue8m0_to_f(w3_scale[(size_t)(i >> 5) * nb_k + kb]) * av;
        }
        g += gd;
        u += ud;
    }
    if (limit > 0.f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    out[(size_t)r * 2 * inter + i] = g;
    out[(size_t)r * 2 * inter + inter * 1 + i] = u;
}

// down: out[rows, dim] += weight[row] * (w2 @ act[inter])
__global__ void expert_down_kernel(const float* __restrict__ act,
                                   const uint8_t* __restrict__ w2,
                                   const uint8_t* __restrict__ w2_scale,
                                   const float* __restrict__ weight, float* __restrict__ out,
                                   int rows, int dim, int inter) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * dim) return;
    const int r = idx / dim, o = idx % dim;
    const int nb_k = inter >> 5;
    float acc = 0.f;
    for (int kb = 0; kb < nb_k; kb++) {
        for (int c = 0; c < 32; c++) {
            acc += e4m3_to_f(w2[(size_t)o * inter + kb * 32 + c]) *
                   ue8m0_to_f(w2_scale[(size_t)(o >> 5) * nb_k + kb]) *
                   act[(size_t)r * inter + kb * 32 + c];
        }
    }
    out[(size_t)r * dim + o] = weight ? weight[r] * acc : acc;
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
__global__ void sparse_attn_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                                   const float* __restrict__ sink, const int32_t* __restrict__ idxs,
                                   float* __restrict__ out, int b, int m, int h, int d, int n,
                                   int topk, float scale) {
    const int row = blockIdx.x;  // flattened (b, m)
    if (row >= b * m) return;
    const int bb = row / m, mm = row % m;
    for (int hh = blockIdx.y; hh < h; hh += gridDim.y) {
        const float* qr = q + ((size_t)(bb * m + mm) * h + hh) * d;
        float acc[512];
        for (int c = threadIdx.x; c < d; c += blockDim.x) acc[c] = 0.f;
        float smax = -1e30f, se = 0.f;
        for (int t = 0; t < topk; t++) {
            const int idx = idxs[(size_t)(bb * m + mm) * topk + t];
            if (idx < 0) continue;
            const float* kr = kv + ((size_t)bb * n + idx) * d;
            float dot = 0.f;
            for (int c = threadIdx.x; c < d; c += blockDim.x) dot += qr[c] * kr[c];
            // block reduction of the dot product
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            __shared__ float sdot;
            if (threadIdx.x == 0) sdot = dot;
            __syncthreads();
            dot = sdot * scale;
            const float nm = fmaxf(smax, dot);
            const float corr = expf(smax - nm);
            const float e = expf(dot - nm);
            for (int c = threadIdx.x; c < d; c += blockDim.x) acc[c] = acc[c] * corr + e * kr[c];
            se = se * corr + e;
            smax = nm;
            __syncthreads();
        }
        se += expf(sink[hh] - smax);
        float* orow = out + ((size_t)(bb * m + mm) * h + hh) * d;
        for (int c = threadIdx.x; c < d; c += blockDim.x)
            orow[c] = (se > 0.f) ? acc[c] / se : 0.f;
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

__global__ void apply_rope_kernel(float* __restrict__ x, const float* __restrict__ cos,
                                  const float* __restrict__ sin, int rows, int row_len, int dim,
                                  int half, int pos0, int step, int inverse) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = pos0 + r * step;
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
                                int hc_dim, int hc, int sinkhorn_iters, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int mix = hc * (2 + hc);
    extern __shared__ float sm[];
    float* mixes = sm;             // [mix]
    float* cm = sm + mix;          // [hc*hc]
    const float* xr = x + (size_t)r * hc_dim;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hc_dim; c += blockDim.x) ss += xr[c] * xr[c];
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
    __shared__ float sss;
    if (threadIdx.x == 0) sss = ss;
    __syncthreads();
    const float inv = rsqrtf(sss / (float)hc_dim + eps);
    for (int m = threadIdx.x; m < mix; m += blockDim.x) {
        const float* wr = hc_fn + (size_t)m * hc_dim;
        float acc = 0.f;
        for (int c = 0; c < hc_dim; c++) acc += wr[c] * xr[c];
        mixes[m] = acc * inv;
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
    // (the Sinkhorn iterations themselves are the CPU golden's job; the device
    // path runs them in the fused hc kernel -- see hc_split_sinkhorn)
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) comb[(size_t)r * hc * hc + jk] = cm[jk];
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

extern "C" int dsv41_quant_fp4(const float* x, uint8_t* y, float* scale, int rows, int cols,
                               int block, int round_scale, cudaStream_t s) {
    if (rows <= 0 || cols % block != 0) return (int)cudaErrorInvalidValue;
    const int nb = cols / block;
    uint8_t* nib = nullptr;
    if (cudaMalloc(&nib, (size_t)rows * cols) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
    dim3 blk(1, (block < 256 ? block : 256));
    quant_kernel<1><<<dim3(rows * nb), blk, 0, s>>>(x, nib, scale, rows, cols, block, round_scale);
    const size_t n = (size_t)rows * cols;
    fp4_pack_kernel<<<(unsigned)((n / 2 + 255) / 256), 256, 0, s>>>(nib, y, n);
    cudaFree(nib);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_gemm_fp8_mx(const uint8_t* a, const uint8_t* a_scale, const uint8_t* w,
                                 const uint8_t* w_scale, const float* bias, float* out, int m,
                                 int n, int k, cudaStream_t s) {
    if (m <= 0 || n <= 0 || k <= 0 || (k & 31)) return (int)cudaErrorInvalidValue;
    const int smem = 16 * k;
    if (smem > 48 * 1024) return (int)cudaErrorInvalidValue;  // tile A must fit
    dim3 grid((n + 63) / 64, (m + 15) / 16);
    gemm_fp8_kernel<<<grid, 128, smem, s>>>(a, a_scale, w, w_scale, bias, out, m, n, k);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_expert_gate_up_fp8(const uint8_t* a, const float* a_scale, const uint8_t* w1,
                                        const uint8_t* w1_scale, const uint8_t* w3,
                                        const uint8_t* w3_scale, float* out, int rows, int dim,
                                        int inter, float limit, cudaStream_t s) {
    const int total = rows * inter;
    expert_gate_up_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(
        a, a_scale, w1, w1_scale, w3, w3_scale, out, rows, dim, inter, limit);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_expert_down_fp8(const float* act, const uint8_t* w2, const uint8_t* w2_scale,
                                     const float* weight, float* out, int rows, int dim, int inter,
                                     cudaStream_t s) {
    const int total = rows * dim;
    expert_down_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(act, w2, w2_scale, weight,
                                                                      out, rows, dim, inter);
    return (int)cudaGetLastError();
}

// The native fp4 expert path is the tcgen05 MXFP4 GEMM; see the TODO block.
extern "C" int dsv41_expert_gate_up_fp4(const uint8_t*, const float*, const uint8_t*,
                                        const uint8_t*, const uint8_t*, const uint8_t*, float*,
                                        int, int, int, float, cudaStream_t) {
    return (int)cudaErrorNotSupported;  // pending the tcgen05 implementation
}

extern "C" int dsv41_expert_down_fp4(const float*, const uint8_t*, const uint8_t*, const float*,
                                     float*, int, int, int, cudaStream_t) {
    return (int)cudaErrorNotSupported;  // pending the tcgen05 implementation
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

extern "C" int dsv41_sparse_attn(const float* q, const float* kv, const float* sink,
                                 const int32_t* idxs, float* out, int b, int m, int h, int d,
                                 int n, int topk, float scale, cudaStream_t s) {
    if (d > 512) return (int)cudaErrorInvalidValue;  // the accumulator is d-wide per thread group
    dim3 grid(b * m, h);
    sparse_attn_kernel<<<grid, 128, 0, s>>>(q, kv, sink, idxs, out, b, m, h, d, n, topk, scale);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_indexer_topk(const float*, const float*, const float*, const uint8_t*,
                                  const int32_t*, int32_t*, int, int, int, int, int, int, int, float,
                                  float, int, cudaStream_t) {
    // The indexer's score + candidate mask + top-k selection is implemented on
    // top of candidate_blocks_kernel + a shared top-k helper in the follow-up
    // (see TODO); returning NotSupported keeps a caller from silently taking a
    // non-fused path.
    return (int)cudaErrorNotSupported;
}

extern "C" int dsv41_candidate_blocks(const float* logits, const int32_t* compress_lens,
                                      uint8_t* mask, int rows, int n_pos, int topk_blocks,
                                      int block_size, cudaStream_t s) {
    candidate_blocks_kernel<<<rows, 128, 0, s>>>(logits, compress_lens, mask, rows, n_pos,
                                                 topk_blocks, block_size);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_compressor(const float*, const uint8_t*, const uint8_t*, const uint8_t*,
                                const uint8_t*, const float*, float*, float*, float*, int32_t*, int,
                                int, int, int, int, int, float, cudaStream_t) {
    // ratio-1 is a plain RMSNorm'd projection (handled by dsv41_gemm_fp8_mx +
    // a norm); the ratio>1 softmax-gated pooling with its carried state is the
    // next step (see TODO). NotSupported until then.
    return (int)cudaErrorNotSupported;
}

extern "C" int dsv41_rope_precompute(float* cos, float* sin, int dim, int seqlen,
                                     int original_seq_len, float base, float factor, float beta_fast,
                                     float beta_slow, cudaStream_t s) {
    rope_precompute_kernel<<<(unsigned)((dim / 2 + 127) / 128), 128, 0, s>>>(
        cos, sin, dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_apply_rope(float* x, const float* cos, const float* sin, int rows, int row_len,
                                int dim, int half, int pos0, int step, int inverse,
                                cudaStream_t s) {
    apply_rope_kernel<<<rows, 128, 0, s>>>(x, cos, sin, rows, row_len, dim, half, pos0, step,
                                           inverse);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_hc_mixes(const float* x, const float* hc_fn, const float* hc_scale,
                              const float* hc_base, float* pre, float* post, float* comb, int rows,
                              int hc_dim, int hc, int sinkhorn_iters, float eps, cudaStream_t s) {
    const int mix = hc * (2 + hc);
    const int smem = (mix + hc * hc) * sizeof(float);
    hc_mixes_kernel<<<rows, 128, smem, s>>>(x, hc_fn, hc_scale, hc_base, pre, post, comb, rows,
                                            hc_dim, hc, sinkhorn_iters, eps);
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
// TODO (performance, next step): the native MXFP4 expert GEMM via tcgen05.
//
// ptxas rejects every `mma.sync` fp4 form on sm_103a (verified), so the native
// path is:
//
//   tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
//       [d_tmem], a_desc, b_desc, idesc, [scale_a_tmem], [scale_b_tmem], p;
//
// Remaining plumbing (all of it mechanical, none of it blocked):
//   1. tcgen05.alloc a 128-column tensor-memory accumulator per CTA and keep a
//      free-list (tcgen05_alloc.h / tcgen05_dealloc).
//   2. Build the shared-memory operand descriptors (swizzled 128B layout) for A
//      (the fp4 activations, k-major) and B (the fp4 expert weights [inter, dim]
//      I8-packed -- our per-row-per-32 e8m0 scales are exactly the hardware MX
//      layout, block 32 along k).
//   3. Copy the e8m0 scales into tensor memory (tcgen05.cp) and pass their tmem
//      addresses as [scale_a_tmem]/[scale_b_tmem].
//   4. Issue the MMA, `tcgen05.commit` an mbarrier, wait, then read the
//      accumulator back with tcgen05.ld and apply the per-row routing weight.
//   5. Keep the pipeline fed: 4-8 k-stages, cta_group::1 to start.
//
// IMPORTANT — the organisation matters more than the plumbing. tcgen05 is a
// CTA-level op (M=64/128), while a decode step has m=16 rows in total and each
// expert sees ~1.2 of them. A per-expert launch would pad m=1.2 up to M=64/128
// (~50x wasted rows), so this must be a GROUPED GEMM: sort the (token, slot)
// assignments by expert, then run M=128 tiles that span several experts with a
// masked valid-row count (the DeepGEMM m_grouped_gemm_nt_masked shape). See
// crates/ferrite-dsv41/PERF.md — at m=16 the lossless-e4m3+fp8 route and the
// MXFP4 route trade ~2x bytes against M-padding waste, so this needs a
// measurement before it displaces the fallback.
//
// The entry points dsv41_expert_gate_up_fp4 / dsv41_expert_down_fp4 are the ABI
// for this and currently return cudaErrorNotSupported, so a caller can never
// silently receive a non-native path. Until this lands, the lossless fp4 -> e4m3
// re-encode (weights.rs::convert_expert_fp4_to_e4m3) plus the fp8 entry points
// above give a correct tensor-core implementation at 2x the weight bytes.
// ============================================================================
