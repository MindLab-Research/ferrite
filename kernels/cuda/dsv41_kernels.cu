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
                for (int c = lane; c < hc_dim; c += 32) acc += wr[c] * xr[c];
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
    __shared__ float row_max[16], row_sum[16], col_sum[16];
    for (int j = 0; j < hc; j++) {
        if (threadIdx.x == 0) {
            float mx = -INFINITY;
            for (int k = 0; k < hc; k++) mx = fmaxf(mx, cm[j * hc + k]);
            row_max[j] = mx;
        }
    }
    __syncthreads();
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
        const int j = jk / hc;
        cm[jk] = expf(cm[jk] - row_max[j]);
    }
    __syncthreads();
    for (int j = 0; j < hc; j++) {
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int k = 0; k < hc; k++) s += cm[j * hc + k];
            row_sum[j] = s;
        }
    }
    __syncthreads();
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
        const int j = jk / hc;
        cm[jk] = cm[jk] / row_sum[j] + eps;
    }
    __syncthreads();
    // Reference sequence (kernel.py hc_split_sinkhorn):
    //   comb = softmax(comb, -1) + eps      (rows)
    //   comb = comb / (colsum + eps)        (columns)
    //   for _ in range(iters-1): row-normalise, then col-normalise
    // It ENDS with a column normalisation; ending with a row one (as the first
    // version did) leaves the rows summing to 1 instead of the columns, so the
    // mixing weights are off by that factor.
    for (int it = 0; it < sinkhorn_iters; it++) {
        if (it > 0) {
            // row normalise (skipped on the first pass: softmax already did it)
            for (int j = 0; j < hc; j++) {
                if (threadIdx.x == 0) {
                    float s = 0.f;
                    for (int k = 0; k < hc; k++) s += cm[j * hc + k];
                    row_sum[j] = s;
                }
            }
            __syncthreads();
            for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
                const int j = jk / hc;
                cm[jk] = cm[jk] / (row_sum[j] + eps);
            }
            __syncthreads();
        }
        // column normalise
        for (int k = 0; k < hc; k++) {
            if (threadIdx.x == 0) {
                float s = 0.f;
                for (int j = 0; j < hc; j++) s += cm[j * hc + k];
                col_sum[k] = s;
            }
        }
        __syncthreads();
        for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
            const int k = jk % hc;
            cm[jk] = cm[jk] / (col_sum[k] + eps);
        }
        __syncthreads();
    }
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


// --------------------------------------------------------- indexer (level 2)
// One CTA per (b, m) row of `indexer_topk` (ops.rs golden):
//   score[p] = sum_h relu(q[b,m,h,:] . k[b,p,:]) * weights[b,m,h]
//              * softmax_scale * head_scale
//   p >= compress_lens[m] -> -inf ; candidate-masked -> -inf
// then top-`cols` with the golden's stable tie rule (equal scores keep the
// lower position), the picked positions re-sorted ascending, and `p + offset`
// written for reachable positions / -1 otherwise.
__global__ void indexer_topk_kernel(const float* __restrict__ q, const float* __restrict__ ik,
                                    const float* __restrict__ w, const uint8_t* __restrict__ cand,
                                    const int32_t* __restrict__ lens, int32_t* __restrict__ out,
                                    int m, int nh, int hd, int n_pos, int topk, int offset,
                                    float softmax_scale, float head_scale, int uses_cand) {
    extern __shared__ float smem[];
    const int cols = topk < n_pos ? topk : n_pos;
    float* s_score = smem;                  // [n_pos]
    int* s_pick = (int*)(s_score + n_pos);  // [cols]
    int* s_sort = s_pick + cols;            // [cols]
    uint8_t* s_used = (uint8_t*)(s_sort + cols);  // [n_pos]
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int mm = blockIdx.x, bb = blockIdx.y;
    const size_t row = (size_t)bb * m + mm;

    int cl = n_pos;
    if (lens != nullptr) cl = lens[mm];
    if (cl > n_pos) cl = n_pos;
    for (int i = tid; i < n_pos; i += nthr) s_used[i] = 0;
    __syncthreads();

    // ---- scores
    const float* qrow = q + row * (size_t)nh * hd;
    for (int p = tid; p < n_pos; p += nthr) {
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
        s_score[p] = sv;
    }
    __syncthreads();

    // ---- stable top-`cols` (iterative block argmax; ties -> lower index)
    __shared__ float s_bv[32];
    __shared__ int s_bi[32];
    const int warp = tid >> 5, lane = tid & 31, nw = nthr >> 5;
    for (int it = 0; it < cols; ++it) {
        float bv = -INFINITY;
        int bi = -1;  // filled on the first scanned entry, even if it is -inf
        for (int p = tid; p < n_pos; p += nthr) {
            if (s_used[p]) continue;
            const float v = s_score[p];
            if (bi < 0 || v > bv) {
                bv = v;
                bi = p;
            }
        }
        if (bi < 0) bi = n_pos;  // this thread had no entries, use the sentinel
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_bv[warp] = bv;
            s_bi[warp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float xv = -INFINITY;
            int xi = n_pos;
            for (int widx = 0; widx < nw; ++widx) {
                const float v = s_bv[widx];
                const int i = s_bi[widx];
                if (v > xv || (v == xv && i < xi)) {
                    xv = v;
                    xi = i;
                }
            }
            s_pick[it] = xi;
            if (xi < n_pos) s_used[xi] = 1;  // consume it (marking -inf is a no-op)
        }
        __syncthreads();
    }

    // ---- sort the picked positions ascending (unique -> rank by counting)
    if (cols > 0) {
        for (int i = tid; i < cols; i += nthr) {
            const int pi = s_pick[i];
            int rank = 0;
            for (int j = 0; j < cols; ++j) {
                const int pj = s_pick[j];
                if (pj < pi || (pj == pi && j < i)) ++rank;
            }
            s_sort[rank] = pi;
        }
    }
    __syncthreads();
    for (int i = tid; i < cols; i += nthr) {
        const int p = s_sort[i];
        out[row * (size_t)cols + i] = (p < cl) ? (p + offset) : -1;
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
                                        int hd, int ratio, int start_pos) {
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
                                       int mode, int grid_n, int out_rows_val, int b, int seqlen,
                                       int hd, int ratio, int start_pos, float eps) {
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
    (void)start_pos;
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

extern "C" int dsv41_sparse_attn(const float* q, const float* kv, const float* sink,
                                 const int32_t* idxs, float* out, int b, int m, int h, int d,
                                 int n, int topk, float scale, cudaStream_t s) {
    if (d > 512) return (int)cudaErrorInvalidValue;  // the accumulator is d-wide per thread group
    dim3 grid(b * m, h);
    sparse_attn_kernel<<<grid, 128, 0, s>>>(q, kv, sink, idxs, out, b, m, h, d, n, topk, scale);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_indexer_topk(const float* q, const float* index_k, const float* weights,
                                  const uint8_t* candidates, const int32_t* compress_lens,
                                  int32_t* out, int b, int m, int nh, int hd, int n_pos, int topk,
                                  int offset, float softmax_scale, float head_scale,
                                  int uses_candidates, cudaStream_t s) {
    if (b <= 0 || m <= 0 || nh <= 0 || hd <= 0 || topk <= 0) return (int)cudaErrorInvalidValue;
    if (n_pos <= 0) return (int)cudaSuccess;  // nothing to select from
    const int cols = topk < n_pos ? topk : n_pos;
    const size_t smem =
        (size_t)n_pos * (sizeof(float) + 1) + (size_t)cols * 2 * sizeof(int) + 64;
    if (smem > 200 * 1024) return (int)cudaErrorInvalidValue;  // one CTA holds all scores
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
                                     int start_pos, float eps, cudaStream_t s) {
    if (b <= 0 || seqlen <= 0 || head_dim <= 0 || ratio <= 0) return (int)cudaErrorInvalidValue;
    if (ratio > 1) {
        const size_t work = start_pos == 0 ? (size_t)b * (seqlen % ratio) * head_dim
                                           : (size_t)b * head_dim;
        const int blocks = (int)((work + 255) / 256);
        if (work > 0)
            compressor_state_kernel<<<(blocks > 0 ? blocks : 1), 256, 0, s>>>(
                kvp, scp, state_kv, state_score, b, seqlen, head_dim, ratio, start_pos);
    }
    int mode, grid_n, out_rows_val;
    if (ratio == 1) {
        mode = 0;
        grid_n = seqlen;
        out_rows_val = seqlen;
    } else if (start_pos == 0) {
        mode = 1;
        grid_n = b * (seqlen / ratio);
        out_rows_val = seqlen / ratio;
    } else {
        mode = 2;
        grid_n = b;
        out_rows_val = ((start_pos + 1) % ratio == 0) ? 1 : 0;
    }
    const unsigned launch = (unsigned)(grid_n > 0 ? grid_n : 1);
    compressor_pool_kernel<<<launch, 128, 0, s>>>(kvp, scp, norm_w, state_kv, state_score, latents,
                                                  out_rows, mode, grid_n, out_rows_val, b, seqlen,
                                                  head_dim, ratio, start_pos, eps);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_compressor(const float* x, const uint8_t* wkv, const uint8_t* wkv_scale,
                                const uint8_t* wgate, const uint8_t* wgate_scale,
                                const float* norm_w, float* state_kv, float* state_score,
                                float* latents, int32_t* out_rows, int b, int seqlen, int dim,
                                int head_dim, int ratio, int start_pos, float eps, cudaStream_t s) {
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
                ratio, start_pos);
    }
    // 4. pooling + RMSNorm + the out_rows decision
    int mode, grid_n, out_rows_val;
    if (ratio == 1) {
        mode = 0;
        grid_n = rows;
        out_rows_val = seqlen;
    } else if (start_pos == 0) {
        mode = 1;
        grid_n = b * (seqlen / ratio);
        out_rows_val = seqlen / ratio;
    } else {
        mode = 2;
        grid_n = b;
        out_rows_val = ((start_pos + 1) % ratio == 0) ? 1 : 0;
    }
    {
        const unsigned launch = (unsigned)(grid_n > 0 ? grid_n : 1);
        compressor_pool_kernel<<<launch, 128, 0, s>>>(
            (const float*)kvp, (const float*)scp, norm_w, state_kv, state_score, latents, out_rows,
            mode, grid_n, out_rows_val, b, seqlen, head_dim, ratio, start_pos, eps);
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
                                int dim, int half, int pos0, int step, int inverse,
                                cudaStream_t s) {
    apply_rope_kernel<<<rows, 128, 0, s>>>(x, cos, sin, rows, row_len, dim, half, pos0, step,
                                           inverse);
    return (int)cudaGetLastError();
}

static const bool g_hc_acc4 = getenv("DSV41_HC_MIXES_ACC4") != nullptr;

extern "C" int dsv41_hc_mixes(const float* x, const float* hc_fn, const float* hc_scale,
                              const float* hc_base, float* pre, float* post, float* comb, int rows,
                              int hc_dim, int hc, int sinkhorn_iters, float eps, cudaStream_t s) {
    const int mix = hc * (2 + hc);
    const int smem = (mix + hc * hc) * sizeof(float);
    // NOTE (negative result, 2026 session): sizing this block to `mix` warps (so
    // every mix row is in flight at once) made decode 2.7x SLOWER (348 ms/token vs
    // 130), even though it reads like an obvious win: the grid is `rows`, which is
    // tiny, so growing the block by 6x collapsed the blocks-per-SM and the kernel
    // simply lost occupancy. Measured, not reasoned - and it was tested together
    // with the four-accumulator unroll below, so the two are not yet separated.
    // DSV41_HC_MIXES_THREADS re-enables the wide block for a controlled retest.
    int nthreads = 128;
    if (const char* e = getenv("DSV41_HC_MIXES_THREADS")) {
        const int v = atoi(e);
        if (v >= 32 && v <= 1024) nthreads = v;
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
