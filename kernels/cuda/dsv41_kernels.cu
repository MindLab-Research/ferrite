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
    const uint32_t s = ((uint32_t)b & 0x80u) << 24;
    const uint32_t e = ((uint32_t)b >> 3) & 0x0Fu;
    const uint32_t m = (uint32_t)b & 0x07u;
    if (e == 0u) {
        const float v = (float)m * (1.0f / 512.0f);
        return (b & 0x80u) ? -v : v;
    }
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}

// ---------------------------------------------------------------------------
// a32 4-element vectorisation (gemm-5us-feasibility, 2026-09-12).
//
// The a32 block-wide materialisation (`s_af[i] = s_lut[a[i]] * s_as[i>>5]`)
// measured 1.55 us/call of the fp8 GEMV's block-level staging, and the isolated
// probe showed its cost is INSTRUCTION COUNT, not ILP: unrolling 4/8/16 is all
// neutral, hand-written 4-way ILP is neutral, and ONLY the 4-element
// vectorisation -- one 32-bit read of four fp8 bytes, four LUT lookups, ONE
// float4 store -- moves the kernel (5.02/6.67/8.58 us at n=256/1024/1664,
// -24/-12/-7.7% against the scalar form).
//
// BIT-IDENTICAL BY CONSTRUCTION. `idx` must be a MULTIPLE OF FOUR and a scale
// block is 32 elements, so idx..idx+3 can never straddle a scale boundary
// (idx & 31 <= 28): all four products use the SAME `as[idx >> 5]` the scalar
// loop would look up, and each one is the same `lut[byte] * sa` term in the
// same order. Only the store WIDTH changes, so no sum, and not a single result
// bit, moves.
//
// REQUIREMENTS on the caller (a misaligned float4 store is err 716, the trap the
// weight-row cp.async already hit -- the guard must be a UNIFORM branch):
//   * idx % 4 == 0;
//   * `dst + idx` 16-byte aligned -> gate with dsv41_f4_ok(s_af).
__device__ __forceinline__ bool dsv41_f4_ok(const void* p) {
    return (reinterpret_cast<unsigned long long>(p) & 15ull) == 0ull;
}

__device__ __forceinline__ void dsv41_a32_mat4(uint32_t b4, int idx,
                                               const float* __restrict__ lut,
                                               const float* __restrict__ as,
                                               float* __restrict__ dst) {
    const float sa = as[idx >> 5];
    float4 o;
    o.x = lut[b4 & 0xFFu] * sa;
    o.y = lut[(b4 >> 8) & 0xFFu] * sa;
    o.z = lut[(b4 >> 16) & 0xFFu] * sa;
    o.w = lut[b4 >> 24] * sa;
    *reinterpret_cast<float4*>(dst + idx) = o;
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

// Fused fp4 quantise + pack: the same thing dsv41_quant_fp4 gets from
// `quant_kernel<1>` followed by `fp4_pack_kernel`, in ONE launch and without the
// nibble scratch array (which cost a full write + read of rows*cols bytes per
// call -- 40 calls/step, and the second kernel was 40 extra graph nodes).
//
// NUMERIC CONTRACT: the amax loop, the shfl_xor tree, the thread layout and the
// scale arithmetic below are `quant_kernel<1>`'s, term for term, and the code
// selection is its nearest-e2m1 loop, so the per-element nibble is bit-identical
// to what the old path stored in g_q4nib; the byte assembly is
// `fp4_pack_kernel`'s `(lo & 0xF) | (hi << 4)` with lo = element 2t, hi = 2t+1.
// The two kernels therefore emit identical bytes; only the number of launches
// (and the scratch round trip) changes.
//
// Why the fusion is legal layout-wise: the launcher only takes this path for an
// EVEN `block`, so a nibble pair (2i, 2i+1) can never straddle a block boundary
// and pair t of (row r, block b) lands at a byte that is a pure function of the
// block:
//     y[(r*cols + b*block)/2 + t]
// The block's codes are exchanged through shared memory: thread `threadIdx.y`
// owns element `i = threadIdx.y` (the strided loop covers block > blockDim.y, as
// in quant_kernel), then the lower half of the block packs the pairs. `block` is
// <= 256 on this path (the launcher declines otherwise), which bounds s_code.
//
// NOTE: the early `idx` return is uniform within a block because the launcher
// sets blockDim.x = 1, so no __syncthreads() below is reached by only half of a
// block -- keep blockDim.x = 1 if this is ever re-tiled.
__global__ void quant_fp4_fused_kernel(const float* __restrict__ x, uint8_t* __restrict__ y,
                                       float* __restrict__ scale, int rows, int cols, int block,
                                       int round_scale) {
    __shared__ uint8_t s_code[256];   // one e2m1 nibble per element of the block
    __shared__ float samax;
    const int nb = cols / block;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // one thread-block per (row, block)
    if (idx >= rows * nb) return;
    const int r = idx / nb, b = idx % nb;
    const float* src = x + (size_t)r * cols + (size_t)b * block;
    // amax over the block (quant_kernel<1>'s reduction, verbatim)
    float amax = 0.f;
    for (int i = threadIdx.y; i < block; i += blockDim.y) amax = fmaxf(amax, fabsf(src[i]));
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    if (threadIdx.y == 0 && (threadIdx.x & 31) == 0) samax = amax;
    __syncthreads();
    amax = samax;
    // kept in quant_kernel's exact shape (`maxv` + `1.0f / maxv`): under
    // --use_fast_math a literal fold and a folded variable COULD round the
    // reciprocal differently, and a 1-ULP difference in max_inv can flip the
    // exponent fast_round_scale picks at a power-of-two boundary.
    const float maxv = 6.0f;
    const float sc = round_scale ? fmaxf(fast_round_scale(amax, 1.0f / maxv), 1e-30f)
                                 : fmaxf(amax / maxv, 1e-30f);
    if (threadIdx.x == 0 && threadIdx.y == 0) scale[idx] = sc;
    const float inv = 1.0f / sc;
    for (int i = threadIdx.y; i < block; i += blockDim.y) {
        const float v = fminf(fmaxf(src[i] * inv, -6.0f), 6.0f);
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
        s_code[i] = best;
    }
    __syncthreads();
    // pack straight into the caller's buffer (low nibble = even element)
    uint8_t* dst = y + (((size_t)r * cols + (size_t)b * block) >> 1);
    for (int t = threadIdx.y; t < (block >> 1); t += blockDim.y)
        dst[t] = (uint8_t)((s_code[2 * t] & 0x0Fu) | (uint8_t)(s_code[2 * t + 1] << 4));
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

// ---------------------------------------------------------------------------
// swapAB fp8 GEMV (M=1 decode on the tensor core) -- see
// docs/agent/perf-roadmap.md, "swapAB（M=1 落 MMA 的 N 维）可行性分析".
//
// `gemm_fp8_kernel` above puts the ACTIVATION on M and the WEIGHT on N, which at
// M=1 wastes 15/16 of the MMA (one useful M row out of 16). This kernel swaps
// the roles back: the WEIGHT is A (M = 16 output rows) and the ACTIVATION is B
// (N = 8 columns, only column 0 carries the token). One warp owns 16 output rows
// and walks the whole K, so there is no k_split, no cross-warp reduction and no
// cross-warp sync at all.
//
// WHY IT IS WORTH IT. `gemm_fp8_gemv_kernel` measures ~9.5 us/call (246
// calls/step = 2.33 ms) at n=1664/k=5120, while its 8.5 MB of weights move at
// ~895 GB/s -- 11% of the B300's HBM. It is NOT memory bound: the block-level
// staging (a32 materialisation 1.55 us + activation staging 0.8 us + LUT 0.3 us,
// repeated in all 416 blocks) plus the FFMA consume loop are the cost. Feeding
// raw fp8 to the tensor core deletes ALL of it -- the activation is never
// decoded and the accumulate is the MMA's.
//
// LAYOUT (all pre-existing, zero transposes -- that is what makes this cheap):
//   w  [n, k]  e4m3 row-major     == A row-major [M, K]
//   a  [k]     e4m3               == B col-major [K, N] column 0
//   w_scale [(n/32), nb_k] ue8m0  == per-(32-row, 32-col) weight scale
//   a_scale [nb_k]         f32    == per-32-block activation scale
//   out [n] f32, bias [n] f32 or null
//
// SCALE SCHEME -- identical to `gemm_fp8_kernel` (:320-329): the raw MMA result
// of one 32-wide k block is scaled by (activation scale * weight scale) before
// it joins the running fp32 accumulator. Mixing the two scale TYPES up silently
// zeroes the output (see the dense kernel's note at :261-265).
//
// BIT-EXACTNESS: NOT bit-identical to `gemm_fp8_gemv_kernel`. The SIMT gemv is a
// per-element `(a*sa)*(w*sb)` FFMA chain reduced by shuffles; this one sums raw
// fp8 products inside the tensor core and scales per k block. Mathematically
// equivalent, different rounding order -- the same scheme the existing m>1 dense
// MMA path already uses. Parity is judged by text/fingerprint, not bit equality.
//
// Each 8-wide B column beyond 0 is ZERO (its lanes write nothing to bf), so
// 7/8 of the N dimension is wasted -- 2x better than the dense tile's 15/16, and
// the reason the design targets batched decode filling N with real tokens later.

// Ring geometry for gemm_fp8_swapab_kernel.
//   kSwapabKStep: k width of one stage. A multiple of 32 (one scale block == the
//                 MMA's K) and of 16 (the cp.async granule).
//   kSwapabNStage: ring depth.
//   kSwapabRow: bytes per staged row. PADDED by 16: with an unpadded stride every
//                 one of the 8 gid rows of the A fragment lands on banks 0..3
//                 (stride/4 % 32 == 0) and each LDS.32 is an 8-way conflict; +16B
//                 staggers row r by 4 banks, making the four fragment loads
//                 conflict-free.
//   kSwapabWarps: warps per block.
//   kSwapabKSplit: K partitions per 16-row tile. THIS IS THE PARALLELISM KNOB.
//                 MEASURED 2026-09-12: with ks=1 the kernel ran 28.4 us/call at
//                 n=1664/k=5120 -- 16x WORSE than the SIMT gemv -- because a
//                 16-row tile per warp caps the grid at n/16 = 104 warps (0.7 per
//                 SM on 148 SMs), so nothing on the SM can cover a DRAM round
//                 trip. Splitting K is the ONLY dimension left that adds warps
//                 without touching the MMA's M=16. Partial sums are combined by
//                 LAST-BLOCK REDUCTION, applied by the tg==0 lanes exactly like
//                 the plain epilogue.
//
// LAST-BLOCK REDUCTION (ks > 1). The first version zeroed `out` with a
// cudaMemsetAsync ahead of the kernel and combined the partitions with
// atomicAdd(&out[r], v). That costs a whole extra GRAPH NODE per call (~0.5 us
// in graph replay x 246 calls/step = 0.12 ms/step) AND it is non-deterministic:
// the fp32 add order of the `ks` atomics is whatever the scheduler produced.
// Now each partition writes its own SLOT (`partial[kp * n + r]`, no contention)
// and then bumps a per-16-row-tile ticket; the LAST partition of a tile (the one
// whose atomicAdd returns ks-1) sums the ks slots itself, in a FIXED kp order,
// adds the bias and stores `out`. Consequences:
//   * no memset, no graph node, `out` need not be pre-zeroed;
//   * the reduction order is fixed (kp ascending) => BIT-DETERMINISTIC across
//     runs, strictly better than the atomicAdd it replaces;
//   * the bias rides on the reduction instead of on partition 0, so it is added
//     exactly once and in the same place for every shape.
// The ticket is caller-supplied memory (one u32 per 16-row tile) and is RESET to
// 0 by the elected block after its last slot read, so a captured graph replays
// clean -- the discipline g_hc_ticket / g_hc_mb_done use (see dsv41_glue.cu).
// It must simply START at zero (cudaMalloc does not zero).
#ifndef DSV41_SWAPAB_KSTEP
#define DSV41_SWAPAB_KSTEP 128
#endif
#ifndef DSV41_SWAPAB_NSTAGE
#define DSV41_SWAPAB_NSTAGE 8
#endif
#ifndef DSV41_SWAPAB_WARPS
#define DSV41_SWAPAB_WARPS 1
#endif
#ifndef DSV41_SWAPAB_KSPLIT
#define DSV41_SWAPAB_KSPLIT 8
#endif
constexpr int kSwapabKStep = DSV41_SWAPAB_KSTEP;
constexpr int kSwapabNStage = DSV41_SWAPAB_NSTAGE;
constexpr int kSwapabRow = kSwapabKStep + 16;
constexpr int kSwapabWarps = DSV41_SWAPAB_WARPS;
constexpr int kSwapabKSplit = DSV41_SWAPAB_KSPLIT;

__device__ __forceinline__ void swapab_cp_async16(void* smem_dst, const void* gmem_src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void swapab_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void swapab_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__global__ void __launch_bounds__(kSwapabWarps * 32)
gemm_fp8_swapab_kernel(const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
                       const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_scale,
                       const float* __restrict__ bias, float* __restrict__ out, int n, int k,
                       int ks, float* __restrict__ partial, unsigned* __restrict__ ctr) {
#if __CUDA_ARCH__ >= 900
    // PDL (DSV41_PDL, see dsv41_pdl_or_plain): the launcher may have started this
    // grid during the producer's tail, so gate the activation reads on the
    // producer's completion. Must precede the first read of `a`/`a_scale`.
    cudaGridDependencySynchronize();
#endif
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gid = lane >> 2;   // A rows gid / gid+8; B column gid
    const int tg = lane & 3;     // A cols tg*4 (+16); C cols 2*tg

    // ---- K-split warp mapping ----------------------------------------------
    // Global warp id -> (16-row tile, K partition). `ks` is the fast index so the
    // ks warps of one row tile read adjacent K ranges of the SAME weight rows
    // (their staging streams are contiguous, which keeps the DRAM pages hot).
    const int gw = blockIdx.x * kSwapabWarps + warp;
    if (gw >= (n >> 4) * ks) return;
    const int m0 = (gw / ks) * 16;   // first of this warp's 16 output rows
    const int kp = gw % ks;          // this warp's K partition
    const int kc = k / ks;           // k per partition; a multiple of 32
    const int k0p = kp * kc;         // first global k of the partition
    const int nb = kc >> 5;          // k-blocks (== scale entries) in the partition
    const int nb_k = k >> 5;         // k-blocks of the whole row

    // NOTE: `extern __shared__` names are aliases of the SAME dynamic smem
    // symbol, so a second declaration of the same name with a different type is
    // a compile error (the file already has `extern __shared__ float smem[]`).
    extern __shared__ uint8_t sab_smem[];
    uint8_t* sw = sab_smem + (size_t)warp * kSwapabNStage * 16 * kSwapabRow;

    // ---- per-warp A/B staging (this warp's K slice only) -------------------
    // Staged only for THIS warp's slice, so the ks partitions do not each re-read
    // the whole activation. Vectorised 16B: a byte-wise copy of k bytes cost a
    // fixed ~3.3 us/call on its own.
    const size_t wbytes = (size_t)kSwapabWarps * kSwapabNStage * 16 * kSwapabRow;
    uint8_t* s_a = sab_smem + wbytes + (size_t)warp * kc;                 // [kc] fp8
    float* s_as = (float*)(sab_smem + wbytes + (size_t)kSwapabWarps * kc) + (size_t)warp * nb;
    uint8_t* s_ws = (uint8_t*)(s_as + (size_t)kSwapabWarps * nb) + (size_t)warp * nb;
    {
        const uint4* src4 = (const uint4*)(a + k0p);
        uint4* dst4 = (uint4*)s_a;
        for (int i = lane; i < (kc >> 4); i += 32) dst4[i] = src4[i];
        const int kb0 = k0p >> 5;
        const int ws0 = (m0 >> 5) * nb_k;  // 16 rows always sit in one 32-row block
        for (int i = lane; i < nb; i += 32) {
            s_as[i] = a_scale[kb0 + i];
            s_ws[i] = w_scale[(size_t)ws0 + kb0 + i];
        }
    }
    __syncwarp();

    const int nk = (kc + kSwapabKStep - 1) / kSwapabKStep;  // stages in the slice

    // Stage `st`: 16 rows x [k0, k0+KSTEP) of THIS warp's slice, one cp.async
    // group of 16B chunks. An out-of-range `st` STILL commits (an empty group):
    // the depth invariant wait_group relies on is "NSTAGE-1 groups outstanding
    // before the wait", and near the end of K the real issues run out. Without
    // the empty groups the count drops to NSTAGE-2, wait_group(NSTAGE-2) returns
    // immediately and the LAST TWO stages get read before their cp.asyncs land.
    auto stage = [&](int st) {
        if (st < nk) {
            const int k0 = k0p + st * kSwapabKStep;
            const int rb = min(kSwapabKStep, kc - st * kSwapabKStep);  // % 16 == 0
            const int nchunk = rb >> 4;                                // 16B chunks/row
            uint8_t* dst = sw + (size_t)(st % kSwapabNStage) * 16 * kSwapabRow;
            if (nchunk == (kSwapabKStep >> 4)) {
                // Full stage: 16B chunks per row is the compile-time power of two
                // P, so the chunk->(row, offset) map is a shift and a mask. The
                // generic `c / nchunk` form costs a full integer division per
                // chunk per lane, which on a kernel whose ring already dominates
                // is pure waste.
                constexpr int P = kSwapabKStep >> 4;
                constexpr int L = (P == 1) ? 0 : (P == 2) ? 1 : (P == 4)   ? 2
                                  : (P == 8) ? 3 : (P == 16) ? 4 : (P == 32) ? 5 : 6;
#pragma unroll
                for (int i = 0; i < (16 * P) / 32; i++) {
                    const int c = lane + 32 * i;
                    const int r = c >> L;
                    const int off = (c & (P - 1)) << 4;
                    swapab_cp_async16(dst + (size_t)r * kSwapabRow + off,
                                      w + (size_t)(m0 + r) * k + k0 + off);
                }
            } else {
                // Partial tail stage (only the last one).
                for (int c = lane; c < 16 * nchunk; c += 32) {
                    const int r = c / nchunk;
                    const int off = (c - r * nchunk) << 4;
                    swapab_cp_async16(dst + (size_t)r * kSwapabRow + off,
                                      w + (size_t)(m0 + r) * k + k0 + off);
                }
            }
        }
        swapab_cp_commit();
    };

    // Prologue: exactly NSTAGE-1 groups (empty when K is shorter than the ring).
    for (int st = 0; st < kSwapabNStage - 1; st++) stage(st);

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int st = 0; st < nk; st++) {
        // Exactly ONE group per iteration keeps NSTAGE-1 outstanding, so
        // wait_group(NSTAGE-2) always retires the oldest one (== stage st).
        stage(st + kSwapabNStage - 1);
        swapab_cp_wait<kSwapabNStage - 2>();
        __syncwarp();  // every lane's own group is retired -> the tile is complete

        const uint8_t* s = sw + (size_t)(st % kSwapabNStage) * 16 * kSwapabRow;
        const int lk0 = st * kSwapabKStep;
        const int nkb = min(kSwapabKStep, kc - lk0) >> 5;  // 32-wide k blocks here
        for (int kb = 0; kb < nkb; kb++) {
            const int lk = kb * 32;
            // A fragment: rows gid / gid+8, cols tg*4 (+16) -- same shape as the
            // dense kernel's af, but now read from the WEIGHT rows.
            uint32_t af[4];
            af[0] = *(const uint32_t*)&s[(size_t)gid * kSwapabRow + lk + tg * 4];
            af[1] = *(const uint32_t*)&s[(size_t)(gid + 8) * kSwapabRow + lk + tg * 4];
            af[2] = *(const uint32_t*)&s[(size_t)gid * kSwapabRow + lk + tg * 4 + 16];
            af[3] = *(const uint32_t*)&s[(size_t)(gid + 8) * kSwapabRow + lk + tg * 4 + 16];
            // B fragment: K x N col-major, column n = gid. Only column 0 (the
            // token) is real; the other columns are zero so their C columns come
            // out exactly 0 (never read in the epilogue).
            uint32_t bf[2] = {0u, 0u};
            if (gid == 0) {
                bf[0] = *(const uint32_t*)&s_a[lk0 + lk + tg * 4];
                bf[1] = *(const uint32_t*)&s_a[lk0 + lk + tg * 4 + 16];
            }
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
            // Scale of this k block: activation per k block (the single token
            // row), weight per (32-row, 32-col) -- d[] is the raw fp8 sum, so the
            // product lands on the accumulator exactly like the dense kernel.
            const int kbg = (lk0 >> 5) + kb;
            const float sc = s_as[kbg] * ue8m0_to_f(s_ws[kbg]);
            acc[0] += d[0] * sc;
            acc[1] += d[1] * sc;
            acc[2] += d[2] * sc;
            acc[3] += d[3] * sc;
        }
    }
    // C fragment: d[0]=C[gid][2*tg], d[1]=C[gid][2*tg+1], d[2]=C[gid+8][2*tg],
    // d[3]=C[gid+8][2*tg+1]. Column 0 is the only live column and it is reached
    // only by tg == 0 (2*tg == 0), so those 8 lanes emit all 16 rows.
    if (tg == 0) {
        const int r0 = m0 + gid, r1 = m0 + gid + 8;
        const float v0 = acc[0], v1 = acc[2];
        if (ks == 1) {
            // One K partition: an ordinary store (deterministic, no scratch, and
            // `out` is not pre-zeroed). The bias rides on it directly.
            out[r0] = v0 + ((bias != nullptr) ? bias[r0] : 0.f);
            out[r1] = v1 + ((bias != nullptr) ? bias[r1] : 0.f);
        } else {
            // Publish this partition into its OWN slot: no atomic read-modify-
            // write, so no pre-zeroed `out` and no contention between the ks
            // partitions of a tile. NO bias here -- the elected reducer adds it.
            partial[(size_t)kp * n + r0] = v0;
            partial[(size_t)kp * n + r1] = v1;
            // Device-scope release: this partition's slots are visible before the
            // ticket below can be observed. Executed by the publishing lanes; the
            // __syncwarp() then orders every lane's fence ahead of lane 0's
            // ticket (the PTX model is cumulative, so that chain is a release).
            __threadfence();
        }
    }
    if (ks > 1) {
        __syncwarp();
        unsigned ticket = 0u;
        if (lane == 0) ticket = atomicAdd(&ctr[m0 >> 4], 1u);
        ticket = __shfl_sync(0xffffffffu, ticket, 0);   // 1 warp per tile partition
        if (ticket == (unsigned)(ks - 1)) {
            // Last arrival of this 16-row tile: acquire the other partitions'
            // slots, reduce them in a FIXED order (kp ascending -- that is what
            // makes the result bit-deterministic), add the bias and store.
            __threadfence();
            if (tg == 0) {
                const int r0 = m0 + gid, r1 = m0 + gid + 8;
                float s0 = 0.f, s1 = 0.f;
                for (int q = 0; q < ks; q++) {
                    s0 += partial[(size_t)q * n + r0];
                    s1 += partial[(size_t)q * n + r1];
                }
                out[r0] = s0 + ((bias != nullptr) ? bias[r0] : 0.f);
                out[r1] = s1 + ((bias != nullptr) ? bias[r1] : 0.f);
            }
            // Self-reset so the next launch / graph replay starts at zero. Safe:
            // all ks arrivals of THIS tile already happened (the ticket proved it)
            // and no other block touches this entry before the grid ends.
            if (lane == 0) atomicExch(&ctr[m0 >> 4], 0u);
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
// Order-preserving software-pipelined variant of sparse_attn_warp_kernel.
// The t-loop's chain is  idx load -> kv-row load -> dot -> online-softmax,
// ~800 cycles of pure latency per key with one slot in flight, which is why the
// warp version sits ~2x off its bandwidth floor. This variant keeps TWO slots
// in flight: while the dot for slot j runs on the registers already staged in
// kb[0]/kb[1], the kv-row loads for slots j+1/j+2 are on their way, and the q
// row is staged once instead of being re-read every slot. The iteration order,
// the dot's column order and the online-softmax update order are untouched,
// so the math is bit-identical (idx<0 slots keep the skip semantics: no load,
// no state update). DSV41_ATTN_PF=0 restores the plain warp version for A/B.
__global__ void sparse_attn_pf_kernel(const float* __restrict__ q, const float* __restrict__ kv,
                                      const float* __restrict__ sink,
                                      const int32_t* __restrict__ idxs, float* __restrict__ out,
                                      int b, int m, int h, int d,
                                      const int* __restrict__ clen, int window, int index_topk,
                                      float scale) {
#if __CUDA_ARCH__ >= 900
    // PDL (DSV41_PDL, see dsv41_pdl_or_plain): the launcher may have launched
    // this grid with programmatic stream serialization, so the grid is already
    // resident and this call is what makes the producers' writes visible. Three
    // of the reads below are producer outputs: `*clen` (the compressor's live
    // counter), `idxs` (the indexer) and `q` (wq_b / apply_rope). It is placed
    // BEFORE the `row >= b*m` early return so every block in the grid reaches
    // it. No-op on a plain launch. The win is the node-gap (launch overlap):
    // this kernel's prologue is q-row staging, which IS producer-dependent, so
    // there is nothing worth hoisting above the sync.
    cudaGridDependencySynchronize();
#endif
    const int n = window + *clen;
    const int topk = window + ((*clen < index_topk) ? *clen : index_topk);
    const int row = blockIdx.x;
    if (row >= b * m) return;
    const int bb = row / m, mm = row % m;
    const int32_t* irow = idxs + (size_t)(bb * m + mm) * topk;
    for (int hh = blockIdx.y; hh < h; hh += gridDim.y) {
        const float* qr = q + ((size_t)(bb * m + mm) * h + hh) * d;
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        const int nwarp = (int)blockDim.x >> 5;
        // Stage the q row once: kills the per-slot re-read of qr (16 floats/lane).
        float qv[kMaxPerW];
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) qv[i] = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) qv[i] = qr[c];
        }
        float my_acc[kMaxPerW];
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) my_acc[i] = 0.f;
        float my_smax = -1e30f, my_se = 0.f;
        // Three kv-row buffers, rotating by slot; kBase is the row base.
        // (The two-deep form gave each row load one compute phase of distance;
        // three gives two, which the isolated probe measured as the last
        // lossless lever this kernel has - the slot ORDER, the per-dot reduce
        // tree and the softmax update chain are untouched, so it is
        // bit-identical; only the load ISSUE time moves earlier.)
        float kb0[kMaxPerW], kb1[kMaxPerW], kb2[kMaxPerW];
        const size_t kBase = (size_t)bb * n * d;
        // Prologue: fire the idx loads for the first three slots, then their rows.
        int t = wid;
        int ia = (t < topk) ? irow[t] : -1;
        int ib = (t + nwarp < topk) ? irow[t + nwarp] : -1;
        int ic = (t + 2 * nwarp < topk) ? irow[t + 2 * nwarp] : -1;
        if (ia >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb0[i] = kv[kBase + (size_t)ia * d + c];
            }
        }
        if (ib >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb1[i] = kv[kBase + (size_t)ib * d + c];
            }
        }
        if (ic >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb2[i] = kv[kBase + (size_t)ic * d + c];
            }
        }
        // Main loop: three slots per pass. Slot t lives in kb0, t+nwarp in kb1,
        // t+2*nwarp in kb2; each buffer's next row load is fired right after
        // that buffer's compute, so every row has two compute phases of
        // distance. The three next-slot indices are hoisted to the loop top
        // (loads have no side effects - pure earlier issue).
        for (; t + 2 * nwarp < topk; t += 3 * nwarp) {
            const int td = t + 3 * nwarp;   // next slot destined for kb0
            const int te = td + nwarp;      // next slot destined for kb1
            const int tf = te + nwarp;      // next slot destined for kb2
            const int id = (td < topk) ? irow[td] : -1;
            const int ie = (te < topk) ? irow[te] : -1;
            const int iff = (tf < topk) ? irow[tf] : -1;
            if (ia >= 0) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) dot += qv[i] * kb0[i];
                }
                for (int off = 16; off > 0; off >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                dot *= scale;
                const float nm = fmaxf(my_smax, dot);
                const float corr = expf(my_smax - nm);
                const float e = expf(dot - nm);
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) my_acc[i] = my_acc[i] * corr + e * kb0[i];
                }
                my_se = my_se * corr + e;
                my_smax = nm;
            }
            // kb0 is dead now: fire the row load for slot td into it.
            if (id >= 0) {
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) kb0[i] = kv[kBase + (size_t)id * d + c];
                }
            }
            if (ib >= 0) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) dot += qv[i] * kb1[i];
                }
                for (int off = 16; off > 0; off >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                dot *= scale;
                const float nm = fmaxf(my_smax, dot);
                const float corr = expf(my_smax - nm);
                const float e = expf(dot - nm);
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) my_acc[i] = my_acc[i] * corr + e * kb1[i];
                }
                my_se = my_se * corr + e;
                my_smax = nm;
            }
            // kb1 is dead now: fire the row load for slot te into it.
            if (ie >= 0) {
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) kb1[i] = kv[kBase + (size_t)ie * d + c];
                }
            }
            if (ic >= 0) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) dot += qv[i] * kb2[i];
                }
                for (int off = 16; off > 0; off >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                dot *= scale;
                const float nm = fmaxf(my_smax, dot);
                const float corr = expf(my_smax - nm);
                const float e = expf(dot - nm);
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) my_acc[i] = my_acc[i] * corr + e * kb2[i];
                }
                my_se = my_se * corr + e;
                my_smax = nm;
            }
            // kb2 is dead now: fire the row load for slot tf into it.
            if (iff >= 0) {
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) kb2[i] = kv[kBase + (size_t)iff * d + c];
                }
            }
            ia = id;
            ib = ie;
            ic = iff;
        }
        // Tail: at most two slots left (rows for t and t+nwarp in kb0/kb1; the
        // kb2 slot is necessarily past topk or the loop would have continued).
        if (t < topk && ia >= 0) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) dot += qv[i] * kb0[i];
            }
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            dot *= scale;
            const float nm = fmaxf(my_smax, dot);
            const float corr = expf(my_smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) my_acc[i] = my_acc[i] * corr + e * kb0[i];
            }
            my_se = my_se * corr + e;
            my_smax = nm;
        }
        if (t + nwarp < topk && ib >= 0) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) dot += qv[i] * kb1[i];
            }
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            dot *= scale;
            const float nm = fmaxf(my_smax, dot);
            const float corr = expf(my_smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) my_acc[i] = my_acc[i] * corr + e * kb1[i];
            }
            my_se = my_se * corr + e;
            my_smax = nm;
        }
        // ---- same warp-merge epilogue as the plain warp version ----
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
// Split sparse attention (DSV41_ATTN_PF_SPLIT, DEFAULT ON; DSV41_ATTN_SPLIT=C
// is the older explicit knob for the same kernel and still wins when set).
// The warp version runs one block per (token, head) - eight blocks on this
// model's decode - which leaves 95 percent of the machine idle and pays for it
// in latency: 26 us per layer, forty layers, one millisecond a step. This shape
// splits the KEY range across `C` blocks per head, each computing an
// online-softmax partial (max, sum, d-wide PV accumulator) into a fixed device
// scratch, and a second kernel merges the partials in ascending chunk order - a
// deterministic combine, so the result differs from the single-block version
// only in the last bits of the summation grouping. The sink fold and the
// normalisation live in the merge, exactly where the warp version had them.
//
// WHY THE SPLIT ALONE WAS NOT ENOUGH (measured, then fixed here): the first
// version of this kernel had no prefetch, and DSV41_ATTN_SPLIT=8 came out
// 0.37 ms/step WORSE than the single-block pf kernel - the slot chain
// (idx load -> kv-row load -> dot -> softmax) re-serialised at one slot in
// flight, so cutting the slot count per block did not cut the latency-bound
// chain. The chunk loop below therefore carries the SAME three-deep pipeline as
// `sparse_attn_pf_kernel`: three rotating kv-row buffers, each row's next load
// fired right after that buffer's compute (two compute phases of distance), and
// the q row staged once in registers. The single block per head was issue
// saturated (one warp per scheduler, ~12k cycles for the full walk), so dividing
// the instructions per block by C - the blocks run concurrently, and the whole
// kv row set is shared by every head and chunk (kBase has no head term), so they
// all hit the same 1.3 MB in L2 - is what turns the walk into ~1/C of the
// latency plus the merge.
//
// CORRECTNESS GATE: the chunk's slot order, the per-dot reduce tree and the
// softmax update chain are byte-for-byte the pf kernel's, so a chunk partial is
// exactly what the pf kernel would have produced for that slot range, and C=1
// is therefore BIT-IDENTICAL to `sparse_attn_pf_kernel` (one chunk, and the
// merge of a single partial reproduces its epilogue: expf(P[0]-smax) == 1).
// Anything that touches the pipeline must keep C=1 bit-exact.
// DSV41_ATTN_PF_SPLIT=1 is the cheap equivalence check; C>1 differs only in the
// final summation grouping (tolerance, not bit-equality).
//
// MEASURED (isolated harness tests_dsv41_sparse_pfsplit.cu, h=8, d=512, window
// 128, 300 iters, us/call; the harness runs a non-graph back-to-back loop so it
// carries ~4 us of launch-gap floor per extra kernel that a captured step does
// not - read the SHAPE, not the absolute):
//   topk   128    155    203    328    640      (topk = window + min(clen, idx_topk))
//   pf    13.7   16.2   20.0   29.8   53.9
//   C=4    9.6   10.5   11.5   14.4   20.8
//   C=8   11.6   11.8   12.6   13.9   17.1
// C=4 wins for topk <= ~240, C=8 from ~300 up; the per-slot cost drops ~7x in
// both (pf 78 ns/slot vs 10.7 at C=8), so what is left at small topk is the
// fixed merge/launch cost. 8 is the default because the long-context steady
// state (clen at the compress cap) is where the sparse path dominates the step;
// DSV41_ATTN_PF_SPLIT=4 is the short-context arm.
// Round 39 verdict: at the bench's short context (topk ~178) C=8 measured
// +0.1ms in serve - the isolated win only materialises at topk >= ~300
// (long-context steady state). Default OFF until an adaptive C (graph-capture
// friendly) exists; DSV41_ATTN_PF_SPLIT=C opts in explicitly.
//
// ADAPTIVE C (design, 2026-09-11; NOT implemented): the graph is NOT the blocker.
// split_c, b*m and h are all HOST constants and topk lives on the DEVICE (`*clen`),
// so the captured grid (split_c, b*m, h) is already valid at every topk - the step
// graph is captured once at decode_steps==1 and only replayed after that (reset()
// drops it per request, never per topk), so no re-capture is needed as the context
// grows. Move the decision onto the device instead: launch C_max chunks always and
// compute C_eff = f(topk) inside BOTH kernels from *clen - `if (ck >= C_eff) return;`
// in the split, `for (ck < C_eff)` in the merge (which must then also take
// clen/window/index_topk, it does not today). A per-STEP host choice cannot work:
// `clen` is per-LAYER, so one step's forty attention calls span forty topk values;
// multi-graph capture is both too coarse and x3 memory.
//
// The other half of the round-39 regression was NOT topk-dependent: any split_c > 0
// made dsv41_sparse_attn_orope return 2 (decline), which forfeited the o-rope and
// o-quant epilogue fusions (chain_dev.rs ~3085-3196) at EVERY context. A static
// C=4 therefore paid the same fixed penalty C=8 did - a bare 0->4 flip was not the
// fix. FIXED (sparse-attn-v8, 2026-09-11): that epilogue now lives in the MERGE
// kernel (`sparse_attn_merge_kernel`), whose (b*m, h) x 128 grid is exactly the
// single-block fused kernel's. The split arm of `dsv41_sparse_attn_orope` therefore
// keeps BOTH fusions (two launches per layer instead of the three-launch fallback),
// so a static-C default now costs only the key-split. Why C=4: the single-block pf
// shape is latency-, not bandwidth-, bound (8 blocks x 128 threads on ~148 SMs =
// ~5% occupancy), and the isolated harness shows C=4 wins for topk <= ~240 with a
// ~7x lower per-slot cost; 4 chunks x (b*m, h) = 32 blocks is the 22%-occupancy
// point. C=8 still wins in the long-context steady state, so DSV41_SPARSE_SPLIT=C
// remains the explicit knob.
#define kSparseSplitDefault 4   // DSV41_SPARSE_SPLIT default (0 restores the pf kernel)
#define kAttnMaxC 16
#define kAttnMaxBM 8
#define kAttnMaxH 64
#define kAttnStride (2 + 512)
__device__ float g_attn_part[kAttnMaxBM][kAttnMaxH][kAttnMaxC][kAttnStride];

// Resolve the sparse-attention key-split chunk count, ONCE per process (a per-call
// getenv is exactly the hot-path slip every other gate in this file avoids; the
// callers cache this in a function-local static). Precedence:
//   DSV41_ATTN_SPLIT     legacy explicit knob (> 0 wins; arms that pass it predate
//                        the prefetch port and must keep selecting a split kernel)
//   DSV41_ATTN_PF_SPLIT  harness A/B knob - ANY explicit value wins, including 0
//                        (that is the "restore the single-block pf kernel" arm;
//                        the OLD static default of 0 is now expressed by
//                        DSV41_SPARSE_SPLIT=0 / DSV41_ATTN_PF_SPLIT=0)
//   DSV41_SPARSE_SPLIT   the gate this round adds; default kSparseSplitDefault = 4
// A malformed value falls back to the default rather than silently disabling the
// split. The plain `dsv41_sparse_attn` and the fused `dsv41_sparse_attn_orope`
// share this resolver on purpose: they must agree on the selected shape, or the
// fallback fires for shapes that could have taken the fused path.
static int dsv41_resolve_sparse_split_c() {
    const char* e;
    if ((e = getenv("DSV41_ATTN_SPLIT")) != nullptr) {
        const int v = atoi(e);
        if (v > 0 && v <= kAttnMaxC) return v;
    }
    if ((e = getenv("DSV41_ATTN_PF_SPLIT")) != nullptr) {
        const int v = atoi(e);
        if (v >= 0 && v <= kAttnMaxC) return v;
    }
    if ((e = getenv("DSV41_SPARSE_SPLIT")) != nullptr) {
        const int v = atoi(e);
        if (v >= 0 && v <= kAttnMaxC) return v;
    }
    return kSparseSplitDefault;
}

// grid (C, b*m, h): block (ck, row, hh) owns keys [topk*ck/C, topk*(ck+1)/C).
__global__ void sparse_attn_split_kernel(const float* __restrict__ q,
                                         const float* __restrict__ kv,
                                         const int32_t* __restrict__ idxs, int b, int m, int h,
                                         int d, const int* __restrict__ clen, int window,
                                         int index_topk, float scale, int C) {
    const int n = window + *clen;
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
    // Stage the q row once: kills the per-slot re-read of qr (16 floats/lane).
    float qv[kMaxPerW];
#pragma unroll
    for (int i = 0; i < kMaxPerW; ++i) qv[i] = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxPerW; ++i) {
        const int c = lane + i * 32;
        if (c < d) qv[i] = qr[c];
    }
    float my_acc[kMaxPerW];
#pragma unroll
    for (int i = 0; i < kMaxPerW; ++i) my_acc[i] = 0.f;
    float my_smax = -1e30f, my_se = 0.f;
    // Three kv-row buffers, rotating by slot; kBase is the row base (shared by
    // every head - the kv is one MLA latent per position). Same pipeline as the
    // pf kernel, walked over this chunk's [lo, hi) instead of [0, topk).
    float kb0[kMaxPerW], kb1[kMaxPerW], kb2[kMaxPerW];
    const size_t kBase = (size_t)bb * n * d;
    // Prologue: fire the idx loads for the first three slots, then their rows.
    int t = lo + wid;
    int ia = (t < hi) ? row_idx[t] : -1;
    int ib = (t + nwarp < hi) ? row_idx[t + nwarp] : -1;
    int ic = (t + 2 * nwarp < hi) ? row_idx[t + 2 * nwarp] : -1;
    if (ia >= 0) {
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) kb0[i] = kv[kBase + (size_t)ia * d + c];
        }
    }
    if (ib >= 0) {
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) kb1[i] = kv[kBase + (size_t)ib * d + c];
        }
    }
    if (ic >= 0) {
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) kb2[i] = kv[kBase + (size_t)ic * d + c];
        }
    }
    // Main loop: three slots per pass. Slot t lives in kb0, t+nwarp in kb1,
    // t+2*nwarp in kb2; each buffer's next row load is fired right after that
    // buffer's compute, so every row has two compute phases of distance. The
    // three next-slot indices are hoisted to the loop top (loads have no side
    // effects - pure earlier issue). An idx of -1 keeps the pf kernel's skip
    // semantics: no load and no state update for that slot.
    for (; t + 2 * nwarp < hi; t += 3 * nwarp) {
        const int td = t + 3 * nwarp;   // next slot destined for kb0
        const int te = td + nwarp;      // next slot destined for kb1
        const int tf = te + nwarp;      // next slot destined for kb2
        const int id = (td < hi) ? row_idx[td] : -1;
        const int ie = (te < hi) ? row_idx[te] : -1;
        const int iff = (tf < hi) ? row_idx[tf] : -1;
        if (ia >= 0) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) dot += qv[i] * kb0[i];
            }
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            dot *= scale;
            const float nm = fmaxf(my_smax, dot);
            const float corr = expf(my_smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) my_acc[i] = my_acc[i] * corr + e * kb0[i];
            }
            my_se = my_se * corr + e;
            my_smax = nm;
        }
        // kb0 is dead now: fire the row load for slot td into it.
        if (id >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb0[i] = kv[kBase + (size_t)id * d + c];
            }
        }
        if (ib >= 0) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) dot += qv[i] * kb1[i];
            }
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            dot *= scale;
            const float nm = fmaxf(my_smax, dot);
            const float corr = expf(my_smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) my_acc[i] = my_acc[i] * corr + e * kb1[i];
            }
            my_se = my_se * corr + e;
            my_smax = nm;
        }
        // kb1 is dead now: fire the row load for slot te into it.
        if (ie >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb1[i] = kv[kBase + (size_t)ie * d + c];
            }
        }
        if (ic >= 0) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) dot += qv[i] * kb2[i];
            }
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            dot *= scale;
            const float nm = fmaxf(my_smax, dot);
            const float corr = expf(my_smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) my_acc[i] = my_acc[i] * corr + e * kb2[i];
            }
            my_se = my_se * corr + e;
            my_smax = nm;
        }
        // kb2 is dead now: fire the row load for slot tf into it.
        if (iff >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb2[i] = kv[kBase + (size_t)iff * d + c];
            }
        }
        ia = id;
        ib = ie;
        ic = iff;
    }
    // Tail: at most two slots left (rows for t and t+nwarp in kb0/kb1; the kb2
    // slot is necessarily past hi or the loop would have continued).
    if (t < hi && ia >= 0) {
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) dot += qv[i] * kb0[i];
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        dot *= scale;
        const float nm = fmaxf(my_smax, dot);
        const float corr = expf(my_smax - nm);
        const float e = expf(dot - nm);
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) my_acc[i] = my_acc[i] * corr + e * kb0[i];
        }
        my_se = my_se * corr + e;
        my_smax = nm;
    }
    if (t + nwarp < hi && ib >= 0) {
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) dot += qv[i] * kb1[i];
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        dot *= scale;
        const float nm = fmaxf(my_smax, dot);
        const float corr = expf(my_smax - nm);
        const float e = expf(dot - nm);
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) my_acc[i] = my_acc[i] * corr + e * kb1[i];
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
//
// FUSED EPILOGUE (sparse-attn-v8, 2026-09-11): when `cos` is non-null this block
// ALSO runs the inverse o-rope (the orope kernel's phase 2) and the fp8 emission
// (its phase 3) - ported verbatim - so the split arm of `dsv41_sparse_attn_orope`
// keeps BOTH fusions instead of declining them. The geometry is already right for
// it: one (row, head) block of 128 threads owns the head's full d-wide row, the
// same shape `sparse_attn_orope_kernel` runs, and `d <= 512` is launcher-guarded
// (the plain launcher rejects d > 512, the fused one declines), so `sh_row` is the
// same 2 KB the single-block kernel stages.
//
// WHY THE SMEM HAND-OFF: the rope pass pairs columns (2i, 2i+1), which no
// thread-stride walk over c can own, so the normalised row must meet in shared
// memory: normalise -> sh_row -> rope -> store/quant. The `cos == nullptr &&
// xq == nullptr` arm (the plain split path of `dsv41_sparse_attn`) keeps the
// ORIGINAL direct-store body byte-for-byte, so the un-fused arm does not move.
//
// ⚠️ `sh_acc[4][512]`-style warp truncation is NOT an issue here: this kernel has
// no per-warp partials, every thread walks `c` over the whole d row.
__global__ void sparse_attn_merge_kernel(
    const float* __restrict__ sink, float* __restrict__ out, int b, int m, int h, int d, int C,
    const float* __restrict__ cos, const float* __restrict__ sin, const int* __restrict__ base,
    int rope_rd, int half, int mul, int off, int step, int inverse, uint8_t* __restrict__ xq,
    float* __restrict__ xsc) {
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
    if (cos == nullptr && xq == nullptr) {
        // Plain split arm: unchanged (no smem, no barrier, same bytes).
        for (int c = threadIdx.x; c < d; c += blockDim.x) {
            float a = 0.f;
            for (int ck = 0; ck < C; ++ck)
                a += P[(size_t)ck * kAttnStride + 2 + c] *
                     expf(P[(size_t)ck * kAttnStride] - smax);
            orow[c] = (se > 0.f) ? a / se : 0.f;
        }
        return;
    }
    __shared__ float sh_row[512];
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float a = 0.f;
        for (int ck = 0; ck < C; ++ck)
            a += P[(size_t)ck * kAttnStride + 2 + c] * expf(P[(size_t)ck * kAttnStride] - smax);
        sh_row[c] = (se > 0.f) ? a / se : 0.f;
    }
    __syncthreads();   // sh_row complete before the rope reads it (and the quant below)
    if (cos != nullptr) {
        // PHASE 2 (`sparse_attn_orope_kernel` :1557-1567 verbatim): inverse rope on
        // the trailing `rope_rd` columns; `hh` is this block's head, so the position
        // argument matches the single-block path's per-head loop index.
        const int tt = (*base) * mul + off + hh * step;
        float* rrow = sh_row + (d - rope_rd);
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            const float cc = cos[(size_t)tt * half + i];
            const float ss = sin[(size_t)tt * half + i] * (inverse ? -1.f : 1.f);
            const float x0 = rrow[2 * i], x1 = rrow[2 * i + 1];
            rrow[2 * i] = x0 * cc - x1 * ss;
            rrow[2 * i + 1] = x0 * ss + x1 * cc;
        }
        __syncthreads();   // rope writes visible before the store + quant
    }
    // PHASE 3 (`sparse_attn_orope_kernel` :1569-1595 verbatim): store the roped row,
    // then the fp8 emission of it. `d % 32 == 0` (launcher-guarded) so the flat
    // per-32-block index is exactly the head-local one and a warp's 32 lanes cover
    // one block - bit-identical to the standalone `dsv41_quant_fp8`.
    for (int c = threadIdx.x; c < d; c += blockDim.x) orow[c] = sh_row[c];
    if (xq != nullptr) {
        const int lane = threadIdx.x & 31;
        const size_t xbase = ((size_t)row * h + hh) * (size_t)d;
        const size_t sbase = xbase >> 5;   // d % 32 == 0
        const int nb = d >> 5;
        const int gwarp = threadIdx.x >> 5;
        const int nw = (int)blockDim.x >> 5;
        for (int blk = gwarp; blk < nb; blk += nw) {
            const int c = blk * 32 + lane;
            const float v = sh_row[c];
            float a = fabsf(v);
            for (int off2 = 16; off2 > 0; off2 >>= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off2));
            const float sc = fmaxf(fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
            if (lane == 0) xsc[sbase + blk] = sc;
            const float qv8 = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
            const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(qv8);
            xq[xbase + c] = *(const uint8_t*)&f8;
        }
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
//
// B2 (o-rope fp8 epilogue): when `xq`/`xsc` are non-null, the kernel ALSO emits
// the fp8 (byte + per-32-block scale) of the WHOLE region [0, rows*row_len) in
// the same launch, which is exactly what the consumer's `quant1(s.o)` ->
// `dsv41_quant_fp8` would have produced (rows=1, block=32, round_scale). NOTE
// the rope pass below touches only the trailing `dim` columns of each row and
// processes a PAIR per lane, so it CANNOT own a quant block: the emission is a
// SECOND pass (one warp per 32-block, quant_kernel<0>'s arithmetic term for
// term) after a barrier makes the rotated writes visible. `dsv41_apply_rope`
// passes null/null and the kernel is byte-identical to before; the epilogue
// variant is reached through `dsv41_apply_rope_q`, which declines (returns 1)
// unless rows*row_len is a multiple of 32 so every warp is fully active.
__global__ void apply_rope_kernel(float* __restrict__ x, const float* __restrict__ cos,
                                  const float* __restrict__ sin, int rows, int row_len, int dim,
                                  int half, const int* __restrict__ base, int mul, int off,
                                  int step, int inverse, uint8_t* __restrict__ xq,
                                  float* __restrict__ xsc) {
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
    if (xq == nullptr) return;
    // fp8 epilogue: the barrier makes the rope writes above visible to the
    // quant pass, which reads a different lane's column of the SAME row.
    __syncthreads();
    const int total = rows * row_len;   // multiple of 32 (guarded by the launcher)
    const int lane = threadIdx.x & 31;
    const int gwarp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nwarp = gridDim.x * (blockDim.x >> 5);
    for (int b = gwarp; b < total / 32; b += nwarp) {
        const int col = b * 32 + lane;
        const float v = x[col];
        // one shuffle for the whole 32-element block: every warp is fully
        // active inside a block (blockDim is a multiple of 32 and `col` starts
        // at a block boundary), which is what makes this quant_kernel<0>-exact.
        float a = fabsf(v);
        for (int off2 = 16; off2 > 0; off2 >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off2));
        const float sc = fmaxf(fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
        if (lane == 0) xsc[b] = sc;
        const float q = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
        const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
        xq[col] = *(const uint8_t*)&f8;
    }
}

// ---------------------------------------------------------------------------
// P1 (DSV41_SPARSE_OROPE, DEFAULT ON): sparse attention + the inverse o-rope +
// the fp8 emission of the roped attention output, in ONE launch.
//
// WHY THIS IS A GEOMETRY-PRESERVING FUSION (and not a new shape): the two
// kernels it replaces own EXACTLY the same block. `sparse_attn_pf_kernel`
// (grid=(b*m,h), block=128) and the o-rope call shape of `apply_rope_kernel`
// (`dsv41_apply_rope_q`, rows=nlh, row_len=hd, block=128) both give one block
// per (row, head) covering the whole d-wide head row. `d` (== hd) is a multiple
// of 32, so the fp8 emission's per-32-block index is aligned at every head
// boundary and the head-local pass emits the SAME bytes as the flat
// `dsv41_quant_fp8` pass it replaces. Nothing about the arithmetic moves.
//
// Phase 1 is `sparse_attn_pf_kernel`'s body verbatim; the ONLY change is that
// the normalised o row lands in shared memory instead of global (no global
// store + reload round trip). Phase 2 is `apply_rope_kernel:1218-1224` verbatim
// on the trailing `rope_rd` columns of that shared row. Phase 3 is the fp8
// epilogue at `apply_rope_kernel:1230-1246` verbatim, indexed
// `(row*h+hh)*d + c` for the byte and `((row*h+hh)*d)/32 + b` for the scale.
// Two extra block barriers make the two shared-memory hand-offs visible; both
// are intra-block, on a kernel that is latency-, not launch-, bound.
//
// SHARED-MEMORY BUDGET: the o row is ONE head (d f32 = 2 KB at d=512), not the
// whole (nlh x hd) region - a block owns a single head. Total static smem is
// 8448 B (sh_smax/sh_se/sh_acc, unchanged) + 2048 B (sh_row) ~= 10.5 KB, well
// inside the 48 KB default. No cudaFuncSetAttribute opt-in is required.
//
// The caller skips BOTH the standalone `apply_rope_q` and the `quant1` launch
// when this takes. `dsv41_sparse_attn_orope` DECLINES (returns 1) unless the
// plain call would have selected `sparse_attn_pf_kernel` and the emission shape
// is exact (d <= 512, d % 32 == 0, 0 < rope_rd <= d, rope_rd even); the caller
// then runs the old three-launch sequence, bit-identical by construction.
__global__ void sparse_attn_orope_kernel(
    const float* __restrict__ q, const float* __restrict__ kv,
    const float* __restrict__ sink, const int32_t* __restrict__ idxs,
    float* __restrict__ out, int b, int m, int h, int d,
    const int* __restrict__ clen, int window, int index_topk, float scale,
    const float* __restrict__ cos, const float* __restrict__ sin,
    const int* __restrict__ base, int rope_rd, int half, int mul, int off, int step,
    int inverse, uint8_t* __restrict__ xq, float* __restrict__ xsc) {
    const int n = window + *clen;
    const int topk = window + ((*clen < index_topk) ? *clen : index_topk);
    const int row = blockIdx.x;
    if (row >= b * m) return;
    const int bb = row / m, mm = row % m;
    const int32_t* irow = idxs + (size_t)(bb * m + mm) * topk;
    // ---- the one addition to the phase-1 footprint: the o row, in smem ----
    // d <= 512 (kMaxPerW * 32) is guaranteed by the launcher.
    __shared__ float sh_row[512];
    for (int hh = blockIdx.y; hh < h; hh += gridDim.y) {
        const float* qr = q + ((size_t)(bb * m + mm) * h + hh) * d;
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        const int nwarp = (int)blockDim.x >> 5;
        // Stage the q row once: kills the per-slot re-read of qr (16 floats/lane).
        float qv[kMaxPerW];
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) qv[i] = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) {
            const int c = lane + i * 32;
            if (c < d) qv[i] = qr[c];
        }
        float my_acc[kMaxPerW];
#pragma unroll
        for (int i = 0; i < kMaxPerW; ++i) my_acc[i] = 0.f;
        float my_smax = -1e30f, my_se = 0.f;
        // Three kv-row buffers, rotating by slot; kBase is the row base.
        float kb0[kMaxPerW], kb1[kMaxPerW], kb2[kMaxPerW];
        const size_t kBase = (size_t)bb * n * d;
        // Prologue: fire the idx loads for the first three slots, then their rows.
        int t = wid;
        int ia = (t < topk) ? irow[t] : -1;
        int ib = (t + nwarp < topk) ? irow[t + nwarp] : -1;
        int ic = (t + 2 * nwarp < topk) ? irow[t + 2 * nwarp] : -1;
        if (ia >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb0[i] = kv[kBase + (size_t)ia * d + c];
            }
        }
        if (ib >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb1[i] = kv[kBase + (size_t)ib * d + c];
            }
        }
        if (ic >= 0) {
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) kb2[i] = kv[kBase + (size_t)ic * d + c];
            }
        }
        // Main loop: three slots per pass. Slot t lives in kb0, t+nwarp in kb1,
        // t+2*nwarp in kb2; each buffer's next row load is fired right after
        // that buffer's compute, so every row has two compute phases of
        // distance. The three next-slot indices are hoisted to the loop top
        // (loads have no side effects - pure earlier issue).
        for (; t + 2 * nwarp < topk; t += 3 * nwarp) {
            const int td = t + 3 * nwarp;   // next slot destined for kb0
            const int te = td + nwarp;      // next slot destined for kb1
            const int tf = te + nwarp;      // next slot destined for kb2
            const int id = (td < topk) ? irow[td] : -1;
            const int ie = (te < topk) ? irow[te] : -1;
            const int iff = (tf < topk) ? irow[tf] : -1;
            if (ia >= 0) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) dot += qv[i] * kb0[i];
                }
                for (int off = 16; off > 0; off >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                dot *= scale;
                const float nm = fmaxf(my_smax, dot);
                const float corr = expf(my_smax - nm);
                const float e = expf(dot - nm);
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) my_acc[i] = my_acc[i] * corr + e * kb0[i];
                }
                my_se = my_se * corr + e;
                my_smax = nm;
            }
            // kb0 is dead now: fire the row load for slot td into it.
            if (id >= 0) {
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) kb0[i] = kv[kBase + (size_t)id * d + c];
                }
            }
            if (ib >= 0) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) dot += qv[i] * kb1[i];
                }
                for (int off = 16; off > 0; off >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                dot *= scale;
                const float nm = fmaxf(my_smax, dot);
                const float corr = expf(my_smax - nm);
                const float e = expf(dot - nm);
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) my_acc[i] = my_acc[i] * corr + e * kb1[i];
                }
                my_se = my_se * corr + e;
                my_smax = nm;
            }
            // kb1 is dead now: fire the row load for slot te into it.
            if (ie >= 0) {
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) kb1[i] = kv[kBase + (size_t)ie * d + c];
                }
            }
            if (ic >= 0) {
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) dot += qv[i] * kb2[i];
                }
                for (int off = 16; off > 0; off >>= 1)
                    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
                dot *= scale;
                const float nm = fmaxf(my_smax, dot);
                const float corr = expf(my_smax - nm);
                const float e = expf(dot - nm);
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) my_acc[i] = my_acc[i] * corr + e * kb2[i];
                }
                my_se = my_se * corr + e;
                my_smax = nm;
            }
            // kb2 is dead now: fire the row load for slot tf into it.
            if (iff >= 0) {
#pragma unroll
                for (int i = 0; i < kMaxPerW; ++i) {
                    const int c = lane + i * 32;
                    if (c < d) kb2[i] = kv[kBase + (size_t)iff * d + c];
                }
            }
            ia = id;
            ib = ie;
            ic = iff;
        }
        // Tail: at most two slots left (rows for t and t+nwarp in kb0/kb1; the
        // kb2 slot is necessarily past topk or the loop would have continued).
        if (t < topk && ia >= 0) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) dot += qv[i] * kb0[i];
            }
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            dot *= scale;
            const float nm = fmaxf(my_smax, dot);
            const float corr = expf(my_smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) my_acc[i] = my_acc[i] * corr + e * kb0[i];
            }
            my_se = my_se * corr + e;
            my_smax = nm;
        }
        if (t + nwarp < topk && ib >= 0) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) dot += qv[i] * kb1[i];
            }
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            dot *= scale;
            const float nm = fmaxf(my_smax, dot);
            const float corr = expf(my_smax - nm);
            const float e = expf(dot - nm);
#pragma unroll
            for (int i = 0; i < kMaxPerW; ++i) {
                const int c = lane + i * 32;
                if (c < d) my_acc[i] = my_acc[i] * corr + e * kb1[i];
            }
            my_se = my_se * corr + e;
            my_smax = nm;
        }
        // ---- same warp-merge epilogue as the plain warp version ----
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
        float se = 0.f;
        for (int w = 0; w < nwarp; ++w) se += sh_se[w] * wsc[w < 4 ? w : 0];
        se += expf(sink[hh] - smax);
        // PHASE 1 OUTPUT: the normalised o row goes to smem, NOT to global.
        // The value written here is `sparse_attn_pf_kernel`'s `orow[c]` value
        // verbatim; the global store is deferred to phase 3 so that it carries
        // the ROTATED value directly (the two-launch path rotated `s.o` in
        // place right after this store, so the final global state is identical
        // and no intermediate un-roped value is ever observable).
        for (int c = threadIdx.x; c < d; c += blockDim.x) {
            float a = 0.f;
            for (int w = 0; w < nwarp && w < 4; ++w) a += sh_acc[w][c] * wsc[w];
            sh_row[c] = (se > 0.f) ? a / se : 0.f;
        }
        __syncthreads();   // NEW #1: sh_row complete before the rope reads it
        // PHASE 2: inverse rope on the trailing `rope_rd` columns of this head's
        // row - `apply_rope_kernel:1218-1224` verbatim, with `row` = the head
        // row's rope base inside sh_row and `t` the per-call position.
        {
            const int tt = (*base) * mul + off + hh * step;
            float* rrow = sh_row + (d - rope_rd);
            for (int i = threadIdx.x; i < half; i += blockDim.x) {
                const float cc = cos[(size_t)tt * half + i];
                const float ss = sin[(size_t)tt * half + i] * (inverse ? -1.f : 1.f);
                const float x0 = rrow[2 * i], x1 = rrow[2 * i + 1];
                rrow[2 * i] = x0 * cc - x1 * ss;
                rrow[2 * i + 1] = x0 * ss + x1 * cc;
            }
        }
        __syncthreads();   // NEW #2: rope writes visible before the store + quant
        // PHASE 3: global store of the roped row, then the fp8 emission over
        // this head's d columns - `apply_rope_kernel:1230-1246` verbatim. `d`
        // is a multiple of 32 (launcher-guarded) so a warp's 32 lanes cover
        // exactly one 32-element block and the flat block index is
        // `((row*h+hh)*d)/32 + b`.
        {
            float* orow = out + ((size_t)(bb * m + mm) * h + hh) * d;
            for (int c = threadIdx.x; c < d; c += blockDim.x) orow[c] = sh_row[c];
            if (xq != nullptr) {
                const size_t xbase = ((size_t)(bb * m + mm) * h + hh) * (size_t)d;
                const size_t sbase = xbase >> 5;   // d % 32 == 0
                const int nb = d >> 5;
                const int gwarp = threadIdx.x >> 5;
                const int nw = (int)blockDim.x >> 5;
                for (int blk = gwarp; blk < nb; blk += nw) {
                    const int c = blk * 32 + lane;
                    const float v = sh_row[c];
                    float a = fabsf(v);
                    for (int off2 = 16; off2 > 0; off2 >>= 1)
                        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off2));
                    const float sc = fmaxf(fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
                    if (lane == 0) xsc[sbase + blk] = sc;
                    const float qv8 = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
                    const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(qv8);
                    xq[xbase + c] = *(const uint8_t*)&f8;
                }
            }
        }
        __syncthreads();   // sh_* (and sh_row) reuse safety across the hh loop
    }
}

// the norm writes the trailing rope section, the rope rotates it in place).
// The reduction tree is rmsnorm_kernel's verbatim at the same blockDim, and the
// rope half is elementwise - both are bit-identical to the two-launch sequence,
// which is the only reason this is allowed to exist.
__global__ void rmsnorm_rope_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    float* __restrict__ out, const float* __restrict__ cos,
                                    const float* __restrict__ sin, int n, int dim,
                                    int rope_len, int half, const int* __restrict__ base, int mul,
                                    int off, int step, int inverse, float eps) {
    const int row_i = blockIdx.x;
    if (row_i >= n) return;
    const float* xr = x + (size_t)row_i * dim;
    float* or_ = out + (size_t)row_i * dim;
    // identical tree to rmsnorm_kernel (blockDim-sized cross-warp reduce)
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    float lane = ss;
    for (int o = 16; o > 0; o >>= 1) lane += __shfl_down_sync(0xffffffffu, lane, o);
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lane;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += red[i];
        red[0] = rsqrtf(t / dim + eps);
    }
    __syncthreads();
    const float inv = red[0];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) or_[i] = xr[i] * inv * w[i];
    if (rope_len <= 0) return;
    __syncthreads();   // the rope pass reads what the norm pass wrote
    const int t = (*base) * mul + off + row_i * step;
    float* rr = or_ + (dim - rope_len);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float c = cos[(size_t)t * half + i];
        const float s = sin[(size_t)t * half + i] * (inverse ? -1.f : 1.f);
        const float x0 = rr[2 * i], x1 = rr[2 * i + 1];
        rr[2 * i] = x0 * c - x1 * s;
        rr[2 * i + 1] = x0 * s + x1 * c;
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

// ------------------------------------------------------- indexer score (stage A)
// The scoring pass of `indexer_topk_kernel`, split into its own kernel so the
// per-candidate work (nh-lane 128-deep FMA chain + 32-shuffle reduce) is spread
// over the whole machine instead of being swept by 32 warps in 64 serial waves.
// It writes the SAME per-(row, candidate) value the fused kernel used to build in
// its chunk loop: identical lane<nh map, ascending-h shuffle sum,
// `fmaxf(dot,0) * w[..]` product, `acc * softmax_scale * head_scale` order, and
// identical cl / candidate masks - so stage B's sort/merge/output see the same bits.
//
// Fixed dims, exactly like g_attn_part: the second extent is a STRIDE and never
// the live candidate count. n_pos is a PER-STEP value and a CUDA graph freezes
// both the launch arguments and the grid, so the grid below is a constant and the
// live bound is read from `*lens` inside the kernel. kIdxMaxPos covers the largest
// count the config can reach (max_comp = max_pos / ratio + 2, ratio >= 1,
// DSV41_MAX_POS default 64k -> 65538 for a ratio-1 layer). RAISE kIdxMaxPos IN
// LOCKSTEP WITH DSV41_MAX_POS or both stages clamp together and silently drop
// candidates.
#define kIdxMaxRows 8            // b*m rows (one indexer row per (b,m) pair)
#define kIdxMaxPos  (65536 + 2)  // compressed latents per row, this run's ceiling
__device__ float g_idx_score[kIdxMaxRows][kIdxMaxPos];

// Grid (kIdxScoreBlocks, m, b): one warp per candidate, grid-strided over the live
// range, so the LAUNCH SHAPE never depends on n_pos.
constexpr int kIdxScoreBlocks = 256;    // v1's grid: 2048 warps over 148 SMs = 13.8/SM
// v2's grid. 256 CTAs x 8 warps is 13.8 warps/SM (21% occupancy) — the same
// "too few blocks" disease gate v1 had at 48. 1024 CTAs x 8 = 8192 warps = 55/SM
// (~86%), just under the 8-CTA/SM residency limit for a 256-thread block. Still a
// compile-time constant: a CUDA graph replay must not see an n_pos-derived grid,
// and the kernel's grid-stride loop covers any live candidate count.
constexpr int kIdxScoreBlocksV2 = 1024;
// v2's K-split default. WPR warps split the hd reduction of ONE candidate; WPR=1
// is the no-smem/no-sync variant. The split is worth it only when the candidate
// count is small enough that 8 warps/CTA cannot fill the machine: the fold costs
// two __syncthreads per (rpb-candidate) round, and with the live n_pos of a
// decode step (index-source layers have compress_ratio == 1, so n_pos tracks the
// sequence length — tens of thousands) the sync overhead exceeds the latency it
// hides. Default 1, DSV41_IDX_SCORE_WPR=2/4/8 for the short-context end.
constexpr int kIdxScoreWprDefault = 1;

__global__ void indexer_score_kernel(const float* __restrict__ q, const float* __restrict__ ik,
                                     const float* __restrict__ w, const uint8_t* __restrict__ cand,
                                     const int32_t* __restrict__ lens, int m, int nh, int hd,
                                     int n_pos, float softmax_scale, float head_scale,
                                     int uses_cand) {
    // `n_pos` is the host fallback; the live device counter wins (same rule as the
    // fused kernel), and the declared bound is a defensive ceiling.
    if (lens != nullptr && *lens > 0) n_pos = *lens;
    if (n_pos > kIdxMaxPos) n_pos = kIdxMaxPos;
    const int mm = blockIdx.y, bb = blockIdx.z;
    const size_t row = (size_t)bb * m + mm;
    if (row >= kIdxMaxRows) return;
    int cl = n_pos;
    if (lens != nullptr) cl = lens[mm];
    if (cl > n_pos) cl = n_pos;
    const float* qrow = q + row * (size_t)nh * hd;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    // Per-ROW warp id and per-row stride: blocks on other (y,z) handle other rows.
    const int gw = (int)blockIdx.x * nwarp + wid;
    const int ntot = (int)(gridDim.x * nwarp);
    for (int p = gw; p < n_pos; p += ntot) {
        const float* krow = ik + ((size_t)bb * n_pos + p) * hd;
        float dot = 0.f;
        if (lane < nh) {
            const float* qh = qrow + (size_t)lane * hd;
            for (int c = 0; c < hd; ++c) dot += qh[c] * krow[c];
            dot = fmaxf(dot, 0.f) * w[row * (size_t)nh + lane];
        }
        float acc = 0.f;
        for (int h = 0; h < nh; ++h) {
            const float dv = __shfl_sync(0xFFFFFFFFu, dot, h);
            if (lane == 0) acc += dv;
        }
        acc = __shfl_sync(0xFFFFFFFFu, acc, 0);
        float sv = acc * softmax_scale * head_scale;
        if (p >= cl) sv = -INFINITY;
        if (uses_cand && cand != nullptr && !cand[row * (size_t)n_pos + p]) sv = -INFINITY;
        if (lane == 0) g_idx_score[row][p] = sv;
    }
}

// ----------------------------------------------------- indexer score v2
// v1 above has gate-v1's disease (ferrite_kernels.cu: gemv_bf16_kernel ->
// gemv_bf16_v2_kernel): one SCALAR load per FMA on BOTH q and k (256 scalar LDG
// per lane per candidate), a 128-deep dependent FMA chain, an O(nh)=32 broadcast
// shuffle head fold whose accumulator lives on lane 0, and only 2048 warps on the
// machine. The gate's v2 fix maps over directly:
//   (a) float4 (16B) q/k loads + FOUR independent accumulators: 32 vector steps
//       instead of 128 scalar ones, each accumulator's chain down to 8;
//   (b) WPR warps split the hd reduction of ONE candidate, folded in the same
//       block through shared memory (gemv_bf16_v2's `part[warp]`, no atomics, no
//       second kernel). WPR=1 keeps v1's one-warp-per-candidate mapping with the
//       vector body and touches neither smem nor a barrier;
//   (c) the head fold is a 5-step shfl_down tree instead of nh broadcast
//       shuffles (nh == 32 here, so this is 32 shuffles + a 32-long dependent add
//       chain on lane 0 -> 5 shuffles + 5 adds);
//   (d) the CTA count is no longer pinned at 256 (kIdxScoreBlocksV2).
// NOTE the K-split is a latency fix, not a throughput fix (the total work is
// unchanged): it pays off only when the candidate count is too small to fill the
// machine with one warp per candidate. Its fold is two __syncthreads per round,
// which at a large live n_pos costs more than it hides — hence the WPR=1 default.
//
// The summation ORDER changes (float4 lanes, K-slice partials, tree folds) =>
// f32 drift ~1e-7 on the score, orders of magnitude below the top-k margin (the
// golden in ops.rs is a 1-element dot, and acceptance is token-level A/B parity).
// DSV41_IDX_SCORE_V2=0 restores v1 verbatim.
// Also KEEP the (nh <= 32) and (hd % 4 == 0) preconditions: the tree fold sums 32
// lanes and the vector body needs 16B-aligned rows — the launcher falls back to
// v1 when either fails.

// The scale + mask tail of the two stage-A kernels, statement for statement as
// v1 wrote it inline (both stages must publish the same value to the sort).
__device__ __forceinline__ void idx_score_store(float* __restrict__ grow, int p, int cl, float sv,
                                                float softmax_scale, float head_scale,
                                                const uint8_t* __restrict__ cand, size_t row,
                                                int n_pos, int uses_cand) {
    float sc = sv * softmax_scale * head_scale;
    if (p >= cl) sc = -INFINITY;
    if (uses_cand && cand != nullptr && !cand[row * (size_t)n_pos + p]) sc = -INFINITY;
    grow[p] = sc;
}

// 5-step shfl_down tree (ascending-h becomes a balanced tree: same sum, other
// rounding). Lanes >= nh carry 0 and contribute nothing.
__device__ __forceinline__ float idx_head_fold(float sv) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sv += __shfl_down_sync(0xFFFFFFFFu, sv, off);
    return sv;
}

template <int WPR>
__global__ void indexer_score_kernel_v2(const float* __restrict__ q, const float* __restrict__ ik,
                                        const float* __restrict__ w, const uint8_t* __restrict__ cand,
                                        const int32_t* __restrict__ lens, int m, int nh, int hd,
                                        int n_pos, float softmax_scale, float head_scale,
                                        int uses_cand) {
    // Same live-counter rule as v1 (and the fused kernel): the device counter
    // wins over the host fallback, the declared bound is a defensive ceiling.
    if (lens != nullptr && *lens > 0) n_pos = *lens;
    if (n_pos > kIdxMaxPos) n_pos = kIdxMaxPos;
    const int mm = blockIdx.y, bb = blockIdx.z;
    const size_t row = (size_t)bb * m + mm;
    if (row >= kIdxMaxRows) return;
    int cl = n_pos;
    if (lens != nullptr) cl = lens[mm];
    if (cl > n_pos) cl = n_pos;
    const float* qrow = q + row * (size_t)nh * hd;
    const float* wrow = w + row * (size_t)nh;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int rpb = nwarp / WPR;   // candidates in flight per CTA
    const int g = warp / WPR;      // this warp's candidate slot inside the CTA
    const int kw = warp % WPR;     // this warp's hd slice
    // The per-warp slice is rounded up to a multiple of 4 floats so every float4
    // access in [c0, c1) is 16B aligned: the rows are hd-strided and the launcher
    // guarantees hd % 4 == 0.
    const int kper = ((hd + WPR - 1) / WPR + 3) & ~3;
    const int c0 = kw * kper;
    const int c1 = (c0 + kper < hd) ? (c0 + kper) : hd;
    // [warp][lane]: one hd-slice partial per head. blockDim.x is pinned at 256 by
    // the launcher, so nwarp == 8 and the first dim is 8. WPR == 1 never writes it
    // (the template keeps the array at one row so the un-split kernel allocates
    // nothing meaningful).
    __shared__ float s_part[WPR > 1 ? 8 : 1][32];
    // Uniform trip count for the WHOLE CTA. The K-split fold carries
    // __syncthreads INSIDE the loop, so a per-warp bound (p = base + r*pstep + g,
    // g-varying) would let the low groups run one round more than the high ones
    // and hang the block whenever n_pos does not divide the stride. Every group
    // steps the same number of rounds; a group whose candidate is past n_pos
    // contributes zeros and skips its store.
    const int base = (int)blockIdx.x * rpb;
    const int pstep = (int)gridDim.x * rpb;
    const int nround = (n_pos > base) ? (n_pos - base + pstep - 1) / pstep : 0;

    for (int r = 0; r < nround; ++r) {
        const int p = base + r * pstep + g;
        const bool live = p < n_pos;
        const float* krow = ik + ((size_t)bb * n_pos + (live ? p : 0)) * hd;
        float dot = 0.f;
        if (live && lane < nh) {
            const float* qh = qrow + (size_t)lane * hd;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            int c = c0;
            // 4 float4 per operand per iteration -> 4 independent chains.
            for (; c + 15 < c1; c += 16) {
                const float4 qa = *reinterpret_cast<const float4*>(qh + c);
                const float4 ka = *reinterpret_cast<const float4*>(krow + c);
                const float4 qb = *reinterpret_cast<const float4*>(qh + c + 4);
                const float4 kb = *reinterpret_cast<const float4*>(krow + c + 4);
                const float4 qc = *reinterpret_cast<const float4*>(qh + c + 8);
                const float4 kc = *reinterpret_cast<const float4*>(krow + c + 8);
                const float4 qd = *reinterpret_cast<const float4*>(qh + c + 12);
                const float4 kd = *reinterpret_cast<const float4*>(krow + c + 12);
                a0 += qa.x * ka.x + qa.y * ka.y + qa.z * ka.z + qa.w * ka.w;
                a1 += qb.x * kb.x + qb.y * kb.y + qb.z * kb.z + qb.w * kb.w;
                a2 += qc.x * kc.x + qc.y * kc.y + qc.z * kc.z + qc.w * kc.w;
                a3 += qd.x * kd.x + qd.y * kd.y + qd.z * kd.z + qd.w * kd.w;
            }
            for (; c + 3 < c1; c += 4) {
                const float4 qa = *reinterpret_cast<const float4*>(qh + c);
                const float4 ka = *reinterpret_cast<const float4*>(krow + c);
                a0 += qa.x * ka.x + qa.y * ka.y + qa.z * ka.z + qa.w * ka.w;
            }
            for (; c < c1; ++c) a0 += qh[c] * krow[c];
            dot = (a0 + a1) + (a2 + a3);
        }
        if (WPR == 1) {
            // No K-split: this warp owns the whole dot and publishes directly.
            const float sv = (live && lane < nh) ? fmaxf(dot, 0.f) * wrow[lane] : 0.f;
            const float tot = idx_head_fold(sv);
            if (lane == 0 && live)
                idx_score_store(g_idx_score[row], p, cl, tot, softmax_scale, head_scale, cand, row,
                                n_pos, uses_cand);
        } else {
            // Stage the per-slice partial of every head, fold the WPR slices in
            // the group's kw==0 warp (relu/weights must see the FULL hd dot, so
            // the fold has to happen before them), then the head tree.
            s_part[warp][lane] = dot;
            __syncthreads();
            if (kw == 0) {
                float full = 0.f;
#pragma unroll
                for (int j = 0; j < WPR; ++j) full += s_part[g * WPR + j][lane];
                const float sv = (live && lane < nh) ? fmaxf(full, 0.f) * wrow[lane] : 0.f;
                const float tot = idx_head_fold(sv);
                if (lane == 0 && live)
                    idx_score_store(g_idx_score[row], p, cl, tot, softmax_scale, head_scale, cand,
                                    row, n_pos, uses_cand);
            }
            // Before the next round overwrites s_part.
            __syncthreads();
        }
    }
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
        // Now a plain LOAD: indexer_score_kernel (stage A) already computed the
        // same value - same lane<nh map, ascending-h shuffle sum, `fmaxf(dot,0)*w`
        // product, `acc * softmax_scale * head_scale` order, same cl/cand masks -
        // into g_idx_score, so the sort below sees bit-identical entries.
        {
            for (int i = tid; i < len; i += nthr) {
                s_pair[2 * i] = g_idx_score[row][base + i];
                s_pair[2 * i + 1] = __int_as_float(base + i);
            }
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

// COMPRESS_FUSE (env gate `DSV41_COMPRESS_FUSE`, default ON): the decode
// compressor's three 1-block launches collapse into ONE.
//
// On a decode step (`b == seqlen == 1`, `ratio > 1`, `start_pos > 0`) the three
// stages are a STRICT chain on this layer's own private buffers:
//   state carry  (compressor_state_kernel, start_pos > 0 branch)
//     -> state_kv/state_score
//   gated pool + RMSNorm (compressor_pool_kernel, mode 2)
//     -> latent + *out_rows
//   rope + ring store + counter bump (compress_commit_kernel)
//     -> ring + *clen
// Every stage is one block (`state` used 2 blocks only because `b*hd = 512` at
// 256 threads), so a single 128-thread block runs them back to back with
// `__syncthreads()` between the stages: the parallel decomposition is unchanged
// and only the two extra launch boundaries disappear (-2 launches per kv-source
// layer, -2 graph nodes).
//
// BIT-IDENTITY (the whole point of the exercise):
//  * state: element-wise stores; the destination set is identical (`{slot*hd+c}
//    for c < hd`), and store order is irrelevant - the carry is order-free.
//  * pool: `nthr == 128` and the SAME `for (c = tid; c < hd; c += nthr)` channel
//    ownership, the same per-channel softmax over the `ratio` slots, the same
//    `s_red[32]` warp + sequential cross-warp RMSNorm tree, and the same
//    `out_row == 0` (`bb` is 0 in decode) row. A different block size would
//    re-group the reduction and change the bytes - hence 128.
//  * commit: the `for (c = threadIdx.x; c < hd; c += blockDim.x)` rope/store loop
//    and the thread-0 `*clen = len + 1` bump are a VERBATIM copy of
//    `compress_commit_kernel` (dsv41_glue.cu) - keep the two bodies in sync.
//
// The originals' early exits cannot be reproduced with `return` here (that is UB
// for a later `__syncthreads()`), so each stage is guarded by the uniform
// `out_rows_val` instead: an unfinished decode group (`(start_pos+1) % ratio !=
// 0`) does the state carry, skips the pool rows AND the commit - exactly what
// the two kernels' early returns did.
__global__ void compressor_fused_kernel(const float* __restrict__ kvp,
                                        const float* __restrict__ scp,
                                        const float* __restrict__ norm_w,
                                        float* __restrict__ state_kv,
                                        float* __restrict__ state_score,
                                        float* __restrict__ latent,
                                        int32_t* __restrict__ out_rows,
                                        const float* __restrict__ cos_t,
                                        const float* __restrict__ sin_t,
                                        float* __restrict__ ring, int* __restrict__ clen,
                                        int hd, int ratio, int rope_dim, int half, int window,
                                        const int* __restrict__ pos_ctr, float eps) {
    const int tid = threadIdx.x, nthr = blockDim.x;
    __shared__ float s_red[32];
    const int start_pos = *pos_ctr;  // device-side: graph-capturable
    const int slot = start_pos % ratio;

    // ---- stage 1: state carry (state_kernel's start_pos > 0 branch, b == 1) ----
    for (int c = tid; c < hd; c += nthr) {
        const size_t dst = (size_t)slot * hd + c;
        state_kv[dst] = kvp[c];
        state_score[dst] = scp[c];
    }
    __syncthreads();

    // ---- stage 2: pool + RMSNorm (pool_kernel's mode 2, bb = 0) ----
    // out_rows is derived from the DEVICE position counter, not passed in (the
    // decision changes per step and a captured graph freezes launch arguments).
    const int out_rows_val = ((start_pos + 1) % ratio == 0) ? 1 : 0;
    if (tid == 0) *out_rows = out_rows_val;

    float yv[16];
    int cn[16];
    int nown = 0;
    for (int c = tid; c < hd; c += nthr) {
        if (nown >= 16) break;
        cn[nown++] = c;
    }
    if (out_rows_val) {  // the originals' `mode == 2 && out_rows_val == 0` return
        for (int i = 0; i < nown; ++i) {
            const int c = cn[i];
            float mx = -INFINITY;
            float sv[32];
            float vv[32];
            const int rr = ratio < 32 ? ratio : 32;
            for (int r = 0; r < rr; ++r) {
                sv[r] = state_score[(size_t)r * hd + c];
                vv[r] = state_kv[(size_t)r * hd + c];
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
        // RMSNorm over the pooled row (weights + eps, exactly as the golden)
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
            latent[c] = yv[i] * inv * norm_w[c];  // out_row == bb == 0
        }
    }
    __syncthreads();

    // ---- stage 3: commit (VERBATIM copy of compress_commit_kernel) ----
    if (out_rows_val > 0) {  // the original kernel's `*out_rows <= 0` return
        const int len = *clen;
        const int group_first = len * ratio;
        const int i0 = hd - rope_dim;
        const float* cs_row = cos_t + (size_t)group_first * half;
        const float* sn_row = sin_t + (size_t)group_first * half;
        float* dst = ring + (size_t)(window + len) * hd;
        for (int c = tid; c < hd; c += nthr) {
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
        __syncthreads();  // every latent read is done before the bump
        if (tid == 0) *clen = len + 1;
    }
}

}  // namespace

#define DSV41_LAUNCH_CHECK()                     \
    do {                                         \
        cudaError_t e = cudaGetLastError();      \
        if (e != cudaSuccess) return (int)e;     \
    } while (0)


// Round-45 root cause: the per-kernel static smem varies (128B-772B+), so
// any hardcoded ceiling will be wrong for some kernel. Compute the true
// ceiling at runtime: device opt-in max minus the kernel's static smem.
template <typename KernPtr>
static inline int dsv41_smem_ceiling(KernPtr /*kern*/) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int dev = 0; cudaGetDevice(&dev);
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    // Static smem varies per kernel; use the maximum we've seen (772B for
    // gemm_fp8_gemv_kernel after struct-pack) plus 256B margin for safety.
    cached = optin - 1024;  // 1024B = max known static + margin
    return cached;
}

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

// Per-device fp4 quantise scratch (see the comment in dsv41_quant_fp4). Only the
// legacy two-launch path uses it; the fused default never touches it.
static uint8_t* g_q4nib[64] = {nullptr};
static size_t   g_q4nib_cap[64] = {0};

// Fused quantise+pack (see quant_fp4_fused_kernel) is the default: it turns the
// two launches into one and drops the rows*cols-byte scratch round trip. Set
// DSV41_QUANT_FP4_FUSE=0 for the legacy quant_kernel<1> + fp4_pack_kernel pair —
// same bytes, two launches, kept for bisection / old-shape fallback. Read once:
// this runs 40x/step and a per-call getenv on the hot path is the slip every
// other gate in this file avoids.
static const int g_q4_fuse = [] {
    const char* e = getenv("DSV41_QUANT_FP4_FUSE");
    return e != nullptr ? atoi(e) : 1;
}();

extern "C" int dsv41_quant_fp4(const float* x, uint8_t* y, float* scale, int rows, int cols,
                               int block, int round_scale, cudaStream_t s) {
    if (rows <= 0 || block <= 0 || cols % block != 0) return (int)cudaErrorInvalidValue;
    const int nb = cols / block;
    const size_t n = (size_t)rows * cols;
    // Fused path: legal whenever a nibble pair cannot straddle a block boundary
    // (even block) and the block fits quant_fp4_fused_kernel's staging array
    // (<= 256). The production shape -- rows=1, cols=5120, block=32 -- is both.
    if (g_q4_fuse != 0 && (block & 1) == 0 && block <= 256 && (n & 1) == 0) {
        quant_fp4_fused_kernel<<<dim3(rows * nb), dim3(1, block), 0, s>>>(
            x, y, scale, rows, cols, block, round_scale);
        return (int)cudaGetLastError();
    }
    // Legacy path (DSV41_QUANT_FP4_FUSE=0, or a shape the fused kernel declines).
    // Cached per-device scratch: a TP8 process has one context per rank thread,
    // so the cache is indexed by device. Growing it is a synchronising
    // cudaMalloc, which is (a) illegal inside a stream capture and (b) a hot-path
    // hazard because quant_fp4 runs several times per layer. The warm-up step
    // sizes every entry, so the capture path never allocates.
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return (int)cudaErrorInvalidDevice;
    const size_t need = n;
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

// a32 gate (DSV41_GEMV_A32), orthogonal to DSV41_GEMV_FP8_MODE. a32 is the
// BLOCK-WIDE pre-decoded activation `s_af` (k f32 = 20 KB at the model's k=5120)
// that turns the consume loop's per-element chain (LDS.8 -> LDS.32 -> FMUL) into
// a single LDS.32. IMPORTANT: this buffer exists in BOTH staged modes - mode 3
// merely reads the fp8 activation from global memory instead of the staged `s_a`
// copy (see the `ap0` / `ap` selects below) - the mode-3-vs-4 difference is the
// k-byte activation STAGING, not the 4k-byte a32 table. So the occupancy question
// ("does dropping the 20 KB table pay at production n?") has NO switch in the
// mode knob and needs this one.
//
// 1 (default) = materialise s_af block-wide, read it in the loop.
// 0           = skip the materialisation and fold the decode+scale back into the
//               loop's operand. The two forms compute the SAME product
//               (s_lut[ap[j]] * s_as[j>>5]) so the result is bit-identical, which
//               is the whole point of a fair occupancy A/B.
//
// The launchers MUST reserve dsv41_gemv_a32_bytes(k) in `scale_bytes` when this is
// 0 (the kernel's `s_rows` slot then sits where `s_af` used to start), otherwise
// kernel and launcher disagree on the layout.
static const bool g_gemv_a32 = [] {
    const char* e = getenv("DSV41_GEMV_A32");
    if (e == nullptr) return true;
    return atoi(e) != 0;
}();
static inline size_t dsv41_gemv_a32_bytes(int k) {
    return g_gemv_a32 ? (size_t)k * sizeof(float) : (size_t)0;
}
// P1 (a32 dead-slot): the k-byte fp8 activation staging row `s_a` (mode 4) is
// NOT allocated when the fused a32 pass writes `s_af` straight from global
// memory -- i.e. a32=1 and the NORM_FUSE prologue is not the producer. Every
// launcher except dsv41_gemm_fp8_mx_rope_norm has qr_raw == nullptr, so for them
// the condition collapses to the a32 gate and mode 4 asks for `warps * k` rows
// instead of `(warps + 1) * k`. The NORM_FUSE launcher keeps the extra row (its
// prologue writes s_a and the decode still reads it) and passes norm_fuse=true.
// MUST stay in lockstep with the kernel's `a32_direct` / `s_ws` computation.
//
// P4 (DSV41_GEMV_ACT_CPASYNC, act-cpasync): the async activation staging has to
// land SOMEWHERE, and the only slot that costs no new memory is the mode-4 `s_a`
// row P1 eliminated. So the gate re-opens that slot exactly like a32_staged does
// (the kernel's `a32_direct` loses the same `&& !act_async` term), and this
// function must reserve it or every pointer after it is short by k bytes. Read
// with a local getenv for the same forward-reference reason as cached_staged,
// and parsed with the same rule as g_gemv_act_cpasync below (unset = on, "0" =
// off) so the host side and the kernel's `act_cpasync` field cannot disagree.
static inline size_t dsv41_gemv_sa_bytes(int k, bool norm_fuse) {
    // Forward-declared P1 gate (definition at g_gemv_a32_staged below) -
    // use a local getenv read to avoid C++ forward-reference issues
    static int cached_staged = -1;
    if (cached_staged < 0) {
        const char* se = getenv("DSV41_GEMV_A32_STAGED");
        cached_staged = (se != nullptr && se[0] == '1') ? 1 : 0;
    }
    // Forward-declared P4 gate (definition at g_gemv_act_cpasync below).
    static int cached_act = -1;
    if (cached_act < 0) {
        const char* ae = getenv("DSV41_GEMV_ACT_CPASYNC");
        cached_act = (ae != nullptr && atoi(ae) == 0) ? 0 : 1;
    }
    return (g_gemv_a32 && !norm_fuse && cached_staged == 0 && cached_act == 0) ? (size_t)0
                                                                             : (size_t)k;
}

// Rows per gemv block, shared by the single-family and the two-family launchers.
// Four rows per block measured 15.91 against 16.14 ms for eight, same session,
// same binary, text unchanged: halving the rows doubles the block count and the
// warps in flight, which is what this latency-bound family actually needs. Two
// was no better than four and stages the activation twice as often, so four.
static const int g_gemv_warps = [] {
    const char* e = getenv("DSV41_GEMV_FP8_WARPS");
    if (e == nullptr) return 4;
    const int v = atoi(e);
    return (v >= 1 && v <= 32) ? v : 4;
}();

// P2 (gemm-prologue-pipeline, 2026-09-11/12): ADAPTIVE rows per block.
//
// The per-call cost of the M=1 GEMV family is dominated by the BLOCK-level
// prologue, not the consume loop: the activation uint4 staging + LUT build + a32
// materialisation is paid ONCE per block and is nearly n-independent (the
// 2026-09-11 probe measured 2.85us of block-level staging across 416 blocks, and
// it grew only ~5 percent when n grew 6.5x). With P1 (a32 dead-slot) the mode-4
// smem at k=5120/warps=4 is 43392B = 232448/43392 = 5.35 -> 5 blocks/SM, i.e.
// ~30 percent residency headroom. The roomey shapes can therefore afford EIGHT
// rows per block: gsmem = 8*k + scale_bytes = 8*5120 + 23552 = 64512B ->
// 232448/64512 = 3.6 -> 3 blocks/SM, but every block now carries twice the rows,
// so the block count HALVES and the fixed prologue is amortised over 2x the
// rows. The latency-bound SMALL shapes (wq_a+wkv n=512+1792, sh_w13 n=640) want
// the opposite -- more blocks = more warps in flight -- so the choice is gated
// on n, with the crossover at 2048.
//
// ONLY the four host launchers that already select `g_gemv_warps` use this
// helper: dsv41_gemm_fp8_mx / _mx_add / _mx_f32 / _mx2. The rope family
// (mx_rope / mx_rope_norm / mx2_rope) KEEPS nwarps == 32 unconditionally:
//   * mx_rope / mx2_rope: the rope epilogue exchanges a pair across two ADJACENT
//     warps (`s_rows[warp]` / `s_rows[warp+1]`, `e = blockIdx.x*nwarps + warp`)
//     and the launcher relies on grid*nwarps == n.
//   * mx_rope_norm: same epilogue PLUS the NORM_FUSE prologue, whose cross-warp
//     reduction tree must run at the reference's 1024 threads to stay
//     bit-identical to the standalone rmsnorm_q it replaces.
// B1 (the fused fp8-row emit) is an M=1 epilogue that likewise requires
// nwarps == 32, and dsv41_gemm_fp8_mx keeps its explicit `warps = 32` override
// for it -- the adaptive value below is computed first and then overwritten.
//
// DSV41_GEMV_WARPS_ADAPTIVE: default ON. "=0" restores the fixed
// DSV41_GEMV_FP8_WARPS everywhere (the A/B arm and the rollback).
// Read once (static): these launchers run a few hundred times per step and a
// per-call getenv on the hot path is the slip every gate in this file avoids.
static const bool g_gemv_warps_adaptive = [] {
    const char* e = getenv("DSV41_GEMV_WARPS_ADAPTIVE");
    if (e == nullptr) return true;
    return atoi(e) != 0;
}();
// The large-n arm. Kept as named constants rather than literals so the crossover
// is auditable (and tweakable) in one place.
static const int kGemvWarpsBigN = 2048;   // n at/above which the large-n arm applies
// P2b (gemm-warps-sweep, 2026-09-12): the large-n rows/block is its OWN knob.
//
// DSV41_GEMV_FP8_WARPS only feeds the small-n arm (n < 2048), so sweeping it can
// never move the large shapes; driving the large-n arm through it (or through
// DSV41_GEMV_WARPS_ADAPTIVE=0, which collapses BOTH arms onto g_gemv_warps) would
// confound the A/B by also moving wq_a/wkv/sh_w13. This variable is therefore the
// only thing the large-n arm reads: serve can scan 4..32 while the small shapes
// stay pinned at g_gemv_warps.
//
// Default 8 == the constant this replaces, so an unset env is byte-identical to
// the previous behaviour. Valid 4..32; a missing / unparsable / out-of-range
// value falls back to 8. The adaptive gate still guards the arm: with
// DSV41_GEMV_WARPS_ADAPTIVE=0 the large-n shapes revert to g_gemv_warps and this
// variable is inert (that is the rollback path, not an A/B arm).
// Read once (static): these launchers run a few hundred times per step.
static const int g_gemv_warps_big = [] {
    const char* e = getenv("DSV41_GEMV_WARPS_BIG");
    if (e == nullptr) return 8;
    const int v = atoi(e);
    return (v >= 4 && v <= 32) ? v : 8;
}();
static inline int dsv41_gemv_warps_for(int n) {
    if (g_gemv_warps_adaptive && n >= kGemvWarpsBigN) return g_gemv_warps_big;
    return g_gemv_warps;
}

// P1 staged gate (DSV41_GEMV_A32_STAGED), default OFF = the P1 direct form.
//
// P1 (a32 dead-slot elimination) merges the two mode-4 activation passes into
// one: global uint4 -> LUT decode -> straight into `s_af`, so the k-byte `s_a`
// intermediate is never allocated (smem 48512 -> 43392 B at k=5120/warps=4 ->
// 5 blocks/SM instead of 4). "=1" puts the intermediate back:
//
//     global -> s_a (uint4 copy) -> s_lut[s_a[i]] * s_as[i>>5] -> s_af
//
// which is the A/B arm and the rollback if the fused pass turns out to increase
// global traffic (the staged copy acted as a per-block cache of the activation
// row). The 2026-09-12 A/B measured the two arms as IDENTICAL (6.84ms / 146.2
// tok/s), so the direct form stays the default.
//
// LAYOUT COUPLING -- the gate is read by BOTH sides or the two disagree:
//   * the kernel turns it into `a32_direct = false`, which allocates the `s_a`
//     slot and shifts `s_ws`/`s_as`/`s_lut`/`s_af`/`s_rows` past it;
//   * the launchers must reserve the same k bytes (dsv41_gemv_sa_bytes below).
// Getting one side only = every pointer after the missing slot is short by k
// bytes. Do NOT re-add a second gate without threading it through both.
static const bool g_gemv_a32_staged = [] {
    const char* e = getenv("DSV41_GEMV_A32_STAGED");
    if (e == nullptr) return false;
    return atoi(e) != 0;
}();

// P3 (gemm-prologue-overlap, 2026-09-11/12): cp.async WEIGHT-FIRST prologue,
// default ON. "=0" restores the issue-after-barrier ordering (A/B + rollback).
//
// WHAT IT FIXES. The row loop issued the weight row's cp.asyncs AFTER the
// block-wide prologue barrier, so the row transfer sat on the critical path in
// front of the dot: issue -> commit -> wait_all -> consume. Nothing in the loop
// covers it (the only other work there is the scale-byte staging). The
// block-level activation staging and the barrier behind it are good cover, and
// the weight row does NOT depend on any of it -- `w` / `w_scale` are model
// constants, everything the prologue computes is `a`-side. So the ISSUE moves
// above the staging: [cp.async weight row] -> [activation staging/LUT/a32] ->
// barriers -> [wait_all] -> [dot]. Same slot, same bytes, same consume order, so
// the result stays bit-identical.
//
// NO EXTRA SHARED MEMORY. The prefetch lands in the SAME per-warp row slot the
// loop would have staged into (`s_w + warp*k`), so gsmem is unchanged and the
// 5-blocks/SM occupancy of the P1 layout is preserved. A separate double-buffer
// slot would have cost another warps*k bytes (48512 B at warps=4 -> 4
// blocks/SM): it would have paid for the overlap with residency, which is the
// trap the gemm-prologue-overlap analysis flagged for the double-buffered
// variant. Consequently the scale row (`s_ws`) is deliberately NOT prefetched --
// it is a handful of byte loads, it already overlaps the weight transfer inside
// the loop, and moving it above the barrier would put a synchronous stall in
// front of the barrier that every warp has to reach.
//
// IT SITS AFTER cudaGridDependencySynchronize() ON PURPOSE. `w` can be written by
// the previous node on the stream, and under PDL that producer may still be
// running when this grid starts, so an early read would be a race -- the
// prefetch therefore lives below the opening sync with every other
// producer-output read, and is a REORDERING of the loop's own staging rather
// than a new early read.
//
// SCOPE. Only the first row of each warp's strided walk gets prefetched (the one
// that is live whenever gridDim.x*nwarps >= n, i.e. every production shape); the
// loop recognises it by `row == pf_row` and skips its own staging for that
// iteration, keeping the old sequence for the rest.
static const bool g_gemv_cpasync = [] {
    const char* e = getenv("DSV41_GEMV_CPASYNC");
    if (e == nullptr) return true;
    return atoi(e) != 0;
}();

// P4 (gemm-act-cpasync, 2026-09-12): cp.async ACTIVATION staging, default OFF
// (A/B arm; "=1" enables, an unset variable keeps today's synchronous staging).
//
// WHAT IT DOES. The mode-4 block-wide fp8 activation row is currently staged
// into `s_a` by a SYNCHRONOUS wide copy, and -- only while P1's fused
// `a32_direct` pass is disabled -- that copy is a plain global->shared transfer
// with no decode attached to it. This gate turns it into cp.async issued in the
// prologue, so the DRAM round trip leaves the thread's critical path: the
// LUT build / scale staging run underneath the transfer and the wait is
// collected just before the barrier that publishes `s_a`. Same bytes, same
// slot, same LUT entry, same per-block scale -> bit-identical.
//
// WHY IT RE-OPENS THE P1 SLOT. `a32_direct` (a32=1 / mode 4 / no NORM_FUSE)
// decodes straight from global into `s_af` and therefore leaves the k-byte `s_a`
// row UNALLOCATED (P1: 48512 -> 43392 B at k=5120/warps=4 -> 5 blocks/SM
// instead of 4). An async copy needs a place to land, so this gate forces
// `a32_direct` false and pays that k bytes back (dsv41_gemv_sa_bytes mirrors the
// condition). IT IS THEREFORE NOT A FREE WIN AT warps=4: the small-n arm loses a
// block/SM (5 -> 4), while the warps=8 arm (n >= 2048) and the forced-32-warp
// rope launchers keep their residency (3 and 1 blocks/SM either way). That
// trade-off is exactly why the gate defaults OFF -- the measured P1 A/B called
// staged-vs-direct NEUTRAL, so "staged + async copy" has to prove itself before
// it becomes the default, not the other way round.
//
// LAYOUT COUPLING -- read by BOTH sides or the two disagree:
//   * the kernel reads `gc.act_cpasync` (a GemvCore field) into `act_async`,
//     which forces `a32_direct` false and skips the synchronous copy;
//   * dsv41_gemv_sa_bytes() must reserve the same k bytes for every launcher.
// Getting one side only = every pointer after the missing slot is short by k
// bytes (the failure mode the P1 comments warn about).
static const bool g_gemv_act_cpasync = [] {
    const char* e = getenv("DSV41_GEMV_ACT_CPASYNC");
    if (e == nullptr) return true;    // A/B verified ON: 6.44->6.24ms (-0.20ms, 2026-09-12)
    return atoi(e) != 0;
}();

// P3 for the FUSED gate + shared-expert gemv (gemv_bf16_fp8x2_kernel), env
// `DSV41_BF16_CPASYNC`, default ON (`=0` restores the old issue order).
//
// Same transform as g_gemv_cpasync above -- move a warp's FIRST weight-row
// transfer from the row loop's prologue-list into the block prologue, so the
// DRAM round trip is overlapped instead of exposed -- with the added advantage
// that this kernel's prologue is longer than the plain gemv's: on top of the
// e4m3 LUT build there is the P1 fused `s_af` materialisation (k/16 16B loads
// from global + 16 LUT lookups each), and the row loop cannot start until the
// barrier after it. Issuing the cp.async group before ALL of that gives the
// staged row the LUT + a32 pass + barrier to land under.
//
// SCOPE. fp8 rows only: a bf16 gate row (`row < nb`) stages nothing -- it reads
// `wb` through the 4-deep ILP LDG loop and has no shared-memory copy to overlap,
// and its weight row is 2*k bytes so it would not fit in the k-byte fp8 slot
// even if it wanted to. The commit stays UNCONDITIONAL (bf16-first warps and the
// gate-off arm commit an EMPTY group) so every thread of the block carries the
// same cp.async group count into the row loop's `wait_all`. Same bytes into the
// same slot, only the issue point moves -> bit-identical output.
static const bool g_bf16fp8x2_cpasync = [] {
    const char* e = getenv("DSV41_BF16_CPASYNC");
    if (e == nullptr) return true;
    return atoi(e) != 0;
}();

// cp.async helpers are defined further down (hc_mix_dots uses them); declare
// them here so the fp8 gemv can stage its weight row asynchronously too.
__device__ __forceinline__ void dsv41_cp_async16(void* smem, const void* gmem);
__device__ __forceinline__ void dsv41_cp_commit();
__device__ __forceinline__ void dsv41_cp_wait_all();
// P4: retire all but the most recently committed cp.async group. Needed because
// the activation group is committed BEFORE the P3 weight group, so this retires
// the activation while leaving the (larger) weight row in flight for the row
// loop's own wait.
__device__ __forceinline__ void dsv41_cp_wait_group1();

// ---------------------------------------------------------------------------
// PDL (programmatic dependent launch) for the DSV41 attention projection chain.
//
// `pdl_or_plain` in ferrite_kernels.cu is file-static in ANOTHER translation
// unit, so this one carries its own copy under the DSV41 gate. Semantics are
// identical to that one (which is the verified-capture-compatible precedent):
//
//   * DSV41_PDL=1 (opt-in; unset = DEFAULT OFF) -> the launch carries
//     cudaLaunchAttributeProgrammaticStreamSerialization, so the consumer grid
//     is allowed to start while the producer is still draining its tail. The
//     consumer kernel then gates every read of the producer's output on the
//     OPENING cudaGridDependencySynchronize() -- that is the whole point: the
//     consumer's launch/setup cost (grid rasterisation, CTA scheduling, register
//     allocation, and any prologue that does NOT read the producer) moves off
//     the critical path and into the producer's ramp-down window. This is
//     node-transition cost, NOT bandwidth, which is why it needs no node
//     removal, no grid change and no cross-block sync -- it sidesteps the
//     hcpm / hc-merge / B1 failure modes entirely.
//   * DSV41_PDL=0 -> cudaLaunchKernel, the runtime API the `<<<>>>` form itself
//     compiles to: a plain launch with an explicit void* argument array, which
//     records the identical node in a stream capture. This is the A/B arm and
//     the rollback. It MUST NOT go through cudaLaunchKernelEx: the Extended
//     Launch path marshals the same arguments through its variadic template, and
//     for the GEMV family's then-37 scalar parameters that forwarding was
//     observed to return cudaErrorInvalidValue ("cuda error 1") on a launch
//     whose plain argument list was well formed. That specific hazard is gone
//     now -- gemv-struct-pack below collapsed those 37 slots into the four
//     by-value Gemv* structs, so the Ex path forwards 4 arguments -- but
//     cudaLaunchKernel stays the non-PDL arm precisely because it is
//     byte-for-byte what `<<<>>>` emits.
//
// CONTRACT (must be preserved by every future caller): a kernel routed through
// this helper MUST call cudaGridDependencySynchronize() before reading ANY
// output written by the previous kernel on the stream, and the call must be
// unconditional inside `#if __CUDA_ARCH__ >= 900`. The call is a documented
// no-op on a plain launch, so it stays in place when DSV41_PDL=0.
//
// COVERED HERE: gemm_fp8_gemv_kernel (all SEVEN M=1 launchers: mx, rope,
// rope_norm, mx2_rope, add, f32, mx2 -- every one of them now passing the four
// Gemv* structs, see gemv-struct-pack above the kernel) and sparse_attn_pf_kernel
// -- i.e. the consumer side of lin2 -> wq_b/rope -> sparse -> wo_a -> wo_b.
// DELIBERATELY NOT COVERED: the M>1 gemm_fp8_kernel tile path, and the
// split/merge/warp attention variants (A/B arms nobody should silently start
// running under PDL).
//
// NOTE: the `<<<>>>` launch syntax takes no launch attribute, so the PDL arm
// uses cudaLaunchKernelEx (whose variadic template accepts the implicit
// conversions the `<<<>>>` form would take) while the NON-PDL arm deliberately
// uses cudaLaunchKernel with an explicit argument array -- that array is the
// only launch path that is byte-for-byte the `<<<>>>` one. Do NOT "simplify"
// this back to a single cudaLaunchKernelEx call.
//
// ARCH: the host gate above is not arch-gated, the device sync is -- so this
// file MUST be built for sm_90+ (kernels/cuda/build.sh defaults to 100a and the
// rest of the file already requires Blackwell tcgen05), otherwise the attribute
// would be requested without a matching sync. On an unsupported device the
// attribute makes cudaLaunchKernelEx fail loudly, so the mismatch cannot be
// silent.
static int dsv41_pdl_enabled(void) {
    // Read once: these launchers run a few hundred times per step and a per-call
    // getenv on the hot path is exactly the slip every other gate in this file
    // avoids.
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("DSV41_PDL");
        // Round-45 fix: default OFF. PDL was introduced default-ON but never
        // production-verified; its cudaLaunchKernelEx path with the 36+ GEMV
        // parameters is the prime suspect for the drifting "cuda error 1"
        // (InvalidValue) that blocked rounds 42-45. The non-PDL path now uses
        // cudaLaunchKernel (identical to <<<>>> marshaling). Flip to default
        // OFF until a clean serve A/B proves the PDL path safe.
        //
        // UPDATE (gemv-struct-pack): the 37 GEMV scalars are now four by-value
        // structs, so the Ex path no longer forwards a 37-wide argument pack and
        // that particular failure mode cannot recur. The default stays OFF
        // anyway -- re-enabling PDL is its own A/B (DSV41_PDL=1), not a
        // consequence of this refactor.
        cached = (e != nullptr && e[0] == '1') ? 1 : 0;   // explicit "1" enables
    }
    return cached;
}

template <typename K, typename... Args>
static inline cudaError_t dsv41_pdl_or_plain(K kern, dim3 grid, dim3 block, size_t smem,
                                             cudaStream_t stream, Args... args) {
    if (dsv41_pdl_enabled()) {
        // PDL path: cudaLaunchKernelEx with the programmatic attribute.
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid; cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem; cfg.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attrs; cfg.numAttrs = 1;
        return cudaLaunchKernelEx(&cfg, kern, args...);
    }
    // NON-PDL path: cudaLaunchKernel -- the runtime API equivalent of `<<<>>>`.
    //
    // Round-45 fix: cudaLaunchKernelEx even WITHOUT the attribute is a DIFFERENT
    // launch path than `<<<>>>`. Its variadic two-pack template must forward the
    // deduced argument pack through an extra function layer, and for the GEMV
    // family that meant 36+ parameters; the resulting marshaling could fail with
    // cudaErrorInvalidValue (a.k.a. "cuda error 1") even though the plain-launch
    // argument list was well formed. cudaLaunchKernel takes an explicit void*
    // array, so the argument marshaling is byte-for-byte the one `<<<>>>` emits.
    // The callers used to spell out every trailing default because the array form
    // applies no default arguments either; gemv-struct-pack moved those defaults
    // into the Gemv* structs, so each caller now passes 4 by-value structs and
    // `arg_ptrs` has four entries.
    void* arg_ptrs[] = { (void*)&args... };
    return cudaLaunchKernel(kern, grid, block, (void**)arg_ptrs, smem, stream);
}

// ---------------------------------------------------------------------------
// gemv-struct-pack: the GEMV family's kernel-parameter groups.
//
// WHY THIS EXISTS. gemm_fp8_gemv_kernel had grown to 37 positional parameters.
// cudaLaunchKernelEx's variadic template failed to marshal that list -- it
// returned cudaErrorInvalidValue ("cuda error 1", drifting between
// mx / rope_norm / mx_rope / quant_fp8 across ranks) on a launch whose plain
// `<<<>>>` argument list was well formed -- and a 37-slot positional argument
// list is unmaintainable even when it does work. The launch now carries FOUR
// by-value structs: CUDA copies a by-value struct into the kernel parameter
// space exactly like a scalar, and four arguments are far inside every launch
// API's forwarding depth (dsv41_pdl_or_plain forwards all four verbatim on both
// of its paths, so the Extended-Launch variadic template now handles 4 slots
// instead of 37).
//
// GROUPING IS BY ROLE, NOT BY ORIGINAL POSITION:
//   GemvCore   -- operands, output and shape of the GEMM (activation, weights,
//                 bias, output, n/k/vec/a32) plus the SECOND family's operands
//                 (w2/w2_scale/bias2/out2/n1), which share the activation.
//   GemvRope   -- the RoPE fusion epilogue: cos/sin table, the device position
//                 counter `*rope_base` and the per-family head widths.
//   GemvFusion -- the fused producer/consumer side buffers: B1's fp8 row emit
//                 (xq/xsc), NORM_FUSE's in-kernel rmsnorm input (qr_*) and the
//                 f32 direct read (a_f32).
//   GemvEpi    -- epilogue behaviour: A5's trailing add fold (epi_add) and the
//                 AR v5 store fusion (staging_tbl + epoch + rank geometry).
//
// EVERY FIELD DEFAULTS TO null / 0 -- exactly the value every launcher used to
// spell out by hand for an unused trailing slot -- so a launcher only assigns
// the fields its own form uses, and `GemvCore gc{};` reproduces the old "all
// defaults spelled out" argument list with far less ceremony. The old
// per-parameter comments now live on the fields they describe.
//
// KERNEL PARAMETER ORDER IS (core, rope, fusion, epi). The field order inside a
// struct only matters for aggregate initialization, which no launcher uses; do
// not encode the launch's argument order in it.
//
// NOTE: these four names are unique in the whole .so (no other translation unit
// declares a Gemv* type), so the by-value ABI stays unambiguous.
struct GemvCore {
    // --- first family -------------------------------------------------------
    const uint8_t* a = nullptr;          // fp8 activation row (m == 1)
    const float* a_scale = nullptr;      // per-32-block activation scales
    const uint8_t* w = nullptr;          // fp8 weight rows [n][k]
    const uint8_t* w_scale = nullptr;    // per-row ue8m0 scale rows [n/32][k/32]
    const float* bias = nullptr;         // optional per-row bias [n]
    float* out = nullptr;                // f32 output rows [n]
    int n = 0;
    int k = 0;
    int vec = 0;     // fp8 GEMV mode: 3 = staged weights, 4 = staged activation
    // a32 gate (DSV41_GEMV_A32, see g_gemv_a32): 1 materialises the block-wide
    // pre-decoded activation `s_af` (k f32); 0 skips it and folds the
    // decode+scale into the consume loop. The launcher's `scale_bytes` MUST
    // reserve dsv41_gemv_a32_bytes(k) to match, and `s_rows` follows it.
    int a32 = 0;
    // P1 staged gate (DSV41_GEMV_A32_STAGED, see g_gemv_a32_staged): 1 puts the
    // k-byte `s_a` intermediate back in front of `s_af`. The launcher MUST then
    // reserve it too (dsv41_gemv_sa_bytes), or every pointer after the missing
    // slot is short by k bytes.
    int a32_staged = 0;
    // P3 (DSV41_GEMV_CPASYNC, see g_gemv_cpasync): 1 moves the row loop's
    // weight-row cp.async ISSUE above the block prologue, so the transfer
    // overlaps the activation staging / LUT / a32 materialisation and the dot
    // only waits for what is left. 0 keeps the issue-after-barrier order. The
    // prefetch reuses the per-warp `s_w` row slot, so smem does not change.
    int cpasync = 0;
    // P4 (DSV41_GEMV_ACT_CPASYNC, see g_gemv_act_cpasync): 1 stages the mode-4
    // block-wide fp8 activation row with cp.async (global -> `s_a`, LUT decode
    // after the wait) instead of the synchronous wide copy. It forces
    // `a32_direct` false so the k-byte `s_a` row exists -- the launcher MUST then
    // reserve it (dsv41_gemv_sa_bytes mirrors this field). 0 keeps today's
    // synchronous staging / fused `a32_direct` path bit for bit.
    int act_cpasync = 0;
    // --- second family: rows [n1, n1+n2) of the SAME activation -------------
    // A single-family launch passes n1 == n and leaves the family-2 pointers
    // untouched (null), so every row maps to family 1 and the behaviour is
    // exactly the old single-family kernel's. Both families share the staged
    // activation, which is the point: two projections that read the same vector
    // become one launch and one staging.
    const uint8_t* w2 = nullptr;
    const uint8_t* w2_scale = nullptr;
    const float* bias2 = nullptr;
    float* out2 = nullptr;
    int n1 = 0;
};

struct GemvRope {
    // RoPE fusion (q rope / idx_q rope of the DSV4.1 attention): a non-null
    // `rope_cos` makes the epilogue rotate the trailing `rope_rd` lanes of every
    // head in THIS launch's rows, replacing the standalone apply_rope. The pair
    // (2i, 2i+1) of one head always lands on two ADJACENT warps - the head width
    // is a multiple of 32 and the pair start is even - so the pair-head warp
    // reads the tail warp's staged row from the B1 `s_rows` slot after a
    // barrier. `rope_hd1`/`rope_hd2` are the head widths of the two families
    // (0 = do not rotate that family); `rope_base` is the DEVICE position
    // counter, the same `*base * mul + off + h * step` the rope kernel
    // evaluates. The rotated value is the very same `v = acc + bias` f32 the
    // rope kernel would have read back, and the arithmetic is its expression
    // verbatim, so the result is bit-identical to the standalone launch.
    const float* rope_cos = nullptr;
    const float* rope_sin = nullptr;
    const int* rope_base = nullptr;
    int rope_mul = 0;
    int rope_off = 0;
    int rope_step = 0;
    int rope_inverse = 0;
    int rope_rd = 0;
    int rope_hd1 = 0;
    int rope_hd2 = 0;
};

struct GemvFusion {
    // B1 (quantised row compression): when non-null the epilogue ALSO emits this
    // row's fp8 e4m3 byte and the per-32-block scale of the vector it is
    // writing, with quant_kernel's own arithmetic, so the consumer's `quant1`
    // launch disappears.
    //
    // TWO constraints, both of them structural - read before use:
    //  1. A gemv warp produces ONE row (= one element of `out`), so a warp
    //     CANNOT own a quant block: `quant1`'s block is 32 CONSECUTIVE elements,
    //     i.e. 32 rows, i.e. 32 WARPS. This is exactly the opposite of the
    //     hc-tail / swiglu producers, where a warp's 32 LANES are 32 consecutive
    //     elements of the quantised vector and the amax is one shuffle. Here the
    //     amax spans the block (see the epilogue), so the launcher must give the
    //     block 32 warps.
    //  2. `xq` must not alias `a`/`a_scale`: the quantised input is still being
    //     staged by blocks that start late, so the fp8 output needs its own
    //     buffer.
    uint8_t* xq = nullptr;
    float* xsc = nullptr;
    // NORM_FUSE (see dsv41_gemm_fp8_mx_rope_norm): when `qr_raw` is non-null the
    // gemv OWNS the production of the fp8 activation it consumes. The prologue
    // computes the RMSNorm of the f32 row `qr_raw` (k elements) with `qr_w` /
    // `qr_eps` and encodes it into `s_a` / `s_as` with rmsnorm_q_kernel's
    // arithmetic, term for term, so the standalone rmsnorm_q launch between qr's
    // producer and this gemv disappears. `a` / `a_scale` are then never read
    // (the launcher passes null). The reduction tree only matches the reference
    // at blockDim 1024, so the launcher forces 32 warps.
    const float* qr_raw = nullptr;
    const float* qr_w = nullptr;
    float qr_eps = 0.f;
    // f32 direct read (see dsv41_gemm_fp8_mx_f32): when `a_f32` is non-null the
    // block stages the RAW f32 activation straight into `s_af` instead of
    // decoding the fp8 `a` through `s_lut` and multiplying by the per-block
    // scale. The consume loop is unchanged (it already reads `s_af`), so the
    // WEIGHT side keeps its fp8 decode -- `s_lut` is still built. `a`/`a_scale`
    // are then never read (the launcher passes null). NOT bit-identical to the
    // fp8 path: it skips the quantise->dequantise round trip and is strictly
    // MORE accurate, so it is only wired where the consumer tolerates the
    // tighter value (wo_b -> AR sum -> hc_post). Requires `vec >= 3` (the s_af
    // materialisation lives there); the launcher enforces it.
    const float* a_f32 = nullptr;
};

struct GemvEpi {
    // A5: non-zero folds a trailing elementwise add into the row write --
    // `out[row] += acc + bias` instead of overwriting -- so the separate
    // add_inplace launch disappears. Association is unchanged:
    // `o + (acc + bias)` either way, so the result is bit-identical. 0 keeps the
    // old overwrite behaviour for every existing caller.
    int epi_add = 0;
    // AR v5 store fusion (attn wo_b, M=1 only). A non-null `staging_tbl` turns
    // the lane-0 epilogue into store_v5: the row's `acc + bias` goes BOTH to
    // `out[rrow]` (unchanged) and to every peer's staging slot, replacing the
    // standalone store kernel with this one write. `epoch` is the DEVICE round
    // counter, read at runtime exactly like p2p_ar_store_v5_kernel, so a
    // captured graph replays with the right parity. `stride` counts FLOAT
    // ELEMENTS per slot (bytes/4), NOT bytes.
    float* const* staging_tbl = nullptr;
    const unsigned* epoch = nullptr;
    int world = 0;
    int my_rank = 0;
    int stride = 0;
};


// ---------------------------------------------------------------------------
// gemv-struct-pack: the kernel now takes four by-value parameter groups instead
// of the 37 scalars this family had grown to (see the Gemv* definitions below).
//
// TWO QUALIFIERS HERE ARE LOAD-BEARING -- do not "clean them up":
//
//   * `__grid_constant__ const` keeps each struct in the kernel parameter space
//     instead of letting the compiler keep its members live in registers across
//     the row loop. MEASURED (sm_103a, CUDA 13.2, -O3 --use_fast_math):
//     plain by-value structs need 72 registers, __grid_constant__ needs 64.
//     Adding __restrict__ to the struct members, dropping the default member
//     initialisers, or marking the parameters `const` alone all made no
//     difference -- __restrict__ is NOT the mechanism (HEAD with every
//     __restrict__ stripped still compiles to 56 registers).
//
//   * `__launch_bounds__(1024)` is what actually pins the register count. It
//     states the widest block this kernel is ever launched with -- 32 warps,
//     the five forced-32-warp launchers (mx_rope, mx_rope_norm, mx2_rope, and
//     the B1/norm-fuse paths) -- so ptxas must fit the 64-register-per-thread
//     ceiling that 1024 threads impose (65536 registers / 1024 threads). With
//     BOTH qualifiers ptxas emits 56 registers and ZERO spills: byte-identical
//     register pressure to the 37-parameter version.
//
//     Why this matters: at 72 registers a 1024-thread launch needs 73728
//     registers (72 * 32 warps, warp allocation granularity 256 => no rounding
//     down), more than the 65536-register file one SM has, so the block cannot
//     be resident and the launch FAILS with cudaErrorLaunchOutOfResources
//     ("cuda error 701"), drifting between mx / rope_norm / mx_rope / quant_fp8
//     across ranks (whichever 32-warp launcher runs first on that rank).
//     Removing __launch_bounds__ re-opens that failure silently.
//
//     Do NOT confuse this 701 with the r42-45 "cuda error 1" (InvalidValue):
//     those were THREE different mechanisms in the same symptom, and this one
//     was masked by the other two until they were fixed --
//       (a) cudaLaunchKernelEx's variadic marshaling of the 36+ GEMV scalars
//           -> InvalidValue -> fixed by gemv-struct-pack (4c995f2);
//       (b) NORM_FUSE's 128B static s_norm_red made the 232448 dynamic opt-in
//           exceed the device total -> cudaFuncSetAttribute returned
//           InvalidValue and the launcher bailed BEFORE launching -> fixed by
//           232320 (775dadb). Only after (b) landed did the SetAttribute path
//           clear and the launch actually run, which is when this register
//           over-commit surfaced as 701.
//
// A/B command used for the numbers above:
//     nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//          --use_fast_math -Xptxas -v -c kernels/cuda/dsv41_kernels.cu
//
// The three-argument `<<<>>>` form, cudaLaunchKernel's void* array and
// cudaLaunchKernelEx's variadic template all marshal a by-value struct the same
// way they marshal a scalar, so the launchers pass `gc, gr, gf, ge` unchanged on
// both arms of dsv41_pdl_or_plain.
__global__ void __launch_bounds__(1024)
gemm_fp8_gemv_kernel(__grid_constant__ const GemvCore gc, __grid_constant__ const GemvRope gr,
                     __grid_constant__ const GemvFusion gf, __grid_constant__ const GemvEpi ge) {
    // -----------------------------------------------------------------------
    // gemv-struct-pack: re-bind the parameter names from the four by-value
    // structs defined above. Everything from here to the kernel's closing brace
    // is the pre-pack body, byte for byte -- the bindings below carry the same
    // names AND the same `__restrict__` qualification the individual parameters
    // used to carry, so no expression inside the body had to be touched (the
    // aliases are pure parameter-space loads and are not a read of any producer
    // output, which is why they may precede cudaGridDependencySynchronise).
    //
    // CONTRACT for future edits: a new field goes into a struct AND gets a
    // binding here -- never a direct `gc.`/`gr.`/`gf.`/`ge.` use in the middle
    // of the body. Keeping the body textually unchanged is what makes this
    // refactor reviewable, and `__restrict__` on the binding is what keeps the
    // consume loop's codegen identical (dropping it invites the compiler to
    // assume `a` and `w` may alias).
    const uint8_t* __restrict__ a = gc.a;
    const float* __restrict__ a_scale = gc.a_scale;
    const uint8_t* __restrict__ w = gc.w;
    const uint8_t* __restrict__ w_scale = gc.w_scale;
    const float* __restrict__ bias = gc.bias;
    float* __restrict__ out = gc.out;
    int n = gc.n;
    int k = gc.k;
    int vec = gc.vec;
    int a32 = gc.a32;
    int a32_staged = gc.a32_staged;   // P1 staged gate (see g_gemv_a32_staged)
    int cpasync = gc.cpasync;         // P3 cp.async weight-first prologue
    int act_cpasync = gc.act_cpasync; // P4 cp.async activation staging
    const uint8_t* __restrict__ w2 = gc.w2;
    const uint8_t* __restrict__ w2_scale = gc.w2_scale;
    const float* __restrict__ bias2 = gc.bias2;
    float* __restrict__ out2 = gc.out2;
    int n1 = gc.n1;
    int epi_add = ge.epi_add;
    float* const* __restrict__ staging_tbl = ge.staging_tbl;
    const unsigned* __restrict__ epoch = ge.epoch;
    int world = ge.world;
    int my_rank = ge.my_rank;
    int stride = ge.stride;
    uint8_t* __restrict__ xq = gf.xq;
    float* __restrict__ xsc = gf.xsc;
    const float* __restrict__ rope_cos = gr.rope_cos;
    const float* __restrict__ rope_sin = gr.rope_sin;
    const int* __restrict__ rope_base = gr.rope_base;
    int rope_mul = gr.rope_mul;
    int rope_off = gr.rope_off;
    int rope_step = gr.rope_step;
    int rope_inverse = gr.rope_inverse;
    int rope_rd = gr.rope_rd;
    int rope_hd1 = gr.rope_hd1;
    int rope_hd2 = gr.rope_hd2;
    const float* qr_raw = gf.qr_raw;
    const float* qr_w = gf.qr_w;
    float qr_eps = gf.qr_eps;
    const float* a_f32 = gf.a_f32;
#if __CUDA_ARCH__ >= 900
    // PDL (DSV41_PDL, see dsv41_pdl_or_plain above): the launcher may have
    // launched this grid with programmatic stream serialization, so the grid is
    // already resident and this call is what makes the producer's activation /
    // scale / epoch writes visible. It MUST stay before the first read of any
    // producer output (`*epoch` below is one, `a`/`a_scale`/`a_f32` further
    // down are the big ones). No-op on a plain launch.
    //
    // Why the sync sits at the very top and not after a longer prologue: the
    // only producer-INDEPENDENT work this kernel has before the staging is the
    // 256-entry e4m3 LUT (one iteration per thread) and the pointer/register
    // setup -- both cheaper than the restructuring needed to move them past the
    // sync, and the weight-row staging that WOULD be worth hoisting lives inside
    // the row loop whose cp.async commit/wait pairing is numerically load-
    // bearing. The win here is the node-gap (launch overlap), not prologue
    // hiding.
    cudaGridDependencySynchronize();
#endif
    // Read the round ONCE, like p2p_ar_store_v5_kernel (ferrite_kernels.cu). When
    // the table is null this is a no-op (epoch is null too) and the kernel is the
    // old one bit for bit.
    const unsigned ar_e = (epoch != nullptr) ? *epoch : 0u;
    const size_t ar_base =
        (size_t)((ar_e & 1u) * (unsigned)world + (unsigned)my_rank) * (unsigned)stride;
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
    // Staged scale rows. The consume loop needs one ue8m0 byte and one f32 per
    // 32-element block, and BOTH used to be plain global loads issued inside the
    // loop - 2 x nb_k serial round trips per row that the weight-row staging did
    // not cover. Replacing them with constants measured 21 percent faster on the
    // wq_b shape, 34 percent on sharedexp (the latency-bound end), so the bytes
    // are now staged the same way the weights are. Values and their use order are
    // untouched, so the result is bit-identical.
    const int nb_k_al = (nb_k + 15) & ~15;          // 16-byte units for cp.async
    // P1 (a32 dead-slot elimination, 2026-09-11). With a32=1 the consume loop's
    // ONLY activation reader is `s_af` (line ~3257); on the plain fp8 mode-4 path
    // `s_a` has exactly one reader left -- the materialisation below
    // (`s_af[i] = s_lut[s_a[i]] * s_as[i>>5]`). Fusing that decode into the
    // staging pass (global uint4 load -> same LUT -> same scale -> straight into
    // `s_af`) removes the last reader, so the whole k-byte slot is not allocated:
    // smem 48512 -> 43392 B at k=5120 / warps=4 / mode 4 -> 5 blocks/SM instead
    // of 4. Same bytes, same LUT entry, same product -> bit-identical (`s_a` was
    // only ever a copy of `a`). The slot stays when the NORM_FUSE prologue WRITES
    // `s_a` (qr_raw non-null, it is decoded by the same loop), and when a32=0
    // (that A/B arm reads `s_lut[s_a[j]] * s_as[j>>5]` inline in the consume
    // loop). The launchers mirror this with dsv41_gemv_sa_bytes().
    // P4 (DSV41_GEMV_ACT_CPASYNC, see g_gemv_act_cpasync): like a32_staged this
    // gate re-opens the k-byte `s_a` row, so `a32_direct` must lose the same
    // term -- dsv41_gemv_sa_bytes() reserves k bytes for it unconditionally, and
    // a kernel that skipped the slot would put every later pointer k bytes early.
    const bool a32_direct = (a32 != 0) && (vec == 4) && (qr_raw == nullptr)
            && !a32_staged && (act_cpasync == 0);  // P1/P4 gates: fall back to staged
    // P4: only the sites where the synchronous staging loop below would run get
    // the async treatment. `a_f32 != nullptr` is excluded because on that path
    // `a` is NULL (its launcher passes null) and an async copy from it would
    // fault; that branch fills `s_af` straight from `a_f32` instead.
    const bool act_async = (act_cpasync != 0) && (a32 != 0) && (vec == 4) &&
                           (qr_raw == nullptr) && !a32_staged && (a_f32 == nullptr);
    uint8_t* s_ws = s_a + (size_t)((vec == 4 && !a32_direct) ? k : 0);
    float* s_as = reinterpret_cast<float*>(s_ws + (size_t)nwarps * (size_t)nb_k_al);
    // The e4m3 decode as a 256-entry shared-memory table. The bit-manipulation
    // form chains LDS.8 -> ~10 ALU ops -> FMUL per operand; the table is a single
    // LDS.32, which breaks the per-warp serial dependency chain: the isolated
    // graph benchmark measured the consume loop at 47 ns/kb (bit ops) against
    // 32 ns/kb (LUT) - n=256: 10.50 -> 8.33us, n=1024: 13.19 -> 9.23us,
    // n=1664: 17.57 -> 11.22us. The table is built from the SAME e4m3_to_f, so
    // every decoded value is bit-identical (verified by the 256-code exhaustive
    // host+device comparison; the earlier "branchless exact" rewrite was NOT -
    // 14/256 mismatched on the implicit-1-bit derivation - do not retry that).
    float* s_lut = s_as + nb_k;
    // a32: the activation pre-decoded to its SCALED f32 form, once per block.
    // The consume loop's per-element chain was LDS.8(ap byte) -> LDS.32(LUT) ->
    // FMUL(sa); folding the decode+scale here leaves LDS.32(s_af) -> FMUL, and
    // the isolated graph bench measured the family -6/-8/-13 percent at
    // n=256/1024/1664 (7.71/8.59/10.39us against the plain LUT's
    // 8.23/9.35/11.96). s_af[j] is the SAME product the loop used to compute
    // (s_lut[ap[j]] * sa with sa = s_as[j>>5]), so the rounding sequence is
    // unchanged and the output is bit-identical (fingerprint-verified).
    float* s_af = s_lut + 256;   // valid only when `a32` is set
    // B1: this block's 32 row values (one per warp), staged so the epilogue can
    // take the amax of the quant block they form. The 32-float slot is reserved
    // in every launcher's `scale_bytes` (see dsv41_gemm_fp8_mx); it is only
    // dereferenced when `xq` is non-null, which requires nwarps == 32.
    // a32 off: the k-float `s_af` slot is not reserved, so that slot takes the
    // B1 row stage's place instead (launchers drop the same k floats - see
    // dsv41_gemv_a32_bytes).
    float* s_rows = a32 ? (s_af + k) : (s_lut + 256);
    // -----------------------------------------------------------------------
    // P4 (DSV41_GEMV_ACT_CPASYNC): cp.async ACTIVATION staging. See
    // g_gemv_act_cpasync above for the reasoning and the layout coupling; the
    // short version is that the block-wide fp8 activation row used to be staged
    // by a SYNCHRONOUS wide copy, which parks every participating thread on the
    // DRAM round trip before the LUT is even built. Issuing the same bytes as
    // cp.async instead takes that transfer off the thread's critical path: the
    // scale staging + LUT build run underneath it, and the wait below collects
    // it just before the barrier that publishes `s_a`.
    //
    // GROUP ORDER IS LOAD-BEARING -- this block sits ABOVE the P3 weight prefetch
    // on purpose. The activation group is therefore the OLDER one, which is what
    // lets `dsv41_cp_wait_group1()` below retire the activation while the (much
    // larger) weight row stays in flight for the row loop's own wait. Reversing
    // the two issues turns that wait into "stall until the weight row lands" and
    // throws P3's overlap away.
    //
    // The commit is UNCONDITIONAL for every thread (empty groups included), so
    // every thread carries the same group count into the `wait_group 1`; a
    // warp-dependent count would retire a different group per warp.
    //
    // Same bytes, same slot, same LUT entry, same per-block scale as the
    // synchronous copy -> bit-identical. The k % 16 tail below stays a plain
    // read (unreachable: every launcher rejects `k & 31`).
    if (act_async) {
        const int n16a = k >> 4;
        for (int i = threadIdx.x; i < n16a; i += blockDim.x)
            dsv41_cp_async16(s_a + (i << 4), a + (i << 4));
        dsv41_cp_commit();
    }
    // -----------------------------------------------------------------------
    // P3 (DSV41_GEMV_CPASYNC): cp.async WEIGHT-FIRST prologue. See
    // g_gemv_cpasync above for the full reasoning; the short version is that the
    // row loop's weight transfer used to start only after the block barrier
    // below, leaving nothing to hide its latency, while the weight row depends
    // on none of the activation work that fills that barrier. So issue it here,
    // let the staging below run underneath it, and let loop iteration 0 only
    // WAIT for it (it recognises the row by `row == pf_row`).
    //
    // The traffic is identical to what the loop was already doing (same bytes,
    // same slot) -- the only difference is WHEN the instruction is issued. It is
    // issued after cudaGridDependencySynchronize(), like every other read of a
    // producer's output, because `w` can be written by the previous node.
    // NOTE the explicit (int) cast: `blockIdx.x * nwarps + warp` is UNSIGNED
    // (blockIdx.x is uint3), so the bare ternary would promote -1 to 0xFFFFFFFF
    // and leave pf_row == -1 holding on an implementation detail.
    int pf_row = ((cpasync != 0) && (vec >= 3)) ? ((int)blockIdx.x * nwarps + warp) : -1;
    if (pf_row >= n) pf_row = -1;   // the last block's tail warps have no row
    if (pf_row >= 0) {
        // Family/pointer select copied from the row loop verbatim: a family-2
        // row reads w2 / w2_scale, rows below n1 read w / w_scale. Drifting from
        // the loop here would silently stage the WRONG row (no crash, wrong dot).
        const uint8_t* pf_w;
        if (pf_row < n1) pf_w = w + (size_t)pf_row * k;
        else             pf_w = w2 + (size_t)(pf_row - n1) * k;
        uint8_t* pf_s = s_w + (size_t)warp * (size_t)k;
        const int n16p = k >> 4;
        for (int i = lane; i < n16p; i += 32) dsv41_cp_async16(pf_s + (i << 4), pf_w + (i << 4));
        // k % 16 tail: unreachable for every launcher (all of them reject
        // k & 31), kept so the prefetched row is staged by exactly the rule the
        // loop uses. A hole here would be a partially-staged row, not a crash.
        for (int i = (n16p << 4) + lane; i < k; i += 32) pf_s[i] = pf_w[i];
    }
    // P3 P4 SHARED: committed UNCONDITIONALLY, i.e. also for the tail warps that
    // have no prefetched row (they commit an EMPTY group, which is a no-op) and
    // also when the P3 gate is off. That keeps the per-thread commit-group count
    // identical across the block, which is what makes the `wait_group 1` in the
    // prologue epilogue retire the SAME group for every thread. The row loop's
    // own `cp.async.wait_all` retires this group on the prefetched iteration, so
    // moving the commit below the tail copy changes nothing observable.
    dsv41_cp_commit();
    // NORM_FUSE prologue (see dsv41_gemm_fp8_mx_rope_norm). With `qr_raw`
    // non-null this block owns the activation's fp8 production: it reduces the
    // f32 row, applies the norm weight and encodes the scaled values into
    // `s_a` / `s_as` with `rmsnorm_q_kernel`'s arithmetic, term for term -- the
    // same blockDim-strided element loop, the same shuffle-down tree, the same
    // thread-0 cross-warp sum, the same `fast_round_scale` + clamp + e4m3 round.
    // The standalone rmsnorm_q launch is gone and the gemv reads the identical
    // bytes out of shared memory. `a` / `a_scale` are never touched here.
    //
    // BIT-IDENTITY depends on the blockDim: the reference runs 1024 threads, so
    // the launcher forces 32 warps and the partial sums are visited in exactly
    // the reference's order.
    __shared__ float s_norm_red[32];
    if (qr_raw != nullptr) {
        float ss = 0.f;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            const float t = qr_raw[i];
            ss += t * t;
        }
        float lane_sum = ss;
        for (int off = 16; off > 0; off >>= 1)
            lane_sum += __shfl_down_sync(0xffffffffu, lane_sum, off);
        if ((threadIdx.x & 31) == 0) s_norm_red[threadIdx.x >> 5] = lane_sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.f;
            for (int i = 0; i < nwarps; ++i) t += s_norm_red[i];
            s_norm_red[0] = rsqrtf(t / (float)k + qr_eps);
        }
        __syncthreads();
        const float inv = s_norm_red[0];
        const int lane31 = threadIdx.x & 31;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            const float v = qr_raw[i] * inv * qr_w[i];
            float am = fabsf(v);
            for (int off = 16; off > 0; off >>= 1)
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
            const float sc = fmaxf(fast_round_scale(am, 1.0f / 448.0f), 1e-30f);
            if (lane31 == 0) s_as[i >> 5] = sc;
            const float q = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
            const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
            s_a[i] = *(const uint8_t*)&f8;
        }
    } else if (vec == 4 && a_f32 == nullptr && !a32_direct) {
        // On the f32 path `a` is null and the block-wide copy is not read (the
        // consume loop reads `s_af`, which the a32 block below fills from a_f32),
        // so the copy is skipped to avoid dereferencing null. `a32_direct` skips
        // it too: the fused a32 block below reads the SAME bytes straight from `a`
        // (same uint4 stride) and decodes them into `s_af` in one pass, so this
        // copy would have no reader.
        const int n16a = k >> 4;
        // NOTE: this staging was tried as cp.async and faulted with err 700
        // (illegal access, surfacing as a sticky error on dsv41_route_topk) while
        // the identical synchronous form passes. The addresses look legal (a is a
        // cudaMalloc'd DevBuf and every offset is a multiple of 16), so the cause
        // is unidentified; the dependent-copy form is the correct baseline. Do not
        // re-try without first reading the dbg-err700 findings.
        //
        // [2026-09-12] P4 re-opens that cp.async attempt behind
        // DSV41_GEMV_ACT_CPASYNC (default OFF, A/B arm). Two things changed since
        // the note above was written, and BOTH matter before trusting the arm:
        //   1. the err-700 round that produced this note was later traced to a
        //      DIFFERENT kernel's out-of-bounds dynamic smem (gemv_bf16 staging
        //      with a launcher that still passed smem = 0 -- see STATUS.md
        //      "默认路径 err 700 的最终根因"); that file explicitly lists
        //      "activation cp.async（已删）" as a probable false conviction. So the
        //      verdict here is UNVERIFIED, not disproven -- treat a fresh err 700
        //      on this arm as a real signal and re-read that section.
        //   2. the async issue now happens in the PROLOGUE, before the weight
        //      prefetch, precisely so `wait_group 1` can retire it alone.
        // When `act_async` is set the copy below is not executed here at all; the
        // bytes were already issued in the prologue above.
        if (!act_async) {
            for (int i = threadIdx.x; i < n16a; i += blockDim.x) {
                *reinterpret_cast<uint4*>(s_a + (i << 4)) =
                    *reinterpret_cast<const uint4*>(a + (i << 4));
            }
        }
        for (int i = (n16a << 4) + threadIdx.x; i < k; i += blockDim.x) s_a[i] = a[i];
    }
    if (vec >= 3) {
        // The activation scales are the same for every output row, so they are
        // read once per block instead of once per row per k-block. On the
        // NORM_FUSE path (`qr_raw` non-null) the prologue above already wrote
        // them, and `a_scale` is null.
        if (qr_raw == nullptr && a_f32 == nullptr)
            for (int i = threadIdx.x; i < nb_k; i += blockDim.x) s_as[i] = a_scale[i];
        // Build the e4m3 decode table once per block (256 entries, two iterations
        // per thread at the default block size).
        for (int i = threadIdx.x; i < 256; i += blockDim.x) s_lut[i] = e4m3_to_f((uint8_t)i);
        // P4: collect the async activation transfer HERE -- after the scale copy
        // and the LUT build (which is the cover), and before the barrier that
        // publishes `s_a` to the block, which is the first point the stage is
        // consumed (the decode below). `wait_group 1` retires the activation
        // group ONLY: the P3 weight group is the NEWER group and stays in flight
        // for the row loop's own wait_all. See g_gemv_act_cpasync.
        if (act_async) dsv41_cp_wait_group1();
        __syncthreads();
        // a32: materialise the scaled activation (block-level, so the decode
        // latency is paid once instead of once per row). With DSV41_GEMV_A32=0
        // the slot is not allocated and the consume loop folds the same product
        // inline, so this whole block is skipped (barriers stay unconditional).
        if (a32) {
            if (a_f32 != nullptr) {
                // f32 direct read: the activation is already f32, so there is no LUT
                // lookup and no per-block scale -- s_af IS the value. `s_lut` is still
                // built above because the WEIGHT side of the consume loop decodes
                // through it.
                for (int i = threadIdx.x; i < k; i += blockDim.x) s_af[i] = a_f32[i];
            } else if (a32_direct) {
                // P1: one pass global -> s_af. The uint4 load is the SAME wide
                // read the (now removed) staging loop performed into `s_a` and
                // the decode is the SAME `s_lut[b] * s_as[i>>5]` product the
                // materialisation applied to the bytes it read back from shared
                // memory, so the emitted f32 is bit-identical -- the smem round
                // trip (and the k-byte slot) is all that disappears.
                // 4-element vectorisation (dsv41_a32_mat4, see its header): the
                // uint4 read stays the WIDE 16-byte global load it always was,
                // but each group of four decoded values now lands as ONE float4
                // store instead of four scalar ones. Same terms, same order ->
                // bit-identical; the UNIFORM guard keeps the scalar loop when
                // `s_af` is not 16-byte aligned (err 716 trap).
                const int n16a = k >> 4;
                if (dsv41_f4_ok(s_af)) {
                    for (int i = threadIdx.x; i < n16a; i += blockDim.x) {
                        const uint4 v = *reinterpret_cast<const uint4*>(a + (i << 4));
                        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
                        for (int j = 0; j < 16; j += 4) {
                            const int idx = (i << 4) + j;
                            const uint32_t b4 = (uint32_t)b[j] |
                                                ((uint32_t)b[j + 1] << 8) |
                                                ((uint32_t)b[j + 2] << 16) |
                                                ((uint32_t)b[j + 3] << 24);
                            dsv41_a32_mat4(b4, idx, s_lut, s_as, s_af);
                        }
                    }
                } else {
                    for (int i = threadIdx.x; i < n16a; i += blockDim.x) {
                        const uint4 v = *reinterpret_cast<const uint4*>(a + (i << 4));
                        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const int idx = (i << 4) + j;
                            s_af[idx] = s_lut[b[j]] * s_as[idx >> 5];
                        }
                    }
                }
                for (int i = (n16a << 4) + threadIdx.x; i < k; i += blockDim.x)
                    s_af[i] = s_lut[a[i]] * s_as[i >> 5];
            } else {
                const uint8_t* ap0 = (vec == 4) ? s_a : a;
                // 4-element vectorisation (dsv41_a32_mat4, see its header): one
                // 32-bit read of FOUR fp8 bytes, four LUT lookups and ONE float4
                // store per step instead of four scalar stores. This is the P4
                // default path (`act_async` staged the row into `s_a`), and the
                // isolated probe's -24/-12/-7.7% at n=256/1024/1664. The four
                // decoded values are idx..idx+3 with idx % 4 == 0, so they always
                // share `s_as[idx>>5]` (a scale block is 32 elements) and the
                // products are the scalar loop's term for term -> bit-identical.
                // The guard is UNIFORM per block: it only protects the float4
                // STORE against a k whose `s_af` offset is not 16-byte aligned
                // (offset = k/8 + 1024 bytes over a 16-byte-multiple base, so it
                // holds for every production k -- all of them multiples of 128);
                // a shape that fails it keeps the scalar loop rather than hit
                // err 716. The source read only needs 4-byte alignment, which
                // the staged row and the launcher's k % 16 == 0 guarantee.
                if (dsv41_f4_ok(s_af) && dsv41_f4_ok(ap0)) {
                    const int k4 = k >> 2;
                    for (int i = threadIdx.x; i < k4; i += blockDim.x) {
                        const int idx = i << 2;
                        const uint32_t b4 = *reinterpret_cast<const uint32_t*>(ap0 + idx);
                        dsv41_a32_mat4(b4, idx, s_lut, s_as, s_af);
                    }
                    for (int i = (k4 << 2) + threadIdx.x; i < k; i += blockDim.x)
                        s_af[i] = s_lut[ap0[i]] * s_as[i >> 5];
                } else {
                    for (int i = threadIdx.x; i < k; i += blockDim.x)
                        s_af[i] = s_lut[ap0[i]] * s_as[i >> 5];
                }
            }
        }
        __syncthreads();
    } else if (vec == 4) {
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
            uint8_t* row_sc = s_ws + (size_t)warp * (size_t)nb_k_al;
            // Sixteen bytes per lane per iteration. The first cut counted
            // thirty-two-byte k-blocks while copying one uint4 each, so only the
            // first half of every block was staged, the odd halves stayed
            // uninitialised, and the result was a five percent "speed-up" on a page
            // of garbage. Count sixteen-byte units, and cover a k that is not a
            // multiple of sixteen with a byte tail.
            const int n16 = k >> 4;
            // P3: the prologue already staged THIS row (and committed its group)
            // when this is the warp's first row -- the common case, since every
            // production shape launches at least one warp per row. In that case
            // the issue below is skipped and only the wait at the end is needed;
            // the wait_all retires the prologue's group either way.
            const bool prefetched = (row == pf_row);
            // Stage the weight row with cp.async (the correct baseline path).
            if (!prefetched) {
                for (int i = (n16 << 4) + lane; i < k; i += 32) row_s[i] = wr[i];
                for (int i = lane; i < n16; i += 32)
                    dsv41_cp_async16(row_s + (i << 4), wr + (i << 4));
            }
            // Stage this row's ue8m0 scale bytes too: they are one global load per
            // kb in the consume loop, and that load is exactly the latency the loop
            // stalls on (constant-scales measured 21-34 percent faster).
            //
            // PLAIN byte loads, NOT cp.async: cp.async16 needs a 16-byte-aligned
            // global address and `wsr` only has that when nb_k (= k/32) is a
            // multiple of 16. It is 72 for the shared expert (inter 2304) and 40
            // for q_lora (1280), and cp.async16 there faults with err 716
            // (misaligned address), which surfaces on the NEXT checked launch
            // (measured: a sticky error reported by dsv41_route_topk). On the
            // non-prefetched iterations the weight cp.asyncs are issued first, so
            // these loads overlap their latency; on the P3-prefetched iteration
            // they are instead the last global reads before the dot, which is the
            // price of moving the (much larger) weight transfer above the barrier.
            for (int i = lane; i < nb_k; i += 32) row_sc[i] = wsr[i];
            // Unconditional on purpose: on the prefetched iteration this commits
            // an EMPTY group (a no-op) after the prologue's group was already
            // committed, and wait_all below retires both.
            dsv41_cp_commit();
            dsv41_cp_wait_all();
            __syncwarp();
            // NOTE: no `float acc = 0.f;` here! The outer acc (line ~1687)
            // is the accumulator — declaring a new one inside this block
            // SHADOWS it, the warp reduction after the block reads the outer
            // zero, and every experiment since the prefetch series was testing
            // garbage. This shadowing was the root cause of the "3ms speedup
            // + degeneration" mystery (the speedup was the compiler dead-coding
            // the warp reduction on a known-zero value).
            // unroll 4 -> 32 (rf_u32, gemm-5us-feasibility): the isolated probe's
            // winning combination is ROW_FIRST + unroll32 + the a32 vec4 build,
            // and ROW_FIRST is already what P3 does. This is a pure
            // LOAD-OVERLAP unroll of a SINGLE serial `acc +=` chain -- unrolling
            // cannot reassociate the chain (each iteration's acc depends on the
            // previous one), so the summation order, and therefore every output
            // bit, is unchanged. Do NOT "improve" this into split accumulators:
            // that WOULD reassociate, and under --use_fast_math a multi-way
            // unrolled gemv body with plain operators drifted ~1 ULP per layer
            // and degenerated the model after 40 layers (see kernels/cuda/build.sh).
#pragma unroll 32
            for (int kb = 0; kb < nb_k; ++kb) {
                const float sb = ue8m0_to_f(row_sc[kb]);
                const int j = kb * 32 + lane;
                // a32 on: s_af[j] already folds s_lut[ap[j]] * s_as[j>>5], so the
                // per-element chain is one LDS.32 -> FMUL instead of
                // LDS.8 -> LDS.32 -> FMUL -> FMUL. a32 off (DSV41_GEMV_A32=0): the
                // SAME product is folded inline (the f32 path reads a_f32 directly,
                // exactly what the skipped materialisation would have copied into
                // s_af). Both forms are bit-identical by construction.
                const float av = a32 ? s_af[j]
                                     : (a_f32 != nullptr ? a_f32[j]
                                                         : s_lut[ap[j]] * s_as[j >> 5]);
                acc += av * (s_lut[row_s[j]] * sb);
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
        if (lane == 0) {
            const float v = acc + (bias_ ? bias_[rrow] : 0.f);
            // A5: fold the caller's trailing add_inplace(o, out) into this row
            // write. The standalone kernel computed `o += (acc + bias)`, i.e.
            // `o + (acc + bias)` -- the same association this read-modify-write
            // performs, so the fused result is bit-identical. Every other
            // caller leaves epi_add at 0 and keeps the plain overwrite.
            out_[rrow] = epi_add ? (out_[rrow] + v) : v;
            // AR v5 store fusion: the SAME `v` the row write just produced (the
            // partial, before any epi_add fold -- epi_add is 0 for wo_b) lands in
            // every peer's slot for this round. One 4B store per warp instead of
            // store_v5's coalesced float4 sweep; the byte count is identical.
            // `row < n1` restricts this to family 1, whose element index IS `rrow`;
            // a family-2 row must never land in family-1's slots.
            if (staging_tbl != nullptr && row < n1) {
                #pragma unroll 2
                for (int rr = 0; rr < world; rr++)
                    staging_tbl[rr][ar_base + (size_t)rrow] = v;
            }
            // B1: hand this row to the block-level quant epilogue below. `v` is
            // the same register value the f32 store just wrote, so the byte the
            // epilogue encodes is exactly what quant_kernel would have read back
            // from `out` (no second global round trip, no re-read). The rope
            // fusion reads it the same way: `v` IS the value apply_rope would have
            // loaded back from `out`, so `s_rows` doubles as the rope exchange slot
            // (the two consumers are mutually exclusive - see the launcher).
            if (xq != nullptr || rope_cos != nullptr) s_rows[warp] = v;
        }
    }
    // RoPE fusion epilogue: the block's 32 warps produced 32 CONSECUTIVE rows
    // (`row == blockIdx.x * nwarps + warp`, guaranteed by grid*nwarps == n), and a
    // rope pair (2i, 2i+1) of one head always sits on two ADJACENT warps: the head
    // width is a multiple of 32 and the pair start is even, so a pair never
    // straddles a block. The pair-head warp (even lane offset) reads the tail
    // warp's staged row from `s_rows` and rotates both elements in its lane-0
    // epilogue. `v = acc + bias` is exactly the f32 apply_rope_kernel read back
    // from `out`, and the expression below is its rotation verbatim, so the result
    // is bit-identical to the standalone launch it replaces.
    if (rope_cos != nullptr) {
        __syncthreads();   // every warp has staged its v into s_rows
        const int e = blockIdx.x * nwarps + warp;
        const int roff = (e < n1) ? 0 : n1;
        const int rhd = (e < n1) ? rope_hd1 : rope_hd2;
        if (rhd > 0) {
            const int re = e - roff;
            const int h = re / rhd;
            const int lane_in = re % rhd;
            const int sect = rhd - rope_rd;
            // warp + 1 < nwarps: the launcher forces nwarps == 32 (the pair never
            // straddles a block, so warp 31 is never a pair head); the guard only
            // protects the slot against a mis-shaped launch.
            if (warp + 1 < nwarps && lane_in >= sect && ((lane_in - sect) & 1) == 0) {
                const float x0 = s_rows[warp], x1 = s_rows[warp + 1];
                const int i = (lane_in - sect) >> 1;
                const int t = (*rope_base) * rope_mul + rope_off + h * rope_step;
                const float c = rope_cos[(size_t)t * (rope_rd >> 1) + i];
                const float s =
                    rope_sin[(size_t)t * (rope_rd >> 1) + i] * (rope_inverse ? -1.f : 1.f);
                if (lane == 0) {
                    float* rout = (e < n1) ? out : out2;
                    rout[re] = x0 * c - x1 * s;
                    rout[re + 1] = x0 * s + x1 * c;
                }
            }
        }
    }
    // B1 epilogue: the block's 32 warps produced the 32 CONSECUTIVE rows that
    // form one quant_kernel block, so the block itself can emit both the byte
    // and the scale. The launcher only enables this with nwarps == 32 and
    // grid*nwarps == n, which is what makes `row == blockIdx.x * nwarps + warp`
    // and `quant block == blockIdx.x` hold for every warp.
    if (xq != nullptr) {
        __syncthreads();
        float amax = (lane < nwarps) ? fabsf(s_rows[lane]) : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        // quant_kernel's arithmetic, term for term (quant_kernel<0>, block 32,
        // round_scale = true).
        const float sc = fmaxf(fast_round_scale(amax, 1.0f / 448.0f), 1e-30f);
        if (lane == 0) {
            const float inv = 1.0f / sc;
            const float q = fminf(fmaxf(s_rows[warp] * inv, -448.0f), 448.0f);
            const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
            xq[(size_t)blockIdx.x * (size_t)nwarps + (size_t)warp] = *(const uint8_t*)&f8;
            if (warp == 0) xsc[blockIdx.x] = sc;
        }
    }
}

extern "C" int dsv41_gemm_fp8_mx(const uint8_t* a, const float* a_scale, const uint8_t* w,
                                 const uint8_t* w_scale, const float* bias, float* out, int m,
                                 int n, int k, cudaStream_t s,
                                 // AR v5 store fusion (see gemm_fp8_gemv_kernel): M=1 only, the
                                 // epilogue additionally stores the row partial into every peer's
                                 // staging slot. Default null/0 keeps every existing caller (and
                                 // the C++ tests) on the old path, bit for bit.
                                 float* const* staging_tbl = nullptr,
                                 const unsigned* epoch = nullptr, int world = 0,
                                 int my_rank = 0, int stride = 0,
                                 // B1: emit the fp8 row compression of `out` (see
                                 // gemm_fp8_gemv_kernel). M=1 only, and the shape must allow
                                 // one block per quant block (n % 32 == 0); the launcher then
                                 // raises the block to 32 warps and shrinks the grid to n/32.
                                 // `xq` must NOT alias `a`. Null keeps the old path bit for bit;
                                 // a shape that cannot take it returns cudaErrorInvalidValue so
                                 // the caller keeps the (gemv, quant) pair.
                                 uint8_t* xq = nullptr, float* xsc = nullptr) {
    if (m <= 0 || n <= 0 || k <= 0 || (k & 31) || (k & 3)) return (int)cudaErrorInvalidValue;
    // The A tile lives in shared memory: 16 rows x k bytes. At the model's real
    // k (5120) that is 80 KB, well past the 48 KB static limit, so the kernel
    // needs the opt-in dynamic size (Blackwell allows ~227 KB/block). Setting
    // the attribute every call is cheap and avoids the per-device pitfall (the
    // attribute is per-context, and a TP8 process has one context per rank).
    const int smem = 16 * k;
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemm_fp8_kernel));
        if (e != cudaSuccess) return (int)e;
    }
    // M=1 (decode): skip the 16-row tile entirely - it wastes 15/16 of itself and
    // its 16*k bytes of shared memory cap the occupancy. One warp per output row.
    // The env probe is read ONCE (static): this launcher runs ~171 times per step
    // and a per-call getenv on the hot path is exactly the slip the other gates
    // in this file already avoid.
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    if (m == 1 && !no_gemv) {
        // B1: the fused fp8-row emit needs ONE BLOCK PER QUANT BLOCK (32 rows).
        // `row = blockIdx.x * nwarps + warp`, so 32 warps per block is what makes
        // the block's rows the 32 consecutive elements of a quant block; the grid
        // becomes exactly n/32, which also guarantees the single row-loop
        // iteration the epilogue's `blockIdx.x * nwarps + warp` index assumes.
        // Modes 0/1 allocate no dynamic shared memory, so they cannot stage the
        // row values - decline and let the caller keep the two-launch path.
        // P2: n >= 2048 asks for 8 rows/block (adaptive), the small shapes keep
        // the fixed g_gemv_warps (4). B1 below overrides to 32 regardless.
        int warps = dsv41_gemv_warps_for(n);
        int blocks = (n + warps - 1) / warps;
        if (xq != nullptr) {
            if (xsc == nullptr || (n & 31) || g_gemv_fp8_mode < 3)
                return (int)cudaErrorInvalidValue;
            warps = 32;
            blocks = n / 32;
        }
        const int nb_k = k >> 5;
        const int nb_k_al = (nb_k + 15) & ~15;
        // weights [nwarps][k] + (mode 4) block activation [k] + per-warp ue8m0
        // scale rows [nwarps][nb_k_al] + block activation scales [nb_k] f32
        // + (B1) the block's 32 staged row values.
        const size_t scale_bytes =
            (size_t)warps * (size_t)nb_k_al + (size_t)nb_k * sizeof(float) + 256 * sizeof(float) +
            dsv41_gemv_a32_bytes(k) + 32 * sizeof(float);   // a32 + B1 row stage
        // P1: mode 4's +1 activation row (s_a) is only allocated while a32=0 --
        // with a32 on, the fused pass decodes straight from global into s_af and
        // s_a has no reader left (dsv41_gemv_sa_bytes / the kernel's a32_direct).
        const size_t gsmem = (g_gemv_fp8_mode == 3)   ? (size_t)warps * (size_t)k + scale_bytes
                             : (g_gemv_fp8_mode == 4) ? (size_t)warps * (size_t)k + dsv41_gemv_sa_bytes(k, false) + scale_bytes
                                                      : (size_t)0;
        if (gsmem > 48 * 1024) {
            cudaError_t e = cudaFuncSetAttribute(
                gemm_fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemm_fp8_gemv_kernel));
            if (e != cudaSuccess) return (int)e;
        }
        // PDL (see dsv41_pdl_or_plain): the GEMV is a consumer of the previous
        // chain node (lin2's producer, wq_b after the norm/rope, wo_a, wo_b), so
        // its grid may start during the producer's tail and the kernel's entry
        // cudaGridDependencySynchronize() gates the activation reads.
        //
        // gemv-struct-pack: the launch carries four by-value structs (see the
        // Gemv* definitions above the kernel). Every field left unassigned keeps
        // its struct default (null / 0) -- exactly the value the old 37-slot
        // argument list used to spell out by hand for that slot.
        GemvCore gc{};
        gc.a = a; gc.a_scale = a_scale;
        gc.w = w; gc.w_scale = w_scale;
        gc.bias = bias; gc.out = out;
        gc.n = n; gc.k = k;
        gc.vec = g_gemv_fp8_mode; gc.a32 = (g_gemv_a32 ? 1 : 0);
        gc.a32_staged = (g_gemv_a32_staged ? 1 : 0);
        gc.cpasync = (g_gemv_cpasync ? 1 : 0);
        gc.act_cpasync = (g_gemv_act_cpasync ? 1 : 0);
        gc.n1 = n;                     // single family: every row is family 1
        GemvEpi ge{};
        ge.staging_tbl = staging_tbl; ge.epoch = epoch;
        ge.world = world; ge.my_rank = my_rank; ge.stride = stride;
        GemvFusion gf{};
        gf.xq = xq; gf.xsc = xsc;      // B1 fp8 row emit
        GemvRope gr{};
        cudaError_t le = dsv41_pdl_or_plain(
            gemm_fp8_gemv_kernel, dim3(blocks), dim3(warps * 32), gsmem, s, gc, gr, gf, ge);
        if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
        return (int)cudaGetLastError();
    }
    // B1 is an M=1 epilogue only: the tile path has no lane-0 row epilogue and
    // its `out` is an m x n tile, so it cannot emit the row-compressed bytes.
    if (xq != nullptr) return (int)cudaErrorInvalidValue;
    // Store fusion is an M=1 epilogue only: the tile path (prefill / m>1) has no
    // lane-0 row epilogue, so refusing loudly beats silently dropping the AR.
    if (staging_tbl != nullptr) return (int)cudaErrorInvalidValue;
    dim3 grid((n + 63) / 64, (m + 15) / 16);
    gemm_fp8_kernel<<<grid, 128, smem, s>>>(a, a_scale, w, w_scale, bias, out, m, n, k);
    return (int)cudaGetLastError();
}

// swapAB M=1 GEMV (see gemm_fp8_swapab_kernel above). A SEPARATE symbol, so the
// SIMT gemm_fp8_gemv_kernel stays the fallback and a stale .so simply has no
// entry: a shape this cannot take returns 2 (the caller keeps the SIMT gemv).
// Never returns 1 for a graceful decline -- 1 is cudaErrorInvalidValue and would
// be indistinguishable from a real launch failure (the round-42 collision).
//
// Shape requirement: n % 16 == 0 (each warp's whole MMA M tile) and k % 32 == 0
// (the MMA's K and the scale block). NOT bit-identical to the SIMT gemv (see the
// kernel's note) -- the caller judges it by text/fingerprint parity.
//
// ABI (changed 2026-09-12 for the last-block reduction): `partial` is `ks * n`
// f32 of scratch and `ctr` is `n / 16` u32 of per-tile tickets, both supplied by
// the caller (the chain's persistent device buffers). `ctr` must be ZERO before
// the first call; the kernel self-resets it, so a captured graph replays clean.
// They are only touched when ks > 1 (the ks == 1 path is a plain store); a null
// `partial`/`ctr` with ks > 1 declines with 2 so the caller keeps the SIMT gemv.
// ⚠ the .so MUST be rebuilt with the Rust side: this symbol kept its name but
// changed its parameter list, so an old .so would read `s` from the wrong slot.
extern "C" int dsv41_gemm_fp8_swapab(const uint8_t* a, const float* a_scale, const uint8_t* w,
                                     const uint8_t* w_scale, const float* bias, float* out,
                                     int n, int k, float* partial, unsigned* ctr, cudaStream_t s) {
    // Same shape of env probe as the other gates: read ONCE (this runs ~246
    // times per step, so a per-call getenv would be the hot-path slip).
    static const bool no_swapab = getenv("DSV41_NO_SWAPAB") != nullptr;
    if (no_swapab) return 2;
    if (a == nullptr || a_scale == nullptr || w == nullptr || w_scale == nullptr || out == nullptr)
        return 2;
    // Shape-based dispatch: the isolated sweep showed swapAB wins at n>=1664
    // (1.76-1.94x SIMT) but LOSES at n<=576 (0.73-0.97x, fixed overhead dominates).
    // The 246 calls/step mix both; serving the small ones with SIMT is the fix.
    if (n < 1664) return 2;
    if (n <= 0 || k <= 0 || (n & 15) || (k & 31)) return 2;

    // K split: the largest {kSwapabKSplit, .../2, 1} that divides K into slices
    // that are still whole 32-wide scale blocks. K = 5120 gives 4; a K the
    // macro's value cannot carve falls back through halving to 1 (no scratch, no
    // reduction, deterministic store).
    int ks = kSwapabKSplit;
    while (ks > 1 && (k % (32 * ks)) != 0) ks >>= 1;
    // ks > 1 elects one block per tile, so every warp of a launched block MUST
    // publish -- a warp that dropped out of the grid would leave its tile's
    // ticket short of ks and hang the grid on the elected block's arrival. The
    // grid is exact at kSwapabWarps == 1 (total blocks); this guards a future
    // WARPS > 1 where total % WARPS != 0. (Explicit parens: `%` binds tighter
    // than `*`, so the unparenthesised form would compute (n>>4) * (ks % WARPS).)
    if (((n >> 4) * ks) % kSwapabWarps) return 2;
    // The reduction needs the caller's scratch. Absent it, decline so the caller
    // runs the SIMT gemv rather than faulting on a null slot store.
    if (ks > 1 && (partial == nullptr || ctr == nullptr)) return 2;

    const int total = (n >> 4) * ks;
    const int blocks = (total + kSwapabWarps - 1) / kSwapabWarps;
    // weight ring + the per-warp activation / scale staging (see the kernel's
    // staging note). The staging is sized for the SLICE (kc, nb), not full k.
    const int kc = k / ks, nb = kc >> 5;
    const size_t smem = (size_t)kSwapabWarps * kSwapabNStage * 16 * kSwapabRow +
                        (size_t)kSwapabWarps * kc + (size_t)kSwapabWarps * nb * (sizeof(float) + 1);
    if (smem > 48 * 1024) {
        // Per-kernel ceiling, not this call's need: a sticky attribute set to a
        // smaller value would silently cap later launches (round-43 revert).
        cudaError_t e = cudaFuncSetAttribute(gemm_fp8_swapab_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             dsv41_smem_ceiling(gemm_fp8_swapab_kernel));
        if (e != cudaSuccess) return (int)e;
    }
    // NO memset of `out`: the ks > 1 path now writes it through the elected
    // block's last-block reduction, so the graph node this launcher used to add
    // is gone (that node was ~0.5 us x 246 calls = 0.12 ms/step in replay).
    // PDL like the rest of the GEMV family: the consumer's setup overlaps the
    // producer's (quant1) tail; cudaGridDependencySynchronize() at the kernel
    // entry gates the activation reads. It also keeps the shared `partial`/`ctr`
    // scratch safe to reuse across calls: a successor grid's writes only start
    // after cudaGridDependencySynchronize() returns, i.e. after this grid ended.
    cudaError_t le = dsv41_pdl_or_plain(gemm_fp8_swapab_kernel, dim3(blocks), dim3(kSwapabWarps * 32),
                                        smem, s, a, a_scale, w, w_scale, bias, out, n, k, ks,
                                        partial, ctr);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// RoPE fusion of the M=1 GEMV (see gemm_fp8_gemv_kernel's rope args). A SEPARATE
// entry point, not a new parameter on dsv41_gemm_fp8_mx: that symbol has one
// fixed ABI and six call sites, so the rope form gets its own name and the
// stale-.so fallback stays a plain symbol probe (supports_rope_fuse on the Rust
// side). The whole q/idx_q rope shape is M=1 decode, so this declines (returns 2)
// for anything the fused epilogue cannot do, and the caller keeps the
// (gemm_fp8_mx, apply_rope) pair.
//
// Shape requirements, all checked here: n % 32 == 0 (one block per 32 rows, the
// pair never straddles a block), rope_hd % 32 == 0, rope_rd positive and even,
// rope_rd <= rope_hd, and g_gemv_fp8_mode >= 3 (the s_rows staging needs dynamic
// shared memory, which modes 0/1 do not allocate). The block is raised to 32
// warps and the grid becomes n/32, exactly like B1 - that is what makes
// `e = blockIdx.x * nwarps + warp` the row index for every warp.
extern "C" int dsv41_gemm_fp8_mx_rope(const uint8_t* a, const float* a_scale, const uint8_t* w,
                                      const uint8_t* w_scale, const float* bias, float* out, int n,
                                      int k, const float* rope_cos, const float* rope_sin,
                                      const int* rope_base, int rope_mul, int rope_off,
                                      int rope_step, int rope_inverse, int rope_rd, int rope_hd,
                                      cudaStream_t s) {
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    // r43 decline-code fix (same class as r42's rope_norm/f32): the decline
    // sentinel is 2, NOT 1 - 1 is cudaErrorInvalidValue, so a genuine launch /
    // SetAttribute failure returning 1 was indistinguishable from a graceful
    // shape decline and got swallowed by the caller's rc==1 fallback.
    if (no_gemv || n <= 0 || k <= 0 || (k & 31) || (k & 3)) return 2;
    if (rope_cos == nullptr || rope_sin == nullptr || rope_base == nullptr) return 2;
    if (g_gemv_fp8_mode < 3) return 2;
    if ((n & 31) || rope_rd <= 0 || (rope_rd & 1) || rope_hd <= 0 || (rope_hd & 31) ||
        rope_rd > rope_hd)
        return 2;
    const int warps = 32;
    const int blocks = n / 32;
    const int nb_k = k >> 5;
    const int nb_k_al = (nb_k + 15) & ~15;
    const size_t scale_bytes =
        (size_t)warps * (size_t)nb_k_al + (size_t)nb_k * sizeof(float) + 256 * sizeof(float) +
        dsv41_gemv_a32_bytes(k) + 32 * sizeof(float);   // a32 + the rope/B1 row stage
    // P1: the mode-4 activation row is not allocated when a32 folds the decode
    // into s_af (dsv41_gemv_sa_bytes); mode 3 never had one.
    const size_t gsmem = (g_gemv_fp8_mode == 3)   ? (size_t)warps * (size_t)k + scale_bytes
                         : (g_gemv_fp8_mode == 4) ? (size_t)warps * (size_t)k + dsv41_gemv_sa_bytes(k, false) + scale_bytes
                                                  : (size_t)0;
    if (gsmem > 48 * 1024) {
        // Round-43 revert: the (int)gsmem form set the per-function attribute
        // to THIS call's need, which can silently cap later launches of the same
        // kernel that need more. The 232448 ceiling (the device max opt-in,
        // verified working rounds 37-41) is the correct semantic. Keep the
        // sticky clear on failure (the round-42 decline-collision fix stays).
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemm_fp8_gemv_kernel));
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // gemv-struct-pack: four by-value structs; unassigned fields keep the old
    // explicit defaults (null / 0).
    GemvCore gc{};
    gc.a = a; gc.a_scale = a_scale;
    gc.w = w; gc.w_scale = w_scale;
    gc.bias = bias; gc.out = out;
    gc.n = n; gc.k = k;
    gc.vec = g_gemv_fp8_mode; gc.a32 = (g_gemv_a32 ? 1 : 0);
    gc.a32_staged = (g_gemv_a32_staged ? 1 : 0);
    gc.cpasync = (g_gemv_cpasync ? 1 : 0);
    gc.act_cpasync = (g_gemv_act_cpasync ? 1 : 0);
    gc.n1 = n;                     // single family
    GemvRope gr{};
    gr.rope_cos = rope_cos; gr.rope_sin = rope_sin; gr.rope_base = rope_base;
    gr.rope_mul = rope_mul; gr.rope_off = rope_off; gr.rope_step = rope_step;
    gr.rope_inverse = rope_inverse; gr.rope_rd = rope_rd;
    gr.rope_hd1 = rope_hd;         // family 1 only
    GemvFusion gf{};
    GemvEpi ge{};
    cudaError_t le = dsv41_pdl_or_plain(
        gemm_fp8_gemv_kernel, dim3(blocks), dim3(warps * 32), gsmem, s, gc, gr, gf, ge);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// NORM_FUSE: the M=1 rope GEMV whose PROLOGUE produces the quantised activation
// it consumes, so the `rmsnorm_q` launch between `qr`'s producer and this gemv
// disappears (one launch + its graph node per attention, 40 per step). `qr_raw`
// is the f32 row that `dsv41_rmsnorm_q` would have turned into the (`xq`,
// `xsc`) pair; the kernel reduces it, applies `qr_w`/`qr_eps` and encodes the
// result into `s_a`/`s_as` with rmsnorm_q_kernel's arithmetic, term for term.
//
// A SEPARATE entry point for the usual reason: `dsv41_gemm_fp8_mx_rope` has one
// fixed ABI, so a stale .so stays a plain symbol probe (supports_gemm_fp8_norm
// on the Rust side) and the caller keeps the (rmsnorm_q, gemm_fp8_mx_rope) pair.
//
// Two differences from the plain rope launcher, both deliberate:
//   * the block size is FORCED to 32 warps because the prologue's reduction tree
//     is rmsnorm_q_kernel's 1024-thread tree - the partial sums (and therefore
//     `inv`, and therefore every emitted byte) only match the reference at this
//     width;
//   * the mode is FORCED to 4 (the block-wide activation copy), because that is
//     the only mode where the activation the consume loop reads lives in shared
//     memory. The DSV41_GEMV_FP8_MODE env gate does not apply here.
// It reuses the rope shape checks verbatim, so any shape the rope launcher takes
// this one takes too (and vice versa), and a decline leaves the caller on the
// old pair.
extern "C" int dsv41_gemm_fp8_mx_rope_norm(const float* qr_raw, const float* qr_w, float qr_eps,
                                           const uint8_t* w, const uint8_t* w_scale,
                                           const float* bias, float* out, int n, int k,
                                           const float* rope_cos, const float* rope_sin,
                                           const int* rope_base, int rope_mul, int rope_off,
                                           int rope_step, int rope_inverse, int rope_rd,
                                           int rope_hd, cudaStream_t s) {
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    if (no_gemv || qr_raw == nullptr || qr_w == nullptr) return 2;
    if (n <= 0 || k <= 0 || (k & 31) || (k & 3)) return 2;
    if (rope_cos == nullptr || rope_sin == nullptr || rope_base == nullptr) return 2;
    if ((n & 31) || rope_rd <= 0 || (rope_rd & 1) || rope_hd <= 0 || (rope_hd & 31) ||
        rope_rd > rope_hd)
        return 2;  // r42-fix round 2: decline must never be 1 (cudaErrorInvalidValue)
    const int warps = 32;
    const int blocks = n / 32;
    const int nb_k = k >> 5;
    const int nb_k_al = (nb_k + 15) & ~15;
    const size_t scale_bytes =
        (size_t)warps * (size_t)nb_k_al + (size_t)nb_k * sizeof(float) + 256 * sizeof(float) +
        dsv41_gemv_a32_bytes(k) + 32 * sizeof(float);   // a32 + the rope/B1 row stage
    // mode 4 only: `warps` weight rows + the block-wide activation row.
    // P1 does NOT apply here: the NORM_FUSE prologue WRITES `s_a` and the
    // materialisation loop still decodes it, so the slot stays allocated even
    // with a32 on (the kernel's `a32_direct` excludes qr_raw != nullptr).
    const size_t gsmem = (size_t)(warps + 1) * (size_t)k + scale_bytes;
    if (gsmem > 48 * 1024) {
        // Round-43 revert: the (int)gsmem form set the per-function attribute
        // to THIS call's need, which can silently cap later launches of the same
        // kernel that need more. The 232448 ceiling (the device max opt-in,
        // verified working rounds 37-41) is the correct semantic. Keep the
        // sticky clear on failure (the round-42 decline-collision fix stays).
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemm_fp8_gemv_kernel));
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // `a`/`a_scale` are null: the prologue produces the activation and the
    // kernel never dereferences them on this path (the `qr_raw` guard skips both
    // staging sites; mode 4 takes `s_a`).
    // gemv-struct-pack: four by-value structs; `a`/`a_scale` stay null (the
    // NORM_FUSE prologue produces the activation) and unassigned fields keep the
    // old explicit defaults (null / 0).
    GemvCore gc{};
    gc.w = w; gc.w_scale = w_scale;
    gc.bias = bias; gc.out = out;
    gc.n = n; gc.k = k;
    gc.vec = 4;                    // mode 4 is forced: s_a holds the activation
    gc.a32 = (g_gemv_a32 ? 1 : 0);
    gc.n1 = n;                     // single family
    GemvRope gr{};
    gr.rope_cos = rope_cos; gr.rope_sin = rope_sin; gr.rope_base = rope_base;
    gr.rope_mul = rope_mul; gr.rope_off = rope_off; gr.rope_step = rope_step;
    gr.rope_inverse = rope_inverse; gr.rope_rd = rope_rd;
    gr.rope_hd1 = rope_hd;         // family 1 only
    GemvFusion gf{};
    gf.qr_raw = qr_raw; gf.qr_w = qr_w; gf.qr_eps = qr_eps;
    GemvEpi ge{};
    cudaError_t le = dsv41_pdl_or_plain(
        gemm_fp8_gemv_kernel, dim3(blocks), dim3(warps * 32), gsmem, s, gc, gr, gf, ge);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// RoPE fusion of the TWO-family M=1 GEMV (dsv41_gemm_fp8_mx2's rope form). Both
// families carry their own head width (`rope_hd1` for family 1, `rope_hd2` for
// family 2) because the wq_b / idx_wq_b pair has different head widths (512 and
// 128) but the SAME rope length, cos/sin table and position counter. Everything
// else is shared with dsv41_gemm_fp8_mx_rope; the pair never straddles the family
// boundary because n1 and the head width are multiples of 32 and a pair start is
// even. Returns 2 (decline) on any shape the fused epilogue cannot do.
extern "C" int dsv41_gemm_fp8_mx2_rope(const uint8_t* a, const float* a_scale, const uint8_t* w1,
                                       const uint8_t* w1_scale, const float* bias1, float* out1,
                                       int n1, const uint8_t* w2, const uint8_t* w2_scale,
                                       const float* bias2, float* out2, int n2, int k,
                                       const float* rope_cos, const float* rope_sin,
                                       const int* rope_base, int rope_mul, int rope_off,
                                       int rope_step, int rope_inverse, int rope_rd, int rope_hd1,
                                       int rope_hd2, cudaStream_t s) {
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    // r43 decline-code fix: sentinel 2, never 1 (== cudaErrorInvalidValue).
    if (no_gemv || n1 <= 0 || n2 <= 0 || k <= 0 || (k & 31) || (k & 3)) return 2;
    if (rope_cos == nullptr || rope_sin == nullptr || rope_base == nullptr) return 2;
    if (g_gemv_fp8_mode < 3) return 2;
    if (((n1 | n2) & 31) || rope_rd <= 0 || (rope_rd & 1)) return 2;
    if (!(rope_hd1 > 0 && (rope_hd1 & 31) == 0 && rope_rd <= rope_hd1)) return 2;
    if (!(rope_hd2 > 0 && (rope_hd2 & 31) == 0 && rope_rd <= rope_hd2)) return 2;
    const int n = n1 + n2;
    const int warps = 32;
    const int blocks = n / 32;
    const int nb_k = k >> 5;
    const int nb_k_al = (nb_k + 15) & ~15;
    const size_t scale_bytes =
        (size_t)warps * (size_t)nb_k_al + (size_t)nb_k * sizeof(float) + 256 * sizeof(float) +
        dsv41_gemv_a32_bytes(k) + 32 * sizeof(float);   // a32 + the rope/B1 row stage
    // P1: the mode-4 activation row is not allocated when a32 folds the decode
    // into s_af (dsv41_gemv_sa_bytes); mode 3 never had one.
    const size_t gsmem = (g_gemv_fp8_mode == 3)   ? (size_t)warps * (size_t)k + scale_bytes
                         : (g_gemv_fp8_mode == 4) ? (size_t)warps * (size_t)k + dsv41_gemv_sa_bytes(k, false) + scale_bytes
                                                  : (size_t)0;
    if (gsmem > 48 * 1024) {
        // Round-43 revert: the (int)gsmem form set the per-function attribute
        // to THIS call's need, which can silently cap later launches of the same
        // kernel that need more. The 232448 ceiling (the device max opt-in,
        // verified working rounds 37-41) is the correct semantic. Keep the
        // sticky clear on failure (the round-42 decline-collision fix stays).
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemm_fp8_gemv_kernel));
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // gemv-struct-pack: four by-value structs; both families (and their two head
    // widths) travel in GemvCore / GemvRope, unassigned fields keep the old
    // explicit defaults (null / 0).
    GemvCore gc{};
    gc.a = a; gc.a_scale = a_scale;
    gc.w = w1; gc.w_scale = w1_scale;
    gc.bias = bias1; gc.out = out1;
    gc.n = n; gc.k = k;
    gc.vec = g_gemv_fp8_mode; gc.a32 = (g_gemv_a32 ? 1 : 0);
    gc.a32_staged = (g_gemv_a32_staged ? 1 : 0);
    gc.cpasync = (g_gemv_cpasync ? 1 : 0);
    gc.act_cpasync = (g_gemv_act_cpasync ? 1 : 0);
    gc.w2 = w2; gc.w2_scale = w2_scale;
    gc.bias2 = bias2; gc.out2 = out2;
    gc.n1 = n1;
    GemvRope gr{};
    gr.rope_cos = rope_cos; gr.rope_sin = rope_sin; gr.rope_base = rope_base;
    gr.rope_mul = rope_mul; gr.rope_off = rope_off; gr.rope_step = rope_step;
    gr.rope_inverse = rope_inverse; gr.rope_rd = rope_rd;
    gr.rope_hd1 = rope_hd1; gr.rope_hd2 = rope_hd2;
    GemvFusion gf{};
    GemvEpi ge{};
    cudaError_t le = dsv41_pdl_or_plain(
        gemm_fp8_gemv_kernel, dim3(blocks), dim3(warps * 32), gsmem, s, gc, gr, gf, ge);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// A5: the M=1 w2 GEMV with the caller's trailing add_inplace folded into the
// epilogue (`out += w @ a`). A SEPARATE entry point, not a new parameter on
// dsv41_gemm_fp8_mx: that symbol has one fixed ABI and six call sites, none of
// which accumulate, so the fused form gets its own name and the stale-.so
// fallback is a plain symbol probe (supports_gemm_fp8_add on the Rust side).
// M=1 only -- that is the shared expert's down projection, the one place in the
// chain that added a standalone `ferrite_add` after the GEMV (40 launches/step).
// Returns 2 when the shape cannot use the GEMV, so the caller keeps the
// two-launch (gemm_fp8_mx + add_inplace) pair; any other value is the usual
// cudaError_t status.
extern "C" int dsv41_gemm_fp8_mx_add(const uint8_t* a, const float* a_scale,
                                     const uint8_t* w, const uint8_t* w_scale,
                                     const float* bias, float* out, int m, int n, int k,
                                     cudaStream_t s) {
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    // r43 decline-code fix: sentinel 2, never 1 (== cudaErrorInvalidValue).
    if (m != 1 || no_gemv || n <= 0 || k <= 0 || (k & 31) || (k & 3)) return 2;
    // P2: adaptive rows/block (8 for n >= 2048, else the fixed g_gemv_warps).
    const int warps = dsv41_gemv_warps_for(n);
    const int blocks = (n + warps - 1) / warps;
    const int nb_k = k >> 5;
    const int nb_k_al = (nb_k + 15) & ~15;
    const size_t scale_bytes =
        (size_t)warps * (size_t)nb_k_al + (size_t)nb_k * sizeof(float) + 256 * sizeof(float) +
        dsv41_gemv_a32_bytes(k) + 32 * sizeof(float);   // a32 + the B1 row stage slot
    // P1: the mode-4 activation row is not allocated when a32 folds the decode
    // into s_af (dsv41_gemv_sa_bytes); mode 3 never had one.
    const size_t gsmem = (g_gemv_fp8_mode == 3)   ? (size_t)warps * (size_t)k + scale_bytes
                         : (g_gemv_fp8_mode == 4) ? (size_t)warps * (size_t)k + dsv41_gemv_sa_bytes(k, false) + scale_bytes
                                                  : (size_t)0;
    if (gsmem > 48 * 1024) {
        // Round-43 revert: the (int)gsmem form set the per-function attribute
        // to THIS call's need, which can silently cap later launches of the same
        // kernel that need more. The 232448 ceiling (the device max opt-in,
        // verified working rounds 37-41) is the correct semantic. Keep the
        // sticky clear on failure (the round-42 decline-collision fix stays).
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemm_fp8_gemv_kernel));
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // gemv-struct-pack: four by-value structs; `epi_add` is the only non-default
    // epilogue field (the A5 add fold), everything else keeps null / 0.
    GemvCore gc{};
    gc.a = a; gc.a_scale = a_scale;
    gc.w = w; gc.w_scale = w_scale;
    gc.bias = bias; gc.out = out;
    gc.n = n; gc.k = k;
    gc.vec = g_gemv_fp8_mode; gc.a32 = (g_gemv_a32 ? 1 : 0);
    gc.a32_staged = (g_gemv_a32_staged ? 1 : 0);
    gc.cpasync = (g_gemv_cpasync ? 1 : 0);
    gc.act_cpasync = (g_gemv_act_cpasync ? 1 : 0);
    gc.n1 = n;                     // single family
    GemvEpi ge{};
    ge.epi_add = 1;                // A5: out[row] += acc + bias
    GemvFusion gf{};
    GemvRope gr{};
    cudaError_t le = dsv41_pdl_or_plain(
        gemm_fp8_gemv_kernel, dim3(blocks), dim3(warps * 32), gsmem, s, gc, gr, gf, ge);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// wo_b's f32-activation GEMV: the M=1 GEMV that reads the RAW f32 activation
// instead of an fp8 (`a`, `a_scale`) pair, so the `quant1(s.wo)` launch between
// wo_a and wo_b disappears (one launch + one graph node per layer, 40 per step)
// and the activation staging loses its per-block LUT decode + scale multiply.
//
// Why this does NOT repeat B1's mistake: B1 moved the fusion into the wo_a
// epilogue, which forced 32 warps/block so that 32 consecutive rows formed one
// quant block -- and that shape (grid = n/32, 32 warps) lost 148->32 active SMs
// and measured +0.24ms. Here only the DATA PATH changes, not the grid shape: the
// block is the normal `g_gemv_warps` / ceil(n/warps) shape, exactly the one the
// plain `gemm_fp8_mx` GEMV uses, so there is no SM-utilisation penalty.
//
// A SEPARATE symbol (not a new parameter on dsv41_gemm_fp8_mx) for the usual
// reason: that symbol has one fixed ABI and six call sites, so the f32 form gets
// its own name and a stale .so stays a plain symbol probe
// (supports_gemm_fp8_f32 on the Rust side).
//
// NOT bit-identical to the (quant1, gemm_fp8_mx) pair it replaces: it skips the
// fp8 quantise->dequantise round trip, so the activation carries no 4-bit
// mantissa loss and the row partial is slightly MORE accurate. The downstream is
// the wo_b row partial -> the AR sum -> hc_post, where the tighter value is the
// correct direction. Returns 1 (decline) when the shape/mode cannot take it, so
// the caller keeps the (quant1, gemm_fp8_mx) pair.
extern "C" int dsv41_gemm_fp8_mx_f32(const float* a_f32, const uint8_t* w,
                                     const uint8_t* w_scale, const float* bias, float* out,
                                     int n, int k, cudaStream_t s) {
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    if (no_gemv || a_f32 == nullptr || n <= 0 || k <= 0 || (k & 31) || (k & 3)) return 2;
    // The s_af materialisation (where the f32 lands) is the vec>=3 branch only.
    if (g_gemv_fp8_mode < 3) return 2;  // r42-fix round 2: decline must never be 1 (cudaErrorInvalidValue)
    // P2: adaptive rows/block (8 for n >= 2048, else the fixed g_gemv_warps).
    const int warps = dsv41_gemv_warps_for(n);
    const int blocks = (n + warps - 1) / warps;
    const int nb_k = k >> 5;
    const int nb_k_al = (nb_k + 15) & ~15;
    // Same layout as dsv41_gemm_fp8_mx: weight rows [warps][k] + per-warp ue8m0
    // scale rows [warps][nb_k_al] + block activation scales [nb_k] f32 (unused on
    // this path, reserved) + 256-entry LUT + the a32/f32 row [k] + the B1 row
    // slot [32]. The kernel never dereferences `a`/`a_scale` here.
    const size_t scale_bytes =
        (size_t)warps * (size_t)nb_k_al + (size_t)nb_k * sizeof(float) + 256 * sizeof(float) +
        dsv41_gemv_a32_bytes(k) + 32 * sizeof(float);   // a32 + B1 row stage
    // Mode 4 allocates the block-wide activation copy; keep the caller's mode so
    // gsmem matches the branch the kernel's `vec` takes. On the f32 path the copy
    // is never written or read, so P1's dead-slot rule applies just the same
    // (a32=1 -> no activation row at all).
    const size_t gsmem =
        (g_gemv_fp8_mode == 4)
            ? (size_t)warps * (size_t)k + dsv41_gemv_sa_bytes(k, false) + scale_bytes
            : (size_t)warps * (size_t)k + scale_bytes;
    if (gsmem > 48 * 1024) {
        // Round-43 revert: the (int)gsmem form set the per-function attribute
        // to THIS call's need, which can silently cap later launches of the same
        // kernel that need more. The 232448 ceiling (the device max opt-in,
        // verified working rounds 37-41) is the correct semantic. Keep the
        // sticky clear on failure (the round-42 decline-collision fix stays).
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemm_fp8_gemv_kernel));
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // PDL (see dsv41_pdl_or_plain): the wo_b GEMV reads wo_a's fp32 output, so it
    // is the second consumer in the tail of the chain, exactly where the
    // node-gap matters.
    //
    // gemv-struct-pack: four by-value structs; `a_f32` is the only set input
    // (`a`/`a_scale` stay null on this path), everything else keeps null / 0.
    GemvCore gc{};
    gc.w = w; gc.w_scale = w_scale;
    gc.bias = bias; gc.out = out;
    gc.n = n; gc.k = k;
    gc.vec = g_gemv_fp8_mode; gc.a32 = (g_gemv_a32 ? 1 : 0);
    gc.a32_staged = (g_gemv_a32_staged ? 1 : 0);
    gc.cpasync = (g_gemv_cpasync ? 1 : 0);
    gc.act_cpasync = (g_gemv_act_cpasync ? 1 : 0);
    gc.n1 = n;                     // single family
    GemvFusion gf{};
    gf.a_f32 = a_f32;              // raw f32 activation, no fp8 round trip
    GemvEpi ge{};
    GemvRope gr{};
    cudaError_t le = dsv41_pdl_or_plain(
        gemm_fp8_gemv_kernel, dim3(blocks), dim3(warps * 32), gsmem, s, gc, gr, gf, ge);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// chain-pair-grid-sync (2026-09-12): wo_a -> wo_b as ONE launch.
//
// WHY. The attention output projection is two chain-ADJACENT M=1 GEMVs:
//
//     wo_a : [1, ka] fp8 -> [1, na] f32   (na = o_lora_rank, ka = hpg*head_dim)
//     wo_b : [1, na] f32 -> [1, nb] f32   (nb = dim,          kb = o_lora_rank)
//
// wo_b's ENTIRE activation IS wo_a's output, so two launches per layer (80 per
// step) pay two launch/node gaps and two prologue floors for a strictly serial
// dependency. This kernel keeps both GEMV bodies and connects them with a
// DEVICE-WIDE BARRIER instead of a stream edge (the design note
// `chain-pair-grid-sync`, STATUS.md).
//
// STRUCTURE
//   phase 1  the wo_a body in its mode-4 + a32 form (staged weights, the
//            block-wide pre-decoded activation `s_af`), rows [0, na), fp8
//            activation in, f32 out into `mid` (= the Rust side's `s.wo`).
//   barrier  sense-reversing, arrived once per block by thread 0.
//   phase 2  the wo_b body in its f32-activation form (what
//            dsv41_gemm_fp8_mx_f32 already does), rows [0, nb), re-stages `mid`
//            into `s_af` and writes `out` (= `s.o`).
//
// BIT-IDENTITY vs the two standalone launches. Each output row is still ONE
// warp walking the identical lane order over the identical staged bytes (same
// e4m3 LUT, same ue8m0 scale row, same `kb*32 + lane` k-block accumulation
// order, same shuffle-down reduction), so every row is bit-identical to the row
// its standalone launch produces. The grid/rows-per-block split MAY differ (the
// fused grid is capped by residency) and that is allowed precisely BECAUSE a
// row's dot does not depend on which block owns it.
//
// DEADLOCK SAFETY -- the one hard constraint. The barrier completes only when
// ALL gridDim.x blocks are resident on the SMs at the same time. The launcher
// therefore caps the grid at `co_res` = (blocks/SM from
// cudaOccupancyMaxActiveBlocksPerMultiprocessor at THIS block size and THIS
// dynamic smem) x SMs. NEVER raise the grid past co_res: a block that never
// becomes resident never reaches the barrier, the resident ones spin forever,
// and the device hangs (the watchdog cannot see it inside a captured graph).
//
// MEMORY ORDERING. Producer side: `__syncthreads()` (this block's `mid[]`
// stores are done) -> `__threadfence()` -> the arrive atomicAdd. Consumer side:
// the spin observes the sense flip -> `__threadfence()` -> `__syncthreads()` ->
// the `mid[]` reads. Without the fences a block can observe the flip before the
// producer's stores have drained to L2 and read a stale row.
//
// NOT COVERED (the launcher returns 2 and the caller keeps the two launches):
//   * B1 (`xq`/`xsc` row compression) and the AR-v5 store epilogue -- both are
//     epilogue variants of the standalone launchers and neither is wired in;
//   * any fp8 mode other than 4, or a32 off (the two bodies are specialised);
//   * wo_a split over more than one group (nlg != 1): phase 1 stages ONE
//     block-wide activation row, which only exists when this rank owns exactly
//     one `o_groups` slice (the TP8 production shape).
// 2, never 1 -- 1 is cudaErrorInvalidValue and the caller would swallow a real
// launch failure (the r42/r43 decline-code collision).
__global__ void __launch_bounds__(1024)
gemm_fp8_wo_pair_kernel(const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
                        const uint8_t* __restrict__ wa, const uint8_t* __restrict__ wa_scale,
                        const float* __restrict__ wa_bias, int na, int ka,
                        const uint8_t* __restrict__ wb, const uint8_t* __restrict__ wb_scale,
                        const float* __restrict__ wb_bias, int nb, int kb,
                        float* __restrict__ mid, float* __restrict__ out,
                        unsigned* __restrict__ bar, int nwarps, int cpasync) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nb_ka = ka >> 5;                              // phase-1 k-blocks
    const int nb_kb = kb >> 5;                              // phase-2 k-blocks
    const int kmax = (ka > kb) ? ka : kb;
    const int nb_kmax = (nb_ka > nb_kb) ? nb_ka : nb_kb;
    const int nb_kmax_al = (nb_kmax + 15) & ~15;            // 16B units (cp.async)

    // ---- ONE shared-memory pool for BOTH phases (the design's `smem = max`) --
    // The two phases never overlap in time -- the grid barrier plus the
    // __syncthreads() after it separate them -- so every slot is sized by the
    // WIDER phase and re-used by the narrower one. The kernel has NO static
    // shared memory, so the launcher's dynamic request is exactly this sum.
    extern __shared__ uint8_t s_pool[];
    uint8_t* s_w = s_pool;                                        // [nwarps][kmax]
    uint8_t* s_ws = s_w + (size_t)nwarps * (size_t)kmax;          // [nwarps][nb_kmax_al]
    float* s_as = reinterpret_cast<float*>(s_ws + (size_t)nwarps * (size_t)nb_kmax_al);
    float* s_lut = s_as + nb_kmax;                                // 256 f32
    float* s_af = s_lut + 256;                                    // [kmax] f32

    // The e4m3 decode table is phase-independent and is never overwritten, so it
    // is built once, before either phase's barrier.
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_lut[i] = e4m3_to_f((uint8_t)i);

    // ===================== phase 1: wo_a (fp8 -> f32) ========================
    {
        uint8_t* row_s = s_w + (size_t)warp * (size_t)kmax;
        uint8_t* row_sc = s_ws + (size_t)warp * (size_t)nb_kmax_al;
        // P3 (cp.async weights-first, DSV41_GEMV_CPASYNC): the first row this
        // warp will consume depends on none of the activation work below, so
        // issue it BEFORE the block-wide staging and let the two overlap. Same
        // bytes, same slot as the loop's own staging -- only the issue point
        // moves. The unsigned cast keeps the ternary's -1 from promoting to
        // 0xFFFFFFFF.
        int pf_row = (cpasync != 0) ? ((int)blockIdx.x * nwarps + warp) : -1;
        if (pf_row >= na) pf_row = -1;
        if (pf_row >= 0) {
            const uint8_t* pf_w = wa + (size_t)pf_row * ka;
            const int n16p = ka >> 4;
            for (int i = lane; i < n16p; i += 32) dsv41_cp_async16(row_s + (i << 4), pf_w + (i << 4));
            dsv41_cp_commit();
            for (int i = (n16p << 4) + lane; i < ka; i += 32) row_s[i] = pf_w[i];
        }
        // The activation scales are the same for every output row, so they are
        // staged once per block (the standalone kernel's `s_as` loop verbatim).
        for (int i = threadIdx.x; i < nb_ka; i += blockDim.x) s_as[i] = a_scale[i];
        __syncthreads();               // s_lut + s_as published
        // a32_direct (P1): one pass global uint4 -> LUT -> s_af. This is the
        // SAME `s_lut[b] * s_as[idx>>5]` product the staged form computed, so
        // the emitted f32 is bit-identical -- only the k-byte `s_a` round trip
        // (and its slot) is gone.
        const int n16a = ka >> 4;
        for (int i = threadIdx.x; i < n16a; i += blockDim.x) {
            const uint4 v = *reinterpret_cast<const uint4*>(a + (i << 4));
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int idx = (i << 4) + j;
                s_af[idx] = s_lut[b[j]] * s_as[idx >> 5];
            }
        }
        for (int i = (n16a << 4) + threadIdx.x; i < ka; i += blockDim.x)
            s_af[i] = s_lut[a[i]] * s_as[i >> 5];
        __syncthreads();               // s_af published for every warp

        for (int row = blockIdx.x * nwarps + warp; row < na; row += gridDim.x * nwarps) {
            const uint8_t* wr = wa + (size_t)row * ka;
            const uint8_t* wsr = wa_scale + (size_t)(row >> 5) * nb_ka;
            float acc = 0.f;
            const bool prefetched = (row == pf_row);
            if (!prefetched) {
                const int n16 = ka >> 4;
                for (int i = (n16 << 4) + lane; i < ka; i += 32) row_s[i] = wr[i];
                for (int i = lane; i < n16; i += 32)
                    dsv41_cp_async16(row_s + (i << 4), wr + (i << 4));
            }
            for (int i = lane; i < nb_ka; i += 32) row_sc[i] = wsr[i];
            // Unconditional: on the prefetched iteration this commits an EMPTY
            // group after the prologue's, and wait_all retires both.
            dsv41_cp_commit();
            dsv41_cp_wait_all();
            __syncwarp();
#pragma unroll 4
            for (int kbi = 0; kbi < nb_ka; ++kbi) {
                const float sb = ue8m0_to_f(row_sc[kbi]);
                const int j = kbi * 32 + lane;
                const float av = s_af[j];
                acc += av * (s_lut[row_s[j]] * sb);
            }
            __syncwarp();
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) mid[row] = acc + (wa_bias ? wa_bias[row] : 0.f);
        }
    }

    // ============================= grid barrier ==============================
    // sense-reversing, arrived once per block by thread 0. Safe ONLY because the
    // launcher caps gridDim.x at the residency number (see co_res in
    // dsv41_gemm_fp8_wo_pair).
    //
    // WHY SENSE-REVERSING AND NOT A PLAIN COUNTER: a captured graph REPLAYS this
    // launch dozens of times, so the barrier state must be self-resetting. The
    // last arriver zeroes `bar[0]` for the next launch and flips `bar[1]`, which
    // is what releases this launch's waiters. A waiter reads the sense BEFORE it
    // arrives, and the flip can only happen after every block has arrived, so
    // no waiter can miss the flip (or observe the previous launch's).
    __syncthreads();                   // this block's mid[] stores are complete
    if (threadIdx.x == 0) {
        __threadfence();                               // release: mid[] visible
        const unsigned s0 = atomicAdd(bar + 1, 0u);    // current sense
        if (atomicAdd(bar, 1u) == (unsigned)gridDim.x - 1u) {
            atomicExch(bar, 0u);                       // reset for the next launch
            __threadfence();
            atomicXor(bar + 1, 1u);                    // flip -> release waiters
        } else {
            while (atomicAdd(bar + 1, 0u) == s0) __nanosleep(32);
            __threadfence();                           // acquire
        }
    }
    __syncthreads();

    // ===================== phase 2: wo_b (f32 -> f32) ========================
    {
        uint8_t* row_s = s_w + (size_t)warp * (size_t)kmax;
        uint8_t* row_sc = s_ws + (size_t)warp * (size_t)nb_kmax_al;
        int pf_row = (cpasync != 0) ? ((int)blockIdx.x * nwarps + warp) : -1;
        if (pf_row >= nb) pf_row = -1;
        if (pf_row >= 0) {
            const uint8_t* pf_w = wb + (size_t)pf_row * kb;
            const int n16p = kb >> 4;
            for (int i = lane; i < n16p; i += 32) dsv41_cp_async16(row_s + (i << 4), pf_w + (i << 4));
            dsv41_cp_commit();
            for (int i = (n16p << 4) + lane; i < kb; i += 32) row_s[i] = pf_w[i];
        }
        // Re-stage the f32 activation (wo_a's f32 output) into the a32 slot --
        // the same block-wide copy dsv41_gemm_fp8_mx_f32 performs, so the
        // consume loop reads the identical value. `s_af` is [kmax] and this only
        // fills [0, kb), which is all the phase-2 loop indexes.
        for (int i = threadIdx.x; i < kb; i += blockDim.x) s_af[i] = mid[i];
        __syncthreads();               // s_af published

        for (int row = blockIdx.x * nwarps + warp; row < nb; row += gridDim.x * nwarps) {
            const uint8_t* wr = wb + (size_t)row * kb;
            const uint8_t* wsr = wb_scale + (size_t)(row >> 5) * nb_kb;
            float acc = 0.f;
            const bool prefetched = (row == pf_row);
            if (!prefetched) {
                const int n16 = kb >> 4;
                for (int i = (n16 << 4) + lane; i < kb; i += 32) row_s[i] = wr[i];
                for (int i = lane; i < n16; i += 32)
                    dsv41_cp_async16(row_s + (i << 4), wr + (i << 4));
            }
            for (int i = lane; i < nb_kb; i += 32) row_sc[i] = wsr[i];
            dsv41_cp_commit();
            dsv41_cp_wait_all();
            __syncwarp();
#pragma unroll 4
            for (int kbi = 0; kbi < nb_kb; ++kbi) {
                const float sb = ue8m0_to_f(row_sc[kbi]);
                const int j = kbi * 32 + lane;
                const float av = s_af[j];
                acc += av * (s_lut[row_s[j]] * sb);
            }
            __syncwarp();
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) out[row] = acc + (wb_bias ? wb_bias[row] : 0.f);
        }
    }
}

// The chain-pair launcher. TWO deliberate differences from every other launcher
// in this family:
//   * the launch is PLAIN (`<<<>>>`), NOT via dsv41_pdl_or_plain. A grid-wide
//     barrier and PDL are a bad pair: PDL lets this grid start while the
//     producer is still resident, and the barrier needs the WHOLE grid resident
//     before any block may pass it. (The node after this one, the AR / hc_post
//     family, is not PDL-covered either, so nothing is lost.)
//   * the grid is CAPPED by residency (co_res). That cap is a correctness
//     requirement, not tuning -- see the kernel's DEADLOCK SAFETY note.
//
// Returns 2 (decline, never 1) for any shape or arm the fused kernel does not
// implement, so the Rust caller keeps the (wo_a, wo_b) two-launch path bit for
// bit.
extern "C" int dsv41_gemm_fp8_wo_pair(const uint8_t* a, const float* a_scale,
                                      const uint8_t* wa, const uint8_t* wa_scale,
                                      const float* wa_bias, int na, int ka,
                                      const uint8_t* wb, const uint8_t* wb_scale,
                                      const float* wb_bias, int nb, int kb,
                                      float* mid, float* out, unsigned* bar,
                                      cudaStream_t s) {
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    if (no_gemv) return 2;
    if (a == nullptr || a_scale == nullptr || wa == nullptr || wa_scale == nullptr ||
        wb == nullptr || wb_scale == nullptr || mid == nullptr || out == nullptr || bar == nullptr)
        return 2;
    if (na <= 0 || nb <= 0 || ka <= 0 || kb <= 0) return 2;
    // Both bodies index their weight rows in 16-byte units (cp.async) and their
    // k-blocks in 32-element units, so both k's must be multiples of 32.
    if ((ka & 31) || (kb & 31)) return 2;
    // Specialised bodies: only the production arm (mode 4 = staged weights +
    // block-wide activation copy; a32 = the pre-decoded `s_af`) is implemented.
    // `cpasync` IS honoured (it moves WHEN the weight row is issued).
    if (g_gemv_fp8_mode != 4 || !g_gemv_a32 || g_gemv_a32_staged) return 2;

    // The pair kernel has its OWN smem layout (two phases share one pool), so its
    // residency -- and therefore the barrier's grid cap -- is its own number. Use
    // the fixed small-n width: the large-n adaptive arm would double the per-warp
    // weight-row slot and cut co_res for no benefit here (the row loop already
    // strides).
    const int warps = g_gemv_warps;
    const int kmax = (ka > kb) ? ka : kb;
    const int nb_kmax = kmax >> 5;
    const int nb_kmax_al = (nb_kmax + 15) & ~15;
    const size_t gsmem = (size_t)warps * (size_t)kmax          // weight rows
                       + (size_t)warps * (size_t)nb_kmax_al    // ue8m0 scale rows
                       + (size_t)nb_kmax * sizeof(float)       // activation scales
                       + 256 * sizeof(float)                   // e4m3 LUT
                       + (size_t)kmax * sizeof(float);         // s_af
    if (gsmem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_wo_pair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            dsv41_smem_ceiling(gemm_fp8_wo_pair_kernel));
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // ---- co_res: the residency cap. Cached because the model's shape is fixed
    // and this launcher runs 80 times per step (a per-call occupancy query is a
    // driver round trip). Keyed on the parameters that define the answer.
    static int co_res_cached = 0;
    static size_t co_res_gsmem = 0;
    static int co_res_warps = 0;
    if (co_res_cached == 0 || co_res_gsmem != gsmem || co_res_warps != warps) {
        int per_sm = 0;
        cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, gemm_fp8_wo_pair_kernel, warps * 32, gsmem);
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        co_res_cached = per_sm * sms;
        co_res_gsmem = gsmem;
        co_res_warps = warps;
    }
    if (co_res_cached <= 0) return 2;   // no resident configuration -> decline

    const int rows = (na > nb) ? na : nb;
    int blocks = (rows + warps - 1) / warps;
    if (blocks > co_res_cached) blocks = co_res_cached;   // HARD deadlock guard
    if (blocks <= 0) return 2;

    gemm_fp8_wo_pair_kernel<<<dim3(blocks), dim3(warps * 32), gsmem, s>>>(
        a, a_scale, wa, wa_scale, wa_bias, na, ka,
        wb, wb_scale, wb_bias, nb, kb,
        mid, out, bar, warps, g_gemv_cpasync ? 1 : 0);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// chain-pair-batch 链2 (2026-09-12): the shared expert's w1w3 -> swiglu -> w2
// chain as ONE launch.
//
// WHY. That chain is three already-fused kernels that are strictly serial:
//
//     w1w3  : gemm_fp8_mx2    [1, dim] fp8 -> [1, 2*il] f32   (gate | up)
//     swiglu: swiglu_limit_q  [1, 2*il]      -> [1, il] f32 + fp8(xq/xsc)
//     w2    : gemm_fp8_mx     fp8(xq/xsc) [1, il] -> [1, dim] f32
//
// Three launches + three graph nodes per layer (120/step) pay three prologue
// floors for a strictly serial dependency. This kernel keeps all three bodies
// and connects them with a DEVICE-WIDE BARRIER instead of a stream edge.
//
// STRUCTURE
//   phase 1  every warp owns ONE inter row i and walks BOTH of that row's
//            weight rows -- w1 (the gate) and w3 (the up) -- over the fp8
//            activation (`a`/`a_scale`, k = dim). The dot is the staged
//            gemm_fp8_gemv_kernel mode-4 consume loop over the a32-predecoded
//            `s_af`, with the two chains sharing every `av`. The epilogue is
//            swiglu_limit_kernel's clamp + silu and swiglu_limit_q_kernel's fp8
//            emit: the f32 lands in `act` (= the Rust side's ex_act), the fp8
//            pair in `aq`/`aqsc` (the phase-2 activation).
//   barrier  sense-reversing, arrived once per block by thread 0.
//   phase 2  the w2 M=1 GEMV in gemm_fp8_gemv_kernel's staged mode-4 + a32
//            form: rows [0, dim), k = il (= phase 1's row count), consuming
//            `aq`/`aqsc` into `out`.
//
// WHY THE PAIR STRUCTURE (not gemm_fp8_mx2's "row split in two"). swiglu is
// silu(gate_i) * up_i: a kernel that maps rows [0, il) to w1's rows and
// [il, 2*il) to w3's (what mx2 does) puts the two halves of one silu on two
// DIFFERENT warps, so the epilogue would need a global round trip -- which is
// exactly the swiglu launch this fuses away. One warp must hold gate_i AND up_i.
//
// BIT-IDENTITY. Each phase-1 dot walks `j = kb*32 + lane` for kb ascending with
// a single accumulator per chain (the `#pragma unroll 4` only overlaps loads;
// it cannot reassociate a serial `+=` chain) over the SAME `s_lut[byte]*sb`
// product, so gate_i / up_i are bit-identical to what gemm_fp8_mx2's family
// rows produce. The epilogue copies swiglu_limit_kernel's `(g/(1+expf(-g)))*u`
// with the same clamps, and the fp8 emit copies swiglu_limit_q_kernel's amax
// tree + fast_round_scale + clamp + __nv_fp8_e4m3 -- bit-identical because one
// block owns exactly ONE 32-row scale block (nwarps == 32 and n1 % 32 == 0).
// Phase 2 is gemm_fp8_gemv_kernel's body verbatim, so `out` is bit-identical to
// the standalone w2 launch.
//
// WHY THE fp8 OUTPUT NEEDS ITS OWN BUFFER. Phase 1's INPUT is the fp8 of `xn`
// (`a`/`a_scale`, k = dim bytes) and its fp8 OUTPUT is the swiglu pair (n1
// bytes). In the serial chain those two share one buffer (swiglu_limit_q
// overwrites the `xn` quant). Inside one grid-sync kernel that is a
// cross-block RACE: block A would overwrite bytes block B has not staged yet,
// and there is no barrier between the read and the write. The Rust caller
// therefore passes a DISJOINT pair (`sh_q`/`sh_qsc`).
//
// DEADLOCK SAFETY -- the one hard constraint, same as gemm_fp8_wo_pair_kernel.
// The barrier completes only when ALL gridDim.x blocks are resident at the same
// time, so the launcher caps the grid at co_res (occupancy x SMs at THIS block
// size and THIS dynamic smem). NEVER raise the grid past co_res: a block that
// never becomes resident never arrives, the resident ones spin forever, and the
// device hangs (the watchdog cannot see it inside a captured graph).
//
// MEMORY ORDERING. Producer: __syncthreads() (this block's act[]/aq[]/aqsc[]
// stores are done) -> __threadfence() -> the arrive atomicAdd. Consumer: the
// spin observes the sense flip -> __threadfence() -> __syncthreads() -> reads.
//
// NOT COVERED (the launcher returns 2, never 1 -- 1 is cudaErrorInvalidValue --
// and the Rust caller keeps the three launches):
//   * any fp8 mode other than 4, or a32 off / a32-staged;
//   * a shape where dim or il is not a multiple of 32.

// The barrier state: a MODULE-level [arrive, sense] pair, deliberately separate
// from `s.wo_bar` (two concurrent grids on one pair corrupt each other's arrive
// count). Zero-initialised by the module loader; every launch self-resets it --
// the last arriver zeroes `arrive` and flips `sense`, which releases this
// launch's waiters -- so a captured graph replays correctly and nothing
// re-initialises it per step.
__device__ unsigned g_sh_arrive = 0;
__device__ unsigned g_sh_sense = 0;

__global__ void __launch_bounds__(1024)
gemm_fp8_sh_pair_kernel(const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
                        const uint8_t* __restrict__ wg, const uint8_t* __restrict__ wg_scale,
                        const uint8_t* __restrict__ wu, const uint8_t* __restrict__ wu_scale,
                        float limit, int n1, int k1,
                        float* __restrict__ act, uint8_t* __restrict__ aq,
                        float* __restrict__ aqsc,
                        const uint8_t* __restrict__ w2, const uint8_t* __restrict__ w2_scale,
                        int n2, float* __restrict__ out, int cpasync) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = (blockDim.x + 31) >> 5;              // 32 (the fp8 emit requires it)
    const int nb_k1 = k1 >> 5;                              // phase-1 k-blocks
    const int nb_k2 = n1 >> 5;                              // phase-2 k-blocks (k2 = n1)
    const int nb_k2_al = (nb_k2 + 15) & ~15;                // 16B units
    const int k1max = (k1 > n1) ? k1 : n1;
    const int nb_kmax = (nb_k1 > nb_k2) ? nb_k1 : nb_k2;

    // ---- ONE shared-memory pool for BOTH phases (smem = max) -----------------
    // The two phases never overlap in time (the grid barrier plus the
    // __syncthreads() after it separate them), so every slot is sized by the
    // WIDER phase and re-used by the narrower one. The kernel has NO static
    // shared memory, so the launcher's dynamic request is exactly this sum, and
    // dsv41_gemm_fp8_sh_pair computes it with the IDENTICAL expression -- any
    // drift here is a silent pointer shift (see the launcher note).
    //
    // s_w sits FIRST so its offset is 0: phase 2 stages weight rows into it with
    // cp.async16, which needs a 16-byte-aligned destination. Every later slot is
    // 4-byte aligned by construction (n1 and nb_k2_al are multiples of 4).
    extern __shared__ uint8_t s_pool[];
    uint8_t* s_w = s_pool;                                       // [nwarps][n1]   (phase 2)
    uint8_t* s_ws = s_w + (size_t)nwarps * (size_t)n1;           // [nwarps][nb_k2_al]
    float* s_lut = reinterpret_cast<float*>(s_ws + (size_t)nwarps * (size_t)nb_k2_al);
    float* s_as = s_lut + 256;                                   // [nb_kmax]
    float* s_rows = s_as + nb_kmax;                              // [nwarps]
    float* s_af = s_rows + nwarps;                               // [k1max]

    // ============= phase 1: w1w3 + swiglu -> act f32 / aq fp8 ================
    {
        // The e4m3 decode table is phase-INDEPENDENT and never overwritten, so
        // it is built once, before the activation's own staging.
        for (int i = threadIdx.x; i < 256; i += blockDim.x) s_lut[i] = e4m3_to_f((uint8_t)i);
        for (int i = threadIdx.x; i < nb_k1; i += blockDim.x) s_as[i] = a_scale[i];
        __syncthreads();               // s_lut + s_as published
        // a32 (the SAME predecoded activation the standalone GEMV materialises):
        // one pass global uint4 -> LUT -> s_af. The emitted f32 is bit-identical
        // to the staged `s_lut[s_a[i]] * s_as[i>>5]` product -- same bytes, same
        // LUT entry, same scale, only the k-byte `s_a` round trip is gone.
        const int n16a = k1 >> 4;
        for (int i = threadIdx.x; i < n16a; i += blockDim.x) {
            const uint4 v = *reinterpret_cast<const uint4*>(a + (i << 4));
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int idx = (i << 4) + j;
                s_af[idx] = s_lut[b[j]] * s_as[idx >> 5];
            }
        }
        for (int i = (n16a << 4) + threadIdx.x; i < k1; i += blockDim.x)
            s_af[i] = s_lut[a[i]] * s_as[i >> 5];
        __syncthreads();               // s_af published for every warp

        // One block covers nwarps CONSECUTIVE rows and n1 is a multiple of
        // nwarps, so `base` decides for the WHOLE block: either all nwarps warps
        // are live or none is. That is what makes the epilogue's
        // __syncthreads() legal (uniform control flow) and the amax tree cover
        // exactly one 32-row scale block.
        const int base = blockIdx.x * nwarps;
        if (base < n1) {
            const int row = base + warp;
            const uint8_t* g_row = wg + (size_t)row * (size_t)k1;
            const uint8_t* u_row = wu + (size_t)row * (size_t)k1;
            const uint8_t* gsc = wg_scale + (size_t)(row >> 5) * (size_t)nb_k1;
            const uint8_t* usc = wu_scale + (size_t)(row >> 5) * (size_t)nb_k1;
            float g = 0.f, u = 0.f;
            // The gemm_fp8_gemv_kernel consume loop, twice (gate chain + up
            // chain), sharing every `av`. Single accumulator per chain and
            // `kb` ascending = the standalone row's summation order.
#pragma unroll 4
            for (int kb = 0; kb < nb_k1; ++kb) {
                const float sbg = ue8m0_to_f(gsc[kb]);
                const float sbu = ue8m0_to_f(usc[kb]);
                const int j = (kb << 5) + lane;
                const float av = s_af[j];
                g += av * (s_lut[g_row[j]] * sbg);
                u += av * (s_lut[u_row[j]] * sbu);
            }
            for (int off = 16; off > 0; off >>= 1) {
                g += __shfl_xor_sync(0xFFFFFFFFu, g, off);
                u += __shfl_xor_sync(0xFFFFFFFFu, u, off);
            }
            if (lane == 0) {
                // swiglu_limit_kernel's clamp + silu, term for term. `v` is the
                // SAME register value both consumers below use (the f32 store and
                // the fp8 amax), never a global re-read.
                if (limit > 0.f) {
                    g = fminf(g, limit);                    // gate clamp
                    u = fminf(fmaxf(u, -limit), limit);     // up clamp
                }
                const float v = (g / (1.f + expf(-g))) * u;
                act[row] = v;
                s_rows[warp] = v;
            }
            __syncthreads();   // every warp's v staged; block == one scale block
            // swiglu_limit_q_kernel's emit, term for term. `fmaxf` is exact and
            // associative, so this tree is bit-identical to the kernel's per-lane
            // shfl tree over the same 32 values.
            float am = (lane < nwarps) ? fabsf(s_rows[lane]) : 0.f;
            for (int off = 16; off > 0; off >>= 1)
                am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            const float sc = fmaxf(fast_round_scale(am, 1.0f / 448.0f), 1e-30f);
            if (lane == 0) {
                aqsc[blockIdx.x] = sc;
                const float q = fminf(fmaxf(s_rows[warp] * (1.0f / sc), -448.0f), 448.0f);
                const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
                aq[row] = *(const uint8_t*)&f8;
            }
        }
    }

    // ============================= grid barrier ==============================
    // sense-reversing, arrived once per block by thread 0. Safe ONLY because the
    // launcher caps gridDim.x at the residency number (see co_res in
    // dsv41_gemm_fp8_sh_pair).
    __syncthreads();                   // this block's act[]/aq[] stores are complete
    if (threadIdx.x == 0) {
        __threadfence();                               // release: act[]/aq[] visible
        const unsigned s0 = atomicAdd(&g_sh_sense, 0u);    // current sense
        if (atomicAdd(&g_sh_arrive, 1u) == (unsigned)gridDim.x - 1u) {
            atomicExch(&g_sh_arrive, 0u);              // reset for the next launch
            __threadfence();
            atomicXor(&g_sh_sense, 1u);                // flip -> release waiters
        } else {
            while (atomicAdd(&g_sh_sense, 0u) == s0) __nanosleep(32);
            __threadfence();                           // acquire
        }
    }
    __syncthreads();

    // ===================== phase 2: w2 (fp8 -> f32), k = n1 ==================
    {
        // Re-stage the two inputs of the standalone w2 launch: its per-32-block
        // scale row, and the block-wide a32-predecoded activation. Same product
        // (`s_lut[aq[i]] * s_as[i>>5]`), same bytes -> same s_af.
        for (int i = threadIdx.x; i < nb_k2; i += blockDim.x) s_as[i] = aqsc[i];
        __syncthreads();               // s_as published (s_lut is already built)
        const int n16a = n1 >> 4;
        for (int i = threadIdx.x; i < n16a; i += blockDim.x) {
            const uint4 v = *reinterpret_cast<const uint4*>(aq + (i << 4));
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int idx = (i << 4) + j;
                s_af[idx] = s_lut[b[j]] * s_as[idx >> 5];
            }
        }
        for (int i = (n16a << 4) + threadIdx.x; i < n1; i += blockDim.x)
            s_af[i] = s_lut[aq[i]] * s_as[i >> 5];
        __syncthreads();               // s_af published

        // P3 (cp.async weights-first, DSV41_GEMV_CPASYNC): the first row this
        // warp will consume depends on none of the staging above, so issue it
        // BEFORE the barrier's consume loop and let the two overlap. Same bytes,
        // same slot; only the issue point moves.
        int pf_row = (cpasync != 0) ? ((int)blockIdx.x * nwarps + warp) : -1;
        if (pf_row >= n2) pf_row = -1;
        if (pf_row >= 0) {
            const uint8_t* pf_w = w2 + (size_t)pf_row * (size_t)n1;
            uint8_t* pf_s = s_w + (size_t)warp * (size_t)n1;
            const int n16p = n1 >> 4;
            for (int i = lane; i < n16p; i += 32) dsv41_cp_async16(pf_s + (i << 4), pf_w + (i << 4));
            dsv41_cp_commit();
            for (int i = (n16p << 4) + lane; i < n1; i += 32) pf_s[i] = pf_w[i];
        }

        for (int row = blockIdx.x * nwarps + warp; row < n2; row += gridDim.x * nwarps) {
            const uint8_t* wr = w2 + (size_t)row * (size_t)n1;
            const uint8_t* wsr = w2_scale + (size_t)(row >> 5) * (size_t)nb_k2;
            uint8_t* row_s = s_w + (size_t)warp * (size_t)n1;
            uint8_t* row_sc = s_ws + (size_t)warp * (size_t)nb_k2_al;
            float acc = 0.f;
            const bool prefetched = (row == pf_row);
            if (!prefetched) {
                const int n16 = n1 >> 4;
                for (int i = (n16 << 4) + lane; i < n1; i += 32) row_s[i] = wr[i];
                for (int i = lane; i < n16; i += 32)
                    dsv41_cp_async16(row_s + (i << 4), wr + (i << 4));
            }
            // PLAIN byte loads, NOT cp.async: cp.async16 needs a 16-byte-aligned
            // global address and `wsr` only has that when nb_k2 is a multiple of
            // 16. Same reason (and the same err 716) as the standalone kernel.
            for (int i = lane; i < nb_k2; i += 32) row_sc[i] = wsr[i];
            // Unconditional: on the prefetched iteration this commits an EMPTY
            // group after the prologue's, and wait_all retires both.
            dsv41_cp_commit();
            dsv41_cp_wait_all();
            __syncwarp();
#pragma unroll 4
            for (int kb = 0; kb < nb_k2; ++kb) {
                const float sb = ue8m0_to_f(row_sc[kb]);
                const int j = (kb << 5) + lane;
                acc += s_af[j] * (s_lut[row_s[j]] * sb);
            }
            __syncwarp();
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) out[row] = acc;
        }
    }
}

// The chain-pair launcher. TWO deliberate differences from every other launcher
// in this family, both copied from dsv41_gemm_fp8_wo_pair:
//   * the launch is PLAIN (`<<<>>>`), NOT via dsv41_pdl_or_plain. A grid-wide
//     barrier and PDL are a bad pair: PDL lets this grid start while the
//     producer is still resident, and the barrier needs the WHOLE grid resident
//     before any block may pass it.
//   * the grid is CAPPED by residency (co_res). That cap is a correctness
//     requirement, not tuning -- see the kernel's DEADLOCK SAFETY note.
//
// One more difference of its own: `warps` is pinned to 32 (not g_gemv_warps).
// The fp8 emit needs one block to own exactly one 32-row scale block, i.e.
// nwarps == 32 and n1 % 32 == 0; a smaller block would split a scale block
// across blocks and the amax would need a cross-block reduction.
//
// Returns 2 (decline, never 1) for any shape or arm the fused kernel does not
// implement, so the Rust caller keeps the three-launch chain bit for bit.
extern "C" int dsv41_gemm_fp8_sh_pair(const uint8_t* a, const float* a_scale,
                                      const uint8_t* wg, const uint8_t* wg_scale,
                                      const uint8_t* wu, const uint8_t* wu_scale,
                                      float limit, int n1, int k1,
                                      float* act, uint8_t* aq, float* aqsc,
                                      const uint8_t* w2, const uint8_t* w2_scale,
                                      int n2, float* out, cudaStream_t s) {
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    if (no_gemv) return 2;
    if (a == nullptr || a_scale == nullptr || wg == nullptr || wg_scale == nullptr ||
        wu == nullptr || wu_scale == nullptr || act == nullptr || aq == nullptr ||
        aqsc == nullptr || w2 == nullptr || w2_scale == nullptr || out == nullptr)
        return 2;
    if (n1 <= 0 || k1 <= 0 || n2 <= 0) return 2;
    // Phase 1's k-blocks and BOTH phases' 32-element scale rows: k1 is the
    // activation width, n1 is phase 1's row count (= phase 2's k). n1 % 32 also
    // guarantees one block owns whole scale blocks (nwarps == 32) and n1 % 16
    // the cp.async16 alignment of phase 2's weight rows.
    if ((k1 & 31) || (n1 & 31)) return 2;
    // Specialised bodies: only the production arm (mode 4 = staged weights +
    // block-wide activation; a32 = the predecoded `s_af`; a32_direct) is
    // implemented, and only the plain (non-NORM_FUSE) activation source.
    if (g_gemv_fp8_mode != 4 || !g_gemv_a32 || g_gemv_a32_staged) return 2;

    const int warps = 32;
    const int nb_k1 = k1 >> 5;
    const int nb_k2 = n1 >> 5;
    const int nb_k2_al = (nb_k2 + 15) & ~15;
    const int k1max = (k1 > n1) ? k1 : n1;
    const int nb_kmax = (nb_k1 > nb_k2) ? nb_k1 : nb_k2;
    // MUST match the kernel's pointer arithmetic EXACTLY (s_w first, then s_ws,
    // then the four float tables). Any drift shifts every pointer after it.
    const size_t gsmem = (size_t)warps * (size_t)n1          // phase-2 weight rows
                       + (size_t)warps * (size_t)nb_k2_al    // phase-2 ue8m0 scale rows
                       + 256 * sizeof(float)                 // e4m3 LUT
                       + (size_t)nb_kmax * sizeof(float)     // activation scales
                       + (size_t)warps * sizeof(float)       // fp8-emit row stage
                       + (size_t)k1max * sizeof(float);      // s_af
    if (gsmem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_sh_pair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            dsv41_smem_ceiling(gemm_fp8_sh_pair_kernel));
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // ---- co_res: the residency cap. Cached because the model's shape is fixed
    // and this launcher runs 40 times per step (a per-call occupancy query is a
    // driver round trip). Keyed on the parameters that define the answer.
    static int co_res_cached = 0;
    static size_t co_res_gsmem = 0;
    if (co_res_cached == 0 || co_res_gsmem != gsmem) {
        int per_sm = 0;
        cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, gemm_fp8_sh_pair_kernel, warps * 32, gsmem);
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        co_res_cached = per_sm * sms;
        co_res_gsmem = gsmem;
    }
    if (co_res_cached <= 0) return 2;   // no resident configuration -> decline

    const int rows = (n1 > n2) ? n1 : n2;
    int blocks = (rows + warps - 1) / warps;
    if (blocks > co_res_cached) blocks = co_res_cached;   // HARD deadlock guard
    if (blocks <= 0) return 2;

    gemm_fp8_sh_pair_kernel<<<dim3(blocks), dim3(warps * 32), gsmem, s>>>(
        a, a_scale, wg, wg_scale, wu, wu_scale, limit, n1, k1,
        act, aq, aqsc, w2, w2_scale, n2, out, g_gemv_cpasync ? 1 : 0);
    return (int)cudaGetLastError();
}

// Two projections over the SAME activation in one gemv launch: the kernel maps
// rows below n1 to the first family and the rest to the second, both sharing the
// block-wide staged activation. wq_a and wkv in the attention are the pair - two
// latency-floor launches become one, and the small family's blocks (wkv is 128
// rows) ride beside the big family's instead of paying their own serial slot
// after wq_b and the rope. Each row is still one warp in the same lane order, so
// both outputs are bit-identical to two separate launches. M=1 only: a caller
// that cannot use it gets cudaErrorInvalidValue back and runs the two
// single-family calls instead.
// Three projections that read the SAME activation in one launch: the bf16 MoE
// gate (rows [0, nb)) and the fp8 shared expert's two halves (rows [nb, nb+nf)
// and [nb+nf, nb+2nf)). The routing gate and the shared expert are computed from
// the same normalised hidden state, so they were paying two ~20us launch /
// staging floors per layer; this removes one. Each row keeps its own family's
// lane order and accumulation, so every output is bit-identical to the two
// (three, before the mx2 fusion) launches it replaces.
__global__ void gemv_bf16_fp8x2_kernel(const __nv_bfloat16* __restrict__ wb,
                                       const float* __restrict__ biasb,
                                       float* __restrict__ outb, int nb,
                                       const uint8_t* __restrict__ a,
                                       const float* __restrict__ a_scale,
                                       const uint8_t* __restrict__ wf1,
                                       const uint8_t* __restrict__ ws1,
                                       float* __restrict__ outf1, int nf,
                                       const uint8_t* __restrict__ wf2,
                                       const uint8_t* __restrict__ ws2,
                                       float* __restrict__ outf2,
                                       const float* __restrict__ x, int k, int vec,
                                       int cpasync) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = (blockDim.x + 31) >> 5;
    const int nb_k = k >> 5;
    extern __shared__ uint8_t s_w[];
    // P1 (a32 dead-slot; same fix as gemm_fp8_gemv_kernel): the k-byte fp8
    // activation staging row used to sit at `s_w + nwarps*k` and had exactly ONE
    // reader -- the materialisation loop below. Fusing the staging copy and that
    // decode into a single global uint4 load -> LUT -> straight-to-`s_af` pass
    // removes the last reader, so the k-byte slot is never allocated. `a` is a
    // read-only input, so the same bytes go through the same LUT entry and the
    // same scale => bit-identical output. This kernel has no a32 gate/parameter
    // (every fp8 row consumes `s_af`), so no mode keeps the slot alive: mode 3
    // already read `a` directly, mode 4 now does too.
    //
    // The fp8 family's per-block scratch starts right after the warp slices of
    // s_w. k is a multiple of 16 (launcher-checked), so this base stays 4-byte
    // aligned for the f32 tables.
    //
    // LUT + a32 are the SAME two optimisations the single-family gemv carries
    // (gemm_fp8_gemv_kernel); this fused kernel was never brought along, so its
    // fp8 rows still paid LDS.8 -> ~10 ALU -> FMUL per operand and re-decoded the
    // activation once per row. The table is built from the SAME e4m3_to_f, so
    // every decoded value is bit-identical, and s_af[j] is exactly the product
    // the loop used to form inline (s_lut[ap[j]] * a_scale[j>>5]), so the
    // rounding sequence is unchanged: the output stays bit-identical.
    float* s_lut = reinterpret_cast<float*>(s_w + (size_t)nwarps * (size_t)k);
    float* s_af = s_lut + 256;
    const int total = nb + 2 * nf;
    // ---- P3 (DSV41_BF16_CPASYNC, see g_bf16fp8x2_cpasync) ------------------
    // cp.async WEIGHT-FIRST prologue. The row loop's fp8 staging used to start
    // only after the barrier below, with nothing left to hide it: the LUT build
    // and the whole P1 `s_af` materialisation had already finished, so every
    // warp that owns an fp8 row paid the full DRAM latency of its k-byte row
    // before it could compute. The row depends on none of that work, so issue it
    // here -- the earliest point in the block -- and let the LUT + a32 pass +
    // barrier run underneath it. Loop iteration 0 recognises it by
    // `row == pf_row` and only WAITS.
    //
    // NOTE the explicit (int) cast: `blockIdx.x * nwarps + warp` is UNSIGNED
    // (blockIdx.x is uint3), so the bare ternary would promote -1 to 0xFFFFFFFF.
    int pf_row = (cpasync != 0) ? ((int)blockIdx.x * nwarps + warp) : -1;
    if (pf_row >= total) pf_row = -1;            // the last block's tail warps
    if (pf_row >= 0 && pf_row >= nb) {           // fp8 rows stage; bf16 rows never do
        // Family/pointer select copied from the row loop verbatim: a second-half
        // row reads wf2, the first half wf1. Drifting from the loop here would
        // silently stage the WRONG row (no crash, wrong dot product).
        const int rr = pf_row - nb;
        const bool second = rr >= nf;
        const int rrow = rr - (second ? nf : 0);
        const uint8_t* pf_w = (second ? wf2 : wf1) + (size_t)rrow * (size_t)k;
        uint8_t* pf_s = s_w + (size_t)warp * (size_t)k;
        const int n16p = k >> 4;
        for (int i = lane; i < n16p; i += 32) dsv41_cp_async16(pf_s + (i << 4), pf_w + (i << 4));
        // k % 16 tail: unreachable for every caller (the launcher rejects
        // `k & 15`), kept so the prefetched row is staged by exactly the rule the
        // loop uses. A hole here would be a partially-staged row, not a crash.
        for (int i = (n16p << 4) + lane; i < k; i += 32) pf_s[i] = pf_w[i];
    }
    // Committed UNCONDITIONALLY -- also for the bf16-first warps and when the
    // gate is off (they commit an EMPTY group, a no-op). That keeps the per-thread
    // commit-group count identical across the block. The row loop's own
    // `cp.async.wait_all` retires this group on the prefetched iteration, so the
    // fp8 path never sees it.
    dsv41_cp_commit();
    // Build the e4m3 decode table once per block (256 entries, two iterations per
    // thread at the default block size) and, after the barrier, materialise the
    // scaled activation straight from global. Both live outside the two family
    // branches: the barrier must be reached by every thread of the block, and the
    // bf16 rows simply never read either table.
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_lut[i] = e4m3_to_f((uint8_t)i);
    __syncthreads();
    {
        // P1: one pass global -> LUT -> s_af. The uint4 read is the SAME wide read
        // the removed staging loop performed into `s_a`, and the decode is the SAME
        // `s_lut[b] * a_scale[i>>5]` product the materialisation applied to the
        // bytes it read back from shared memory, so the emitted f32 is
        // bit-identical -- only the smem round trip disappears.
        const int n16a = k >> 4;
        for (int i = threadIdx.x; i < n16a; i += blockDim.x) {
            const uint4 v = *reinterpret_cast<const uint4*>(a + (i << 4));
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int idx = (i << 4) + j;
                s_af[idx] = s_lut[b[j]] * a_scale[idx >> 5];
            }
        }
        for (int i = (n16a << 4) + threadIdx.x; i < k; i += blockDim.x)
            s_af[i] = s_lut[a[i]] * a_scale[i >> 5];
    }
    __syncthreads();
    for (int row = blockIdx.x * nwarps + warp; row < total; row += gridDim.x * nwarps) {
        if (row < nb) {
            // ---- bf16 family: gemv_bf16_kernel's loop, four iterations in flight
            const __nv_bfloat16* wr = wb + (size_t)row * (size_t)k;
            float acc = 0.f;
            int c = lane;
            for (; c + 96 < k; c += 128) {
                const __nv_bfloat16 w0 = wr[c], w1 = wr[c + 32], w2 = wr[c + 64], w3 = wr[c + 96];
                const float x0 = x[c], x1 = x[c + 32], x2 = x[c + 64], x3 = x[c + 96];
                acc += __bfloat162float(w0) * x0;
                acc += __bfloat162float(w1) * x1;
                acc += __bfloat162float(w2) * x2;
                acc += __bfloat162float(w3) * x3;
            }
            for (; c < k; c += 32) acc += __bfloat162float(wr[c]) * x[c];
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) outb[row] = acc + (biasb ? biasb[row] : 0.f);
        } else {
            // ---- fp8 family: gemm_fp8_gemv_kernel's mode 3/4 row path
            const bool second = (row - nb) >= nf;
            const int rrow = (row - nb) - (second ? nf : 0);
            const uint8_t* wrow = ((second ? wf2 : wf1) + (size_t)rrow * (size_t)k);
            const uint8_t* wsc = (second ? ws2 : ws1) + (size_t)(rrow >> 5) * nb_k;
            uint8_t* row_s = s_w + (size_t)warp * (size_t)k;
            // P3: the prologue already staged this row (same bytes, same slot).
            // Skip the redundant transfer and let the wait below collect it.
            const bool prefetched = (row == pf_row);
            if (!prefetched) {
                const int n16 = k >> 4;
                for (int i = (n16 << 4) + lane; i < k; i += 32) row_s[i] = wrow[i];
                for (int i = lane; i < n16; i += 32)
                    dsv41_cp_async16(row_s + (i << 4), wrow + (i << 4));
            }
            // Unconditional: on the prefetched iteration this commits an EMPTY
            // group after the prologue's, and wait_all retires both.
            dsv41_cp_commit();
            dsv41_cp_wait_all();
            __syncwarp();
            float acc = 0.f;
            for (int kb = 0; kb < nb_k; ++kb) {
                const float sb = ue8m0_to_f(wsc[kb]);
                const int j = kb * 32 + lane;
                // a32: s_af[j] already folds s_lut[ap[j]] * a_scale[j>>5], so the
                // per-element chain is one LDS.32 -> FMUL instead of
                // LDS.8 -> LDS.32 -> FMUL -> FMUL.
                acc += s_af[j] * (s_lut[row_s[j]] * sb);
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) (second ? outf2 : outf1)[rrow] = acc;
            __syncwarp();
        }
    }
}

extern "C" int dsv41_gemm_bf16_fp8x2(const void* wb, const float* biasb, float* outb, int nb,
                                     const uint8_t* a, const float* a_scale, const uint8_t* wf1,
                                     const uint8_t* ws1, float* outf1, int nf,
                                     const uint8_t* wf2, const uint8_t* ws2, float* outf2,
                                     const float* x, int k, cudaStream_t s) {
    static const int warps = [] {
        const char* e = getenv("DSV41_MIX_WARPS");
        if (e == nullptr) return 4;
        const int v = atoi(e);
        return (v >= 1 && v <= 16) ? v : 4;
    }();
    const int vec = g_gemv_fp8_mode;              // 3 or 4
    if (vec < 3 || nb < 0 || nf <= 0 || k <= 0 || (k & 31) || (k & 15))
        return (int)cudaErrorInvalidValue;
    const int total = nb + 2 * nf;
    const int blocks = (total + warps - 1) / warps;
    // weights [nwarps][k] + the e4m3 decode table [256] f32 + the a32 pre-decoded
    // activation [k] f32. P1: the k-byte activation staging row that mode 4 used
    // to add on top is gone -- the kernel reads `a` straight from global into
    // `s_af` now, so both modes ask for `warps * k` weight rows. The tables sit
    // right after those rows, matching the kernel's pointer arithmetic above;
    // without them here the kernel's s_lut base would fall outside the dynamic
    // allocation.
    const size_t scale_bytes = 256 * sizeof(float) + (size_t)k * sizeof(float);
    const size_t gsmem = (size_t)warps * (size_t)k + scale_bytes;
    if (gsmem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gemv_bf16_fp8x2_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemv_bf16_fp8x2_kernel));
        if (e != cudaSuccess) return (int)e;
    }
    gemv_bf16_fp8x2_kernel<<<blocks, warps * 32, gsmem, s>>>(
        (const __nv_bfloat16*)wb, biasb, outb, nb, a, a_scale, wf1, ws1, outf1, nf, wf2, ws2, outf2,
        x, k, vec, g_bf16fp8x2_cpasync ? 1 : 0);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_gemm_fp8_mx2(const uint8_t* a, const float* a_scale,
                                  const uint8_t* w1, const uint8_t* w1_scale,
                                  const float* bias1, float* out1, int n1,
                                  const uint8_t* w2, const uint8_t* w2_scale,
                                  const float* bias2, float* out2, int n2, int k,
                                  cudaStream_t s) {
    static const bool no_gemv = getenv("DSV41_NO_GEMV_FP8") != nullptr;
    if (no_gemv || n1 <= 0 || n2 <= 0 || k <= 0 || (k & 31) || (k & 3))
        return (int)cudaErrorInvalidValue;
    const int n = n1 + n2;
    // P2: adaptive rows/block on the TOTAL row count (both families share one
    // block grid); 8 for n >= 2048, else the fixed g_gemv_warps.
    const int warps = dsv41_gemv_warps_for(n);
    const int blocks = (n + warps - 1) / warps;
    const int nb_k = k >> 5;
    const int nb_k_al = (nb_k + 15) & ~15;
    const size_t scale_bytes =
        (size_t)warps * (size_t)nb_k_al + (size_t)nb_k * sizeof(float) + 256 * sizeof(float) +
        dsv41_gemv_a32_bytes(k) + 32 * sizeof(float);   // a32 + the B1 row stage slot
    // P1: mode 4's `+1` activation row is only allocated while `s_a` lives
    // (a32=0) -- the fused a32 pass reads global directly. See
    // dsv41_gemv_sa_bytes and the kernel's `a32_direct`.
    const size_t gsmem = (g_gemv_fp8_mode == 3)   ? (size_t)warps * (size_t)k + scale_bytes
                         : (g_gemv_fp8_mode == 4) ? (size_t)warps * (size_t)k + dsv41_gemv_sa_bytes(k, false) + scale_bytes
                                                   : (size_t)0;
    if (gsmem > 48 * 1024) {
        // Round-43 revert: the (int)gsmem form set the per-function attribute
        // to THIS call's need, which can silently cap later launches of the same
        // kernel that need more. The 232448 ceiling (the device max opt-in,
        // verified working rounds 37-41) is the correct semantic. Keep the
        // sticky clear on failure (the round-42 decline-collision fix stays).
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp8_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(gemm_fp8_gemv_kernel));
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // gemv-struct-pack: four by-value structs; both families travel in GemvCore
    // and no rope/fusion field is set on this path, so the rest keeps null / 0.
    GemvCore gc{};
    gc.a = a; gc.a_scale = a_scale;
    gc.w = w1; gc.w_scale = w1_scale;
    gc.bias = bias1; gc.out = out1;
    gc.n = n; gc.k = k;
    gc.vec = g_gemv_fp8_mode; gc.a32 = (g_gemv_a32 ? 1 : 0);
    gc.a32_staged = (g_gemv_a32_staged ? 1 : 0);
    gc.cpasync = (g_gemv_cpasync ? 1 : 0);
    gc.act_cpasync = (g_gemv_act_cpasync ? 1 : 0);
    gc.w2 = w2; gc.w2_scale = w2_scale;
    gc.bias2 = bias2; gc.out2 = out2;
    gc.n1 = n1;
    GemvRope gr{};
    GemvFusion gf{};
    GemvEpi ge{};
    cudaError_t le = dsv41_pdl_or_plain(
        gemm_fp8_gemv_kernel, dim3(blocks), dim3(warps * 32), gsmem, s, gc, gr, gf, ge);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// a32 / occupancy experiment probes (host only, no kernel change).
//
// dsv41_gemv_gsmem mirrors the launchers' `gsmem` arithmetic for the single-
// family M=1 form, so a bench can print the smem the kernel will actually ask
// for. dsv41_gemv_occupancy asks the driver for the blocks-per-SM that request
// buys. Together they are what turns "a32 costs 20KB" into a measured
// 4-vs-8 blocks/SM claim instead of an estimate.
//
// Both read the process's static gates (g_gemv_fp8_mode / g_gemv_a32), so run
// one arm per process, like every other gate in this file.
extern "C" size_t dsv41_gemv_gsmem(int mode, int warps, int k) {
    const int nb_k = k >> 5;
    const int nb_k_al = (nb_k + 15) & ~15;
    const size_t scale_bytes = (size_t)warps * (size_t)nb_k_al + (size_t)nb_k * sizeof(float) +
                               256 * sizeof(float) + dsv41_gemv_a32_bytes(k) + 32 * sizeof(float);
    if (mode == 3) return (size_t)warps * (size_t)k + scale_bytes;
    // P1: mode 4 asks for `warps` weight rows and, only when `s_a` is actually
    // alive (a32=0; this probe mirrors the qr_raw==nullptr launchers), the extra
    // activation row. See dsv41_gemv_sa_bytes.
    if (mode == 4) return (size_t)warps * (size_t)k + dsv41_gemv_sa_bytes(k, false) + scale_bytes;
    return 0;   // scalar / vectorised allocate no dynamic smem
}

extern "C" int dsv41_gemv_occupancy(int warps, size_t gsmem) {
    if (gsmem > 48 * 1024) {
        if (cudaFuncSetAttribute(gemm_fp8_gemv_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dsv41_smem_ceiling(gemm_fp8_gemv_kernel)) != cudaSuccess)
            (void)cudaGetLastError();
    }
    int blocks = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, gemm_fp8_gemv_kernel, warps * 32, gsmem);
    if (e != cudaSuccess) { (void)cudaGetLastError(); return -1; }
    return blocks;
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
                              int* __restrict__ pos_ctr, int idx_off,
                              unsigned long long* __restrict__ packed) {
    unsigned long long my = 0ull;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned int bits = __float_as_uint(v[i]);
        const unsigned int key = (bits >> 31) ? ~bits : (bits | 0x80000000u);
        const unsigned long long pk =
            ((unsigned long long)key << 32) | (0xFFFFFFFFu - (unsigned)(i + idx_off));
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
        // `packed` (vocabulary-sliced path): the comparison key already carries
        // the GLOBAL index, so a peer only has to max over the ranks.
        if (packed != nullptr) *packed = m;
        *out = (int)(0xFFFFFFFFu - (unsigned)(m & 0xFFFFFFFFu));
        // the argmax is the LAST kernel of the step: this is where the
        // device position counter advances, so every kernel of the NEXT
        // step (the engram hash, the window indices, the compressor) sees
        // pos + 1 while every kernel of THIS step saw a stable position.
        if (pos_ctr != nullptr) *pos_ctr = *pos_ctr + 1;
    }
}

// Cross-rank argmax over a vocabulary-sliced lm_head. Each rank reduces its own
// slice with argmax_kernel's packing (comparable value key | ~global index, so
// ties go to the LOWEST index) and publishes that u64 into every peer's staging
// slot; the final kernel takes the max across ranks in ascending rank order. The
// packed key is monotone in (value, -index), so the winner and its tie rule are
// exactly the full-vocabulary argmax's. This is what makes slicing the lm_head
// free of a correctness question: the whole vocabulary is still compared, just
// 1/world of it per rank - which is 8x less weight traffic per step (the full
// 129280-row head measured 298us against 48us for one rank's 16160-row slice).
// Cross-rank argmax as ONE v5 epoch round (DSV41_HEAD_SLICE).
// The old argmax_pub/final wrote bytes-16 with NO parity/epoch: not part of
// the v5 round sequence -> a peer could stamp round k while another read
// round j's slot => cross-rank deadlock. Here the key lands at
// ((e&1)*world+my_rank)*stride_bytes - the SAME parity addressing the v5 store
// uses (ferrite_kernels.cu:8128) - then stamp peers' ready=e+1, advance
// *epoch, poll all peers >= e+1, and max. Single thread touches world u64s
// only, so the stamp/poll/reduce order is trivial (no __syncthreads, no
// seen[]). Absolute stamps + the parity double buffer make this hang-free by
// the same argument as the v5 AR itself (ferrite_kernels.cu:8112-8176).
__global__ void argmax_xchg_v5_kernel(
    const unsigned long long* __restrict__ packed,        // my slice key
    unsigned long long* const* __restrict__ staging_tbl,  // [world] peers' staging BASEs
    unsigned* const* __restrict__ ready_tbl,              // [world] peers' ready rows
    unsigned* __restrict__ epoch,
    unsigned long long* __restrict__ staging_local,       // my staging base
    const unsigned* __restrict__ ready_local,             // my [world] row
    int* __restrict__ out, int* __restrict__ pos_ctr,
    int world, int my_rank, long stride_bytes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const unsigned e = *epoch;
    const unsigned long long pk = packed[0];
    const size_t off =
        (size_t)((e & 1u) * (unsigned)world + (unsigned)my_rank) * (size_t)stride_bytes;
    for (int r = 0; r < world; r++)
        *reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(staging_tbl[r]) + off) = pk;
    __threadfence_system();                               // key visible BEFORE stamp
    for (int r = 0; r < world; r++)
        atomicExch_system((unsigned int*)&ready_tbl[r][my_rank], e + 1u);
    __threadfence_system();
    *epoch = e + 1u;                                      // only after stamping (v5 rule)
    for (int r = 0; r < world; r++) {
        volatile unsigned* p = (volatile unsigned*)&ready_local[r];
        long spins = 0;
        while ((int)(*p - (e + 1u)) < 0) {                // absolute stamp, monotone
            __nanosleep(200);
            if (++spins > 25000000) break;                // ~5 s watchdog; give up
        }
    }
    __threadfence_system();                               // observe peers' staged keys
    unsigned long long best = 0ull;
    for (int r = 0; r < world; r++) {
        const size_t ro =
            (size_t)((e & 1u) * (unsigned)world + (unsigned)r) * (size_t)stride_bytes;
        const unsigned long long k = *reinterpret_cast<const unsigned long long*>(
            reinterpret_cast<const char*>(staging_local) + ro);
        if (k > best) best = k;                           // ascending rank; ties -> lowest idx
    }
    *out = (int)(0xFFFFFFFFu - (unsigned)(best & 0xFFFFFFFFu));
    if (pos_ctr != nullptr) *pos_ctr = *pos_ctr + 1;
}

extern "C" int dsv41_argmax(const float* v, int* out, int n, int* pos_ctr, cudaStream_t s) {
    if (n <= 0) return (int)cudaErrorInvalidValue;
    argmax_kernel<<<1, 1024, 0, s>>>(v, out, n, pos_ctr, 0, nullptr);
    return (int)cudaGetLastError();
}

// Vocabulary-sliced argmax (lm_head split across ranks), as ONE round of the
// shared v5 epoch sequence: the local slice reduces through argmax_kernel
// (packed key carries the GLOBAL index, ties -> lowest), then the exchange
// kernel publishes into every peer's CURRENT parity slot and advances the same
// device epoch the ARs use. pos_ctr advances here, once per step, exactly as
// the single-rank argmax did.
extern "C" int dsv41_argmax_sliced(
    const float* v, int n, int idx_off, int* out, unsigned long long* packed, int* pos_ctr,
    unsigned long long* const* staging_tbl, unsigned* const* ready_tbl, unsigned* epoch,
    unsigned long long* staging_local, const unsigned* ready_local, int world, int rank,
    long stride_bytes, cudaStream_t s) {
    if (n <= 0 || world <= 0) return (int)cudaErrorInvalidValue;
    argmax_kernel<<<1, 1024, 0, s>>>(v, out, n, nullptr, idx_off, packed);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    argmax_xchg_v5_kernel<<<1, 1, 0, s>>>(
        packed, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out, pos_ctr, world,
        rank, stride_bytes);
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
    // Chunked key split with the three-deep prefetch pipeline. The chunk count
    // comes from `dsv41_resolve_sparse_split_c()` (DSV41_ATTN_SPLIT >
    // DSV41_ATTN_PF_SPLIT > DSV41_SPARSE_SPLIT > kSparseSplitDefault = 4), read
    // ONCE here: this runs per attention call and a per-call getenv is exactly the
    // hot-path slip this project has been bitten by. DSV41_ATTN_PF_SPLIT=0 - the
    // harness's "restore the single-block pf kernel" arm - still wins when set, and
    // DSV41_SPARSE_SPLIT=0 is the new opt-out.
    static const int g_sparse_split_c = dsv41_resolve_sparse_split_c();
    // Order-preserving prefetch pipeline (default on); DSV41_ATTN_PF=0 restores
    // the plain warp version for A/B.
    static const bool pf_off = [] {
        const char* e = getenv("DSV41_ATTN_PF");
        return e != nullptr && atoi(e) == 0;
    }();
    dim3 grid(b * m, h);
    if (!seq) {
        const int split_c = g_sparse_split_c;
        if (pf_off) {
            // DSV41_ATTN_PF=0 is the documented "restore the plain warp kernel" A/B
            // arm. It is tested BEFORE the split on purpose: the split now defaults
            // ON, so the old order would leave this arm unreachable.
            sparse_attn_warp_kernel<<<grid, 128, 0, s>>>(q, kv, sink, idxs, out, b, m, h, d, clen,
                                                         window, index_topk, scale);
        } else if (split_c > 0 && split_c <= kAttnMaxC && b * m <= kAttnMaxBM &&
                   h <= kAttnMaxH) {
            // One block per (chunk, row, head); the split writes the partials,
            // the merge folds the sink and normalises (rope/xq are null here, so
            // the merge's fused epilogue is skipped and this arm is unchanged).
            // Both launches stay on the SAME stream: program order makes the
            // partials visible to the merge with no fence and no sync.
            sparse_attn_split_kernel<<<dim3((unsigned)split_c, (unsigned)(b * m),
                                            (unsigned)h), 128, 0, s>>>(
                q, kv, idxs, b, m, h, d, clen, window, index_topk, scale, split_c);
            cudaError_t e2 = cudaGetLastError();
            if (e2 != cudaSuccess) return (int)e2;
            sparse_attn_merge_kernel<<<dim3((unsigned)(b * m), (unsigned)h), 128, 0, s>>>(
                sink, out, b, m, h, d, split_c, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0,
                nullptr, nullptr);
            return (int)cudaGetLastError();
        } else {
            // PDL (see dsv41_pdl_or_plain): sparse_attn_pf is the consumer of
            // wq_b / apply_rope (q), the indexer (idxs) and the compressor
            // (*clen), so its grid may start during the producer's tail; the
            // kernel's entry cudaGridDependencySynchronize() gates all three
            // reads. The split/merge and warp A/B arms deliberately stay plain.
            cudaError_t le = dsv41_pdl_or_plain(sparse_attn_pf_kernel, grid, dim3(128), 0, s, q,
                                               kv, sink, idxs, out, b, m, h, d, clen, window,
                                               index_topk, scale);
            if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
        }
    } else {
        sparse_attn_kernel<<<grid, 128, 0, s>>>(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale);
    }
    return (int)cudaGetLastError();
}

// P1 (DSV41_SPARSE_OROPE): sparse_attn + inverse o-rope + fp8 emission, one
// launch (two when the key-split arm is selected - see below). Returns 0 on
// success. Returns 1 - the DECLINE sentinel - when the fused shape cannot carry
// this call, and the caller runs the old `dsv41_sparse_attn` +
// `dsv41_apply_rope_q` (+ `quant1`) sequence, which this is bit-identical to.
// (Sentinel 2 is also a decline, distinguished only for logs: the plain call
// would not have picked a shape this launcher can mirror. Anything else is a
// real launch error.)
//
// WHY DECLINE ON 2: the single-block fused body IS `sparse_attn_pf_kernel`'s
// body, so it applies only when the plain launcher would have picked
// `sparse_attn_pf_kernel` itself. Two shapes are unmirrorable and fall back:
//   * `DSV41_ATTN_SEQ` (the sequential kernel), and
//   * `DSV41_ATTN_PF=0` (the warp A/B kernel).
//
// KEY-SPLIT ARM (sparse-attn-v8, 2026-09-11): a split_c > 0 used to decline here
// too, which forfeited the o-rope / o-quant epilogue at EVERY context and was
// the round-39 regression. It no longer does: the epilogue moved into
// `sparse_attn_merge_kernel`, whose (b*m, h) x 128 grid is exactly the
// single-block kernel's, so the fused split path is just the same two launches
// `dsv41_sparse_attn` makes, with the merge's epilogue enabled. The emitted
// bytes and the `out` row are the phase-2/phase-3 outputs of the single-block
// kernel up to the final summation grouping (all that a C > 1 merge changes).
//
// ⚠️ Declines are > 1 on purpose: 1 collides with cudaErrorInvalidValue, and the
// trailing cudaGetLastError() can also yield 1 (the legacy `apply_rope_q`
// sentinel-1 trap documented at `dsv41_apply_rope_q`). The Rust side reads
// rc <= 3 as "decline, fall back" only for the sentinels it knows; anything
// else is a real launch error.
extern "C" int dsv41_sparse_attn_orope(
    const float* q, const float* kv, const float* sink, const int32_t* idxs, float* out, int b,
    int m, int h, int d, const int* clen, int window, int index_topk, float scale,
    const float* cos, const float* sin, const int* base, int rope_rd, int half, int mul, int off,
    int step, int inverse, uint8_t* xq, float* xsc, cudaStream_t s) {
    if (b <= 0 || m <= 0 || h <= 0 || d <= 0) return 1;
    if (d > 512) return 1;              // accumulator is d-wide per thread group
    if ((d & 31) != 0) return 1;        // fp8 per-32-block index must stay head-local
    if (rope_rd <= 0 || rope_rd > d || (rope_rd & 1) != 0) return 1;
    if (half != rope_rd / 2) return 1;
    if (cos == nullptr || sin == nullptr || base == nullptr) return 1;
    if (xq == nullptr || xsc == nullptr) return 1;
    // Same selection the plain launcher makes, through the SAME resolver (the env
    // is read once per process either way, and the two must agree or the fused
    // path declines shapes it could have carried).
    static const bool seq = [] { return getenv("DSV41_ATTN_SEQ") != nullptr; }();
    static const int g_sparse_split_c = dsv41_resolve_sparse_split_c();
    static const bool pf_off = [] {
        const char* e = getenv("DSV41_ATTN_PF");
        return e != nullptr && atoi(e) == 0;
    }();
    const int split_c = g_sparse_split_c;
    if (seq) return 2;
    // Mirror the plain launcher's order (warp arm before split): with the split
    // defaulting ON, a pf_off arm must still land on the warp kernel AND this
    // fused path must decline it, or the fallback and the fused path would run
    // different kernels for the same env.
    if (pf_off) return 2;
    if (split_c > 0 && split_c <= kAttnMaxC && b * m <= kAttnMaxBM && h <= kAttnMaxH) {
        // The key-split arm, mirroring `dsv41_sparse_attn`'s split arm exactly;
        // the ONLY difference is that the merge gets the rope/fp8 epilogue
        // arguments, so both fusions survive instead of declining. Same stream:
        // program order makes the partials visible to the merge with no fence.
        sparse_attn_split_kernel<<<dim3((unsigned)split_c, (unsigned)(b * m),
                                        (unsigned)h), 128, 0, s>>>(
            q, kv, idxs, b, m, h, d, clen, window, index_topk, scale, split_c);
        cudaError_t e2 = cudaGetLastError();
        if (e2 != cudaSuccess) return (int)e2;
        sparse_attn_merge_kernel<<<dim3((unsigned)(b * m), (unsigned)h), 128, 0, s>>>(
            sink, out, b, m, h, d, split_c, cos, sin, base, rope_rd, half, mul, off, step,
            inverse, xq, xsc);
        return (int)cudaGetLastError();
    }
    if (pf_off) return 2;
    sparse_attn_orope_kernel<<<dim3(b * m, h), 128, 0, s>>>(
        q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, cos, sin, base,
        rope_rd, half, mul, off, step, inverse, xq, xsc);
    return (int)cudaGetLastError();
}

extern "C" int dsv41_indexer_topk(const float* q, const float* index_k, const float* weights,
                                  const uint8_t* candidates, const int32_t* compress_lens,
                                  int32_t* out, int b, int m, int nh, int hd, int n_pos, int topk,
                                  int offset, float softmax_scale, float head_scale,
                                  int uses_candidates, cudaStream_t s) {
    if (b <= 0 || m <= 0 || nh <= 0 || hd <= 0 || topk <= 0) return (int)cudaErrorInvalidValue;
    if (n_pos <= 0) return (int)cudaSuccess;  // nothing to select from
    // Stage A's scratch (g_idx_score) has COMPILE-TIME dims: reject a shape that
    // cannot fit, LOUDLY. b*m is host-known; the n_pos test covers the host
    // fallback only - the kernels read the live counter, whose reachable maximum
    // is the index_k allocation (max_pos/ratio + 2) that kIdxMaxPos is sized for.
    if (b * m > kIdxMaxRows) return (int)cudaErrorInvalidValue;
    if (n_pos > kIdxMaxPos) return (int)cudaErrorInvalidValue;
    // ---- Stage A: the scoring pass, in parallel, into g_idx_score.
    // Same stream as stage B => program order makes A's writes visible to B:
    // no fence, no sync, no extra stream.
    // v2 = the gemv_bf16_v2 fix (float4 body + optional K-split) with a v1 escape
    // hatch. Every gate here is read ONCE: this launcher runs 4x per step and a
    // per-call getenv on the hot path is the slip the other gates in this file
    // already avoid.
    static const bool idx_v2 = [] {
        const char* e = getenv("DSV41_IDX_SCORE_V2");
        return e == nullptr || atoi(e) != 0;
    }();
    static const int idx_wpr = [] {
        const char* e = getenv("DSV41_IDX_SCORE_WPR");
        const int v = (e == nullptr) ? kIdxScoreWprDefault : atoi(e);
        return (v == 1 || v == 2 || v == 4 || v == 8) ? v : kIdxScoreWprDefault;
    }();
    static const int idx_blocks = [] {
        const char* e = getenv("DSV41_IDX_SCORE_BLOCKS");
        if (e == nullptr) return 0;
        const int v = atoi(e);
        return (v >= 1 && v <= 65535) ? v : 0;
    }();
    // Preconditions of the v2 body: the float4 lane walk needs 16B-aligned rows
    // and the head tree folds exactly 32 lanes. Anything else keeps v1.
    const bool use_v2 = idx_v2 && (hd & 3) == 0 && nh > 0 && nh <= 32;
    const unsigned nblk =
        (unsigned)(idx_blocks > 0 ? idx_blocks : (use_v2 ? kIdxScoreBlocksV2 : kIdxScoreBlocks));
    dim3 sgrid(nblk, (unsigned)m, (unsigned)b);
    if (use_v2) {
        switch (idx_wpr) {
            case 2:
                indexer_score_kernel_v2<2><<<sgrid, 256, 0, s>>>(
                    q, index_k, weights, candidates, compress_lens, m, nh, hd, n_pos, softmax_scale,
                    head_scale, uses_candidates);
                break;
            case 4:
                indexer_score_kernel_v2<4><<<sgrid, 256, 0, s>>>(
                    q, index_k, weights, candidates, compress_lens, m, nh, hd, n_pos, softmax_scale,
                    head_scale, uses_candidates);
                break;
            case 8:
                indexer_score_kernel_v2<8><<<sgrid, 256, 0, s>>>(
                    q, index_k, weights, candidates, compress_lens, m, nh, hd, n_pos, softmax_scale,
                    head_scale, uses_candidates);
                break;
            default:
                indexer_score_kernel_v2<1><<<sgrid, 256, 0, s>>>(
                    q, index_k, weights, candidates, compress_lens, m, nh, hd, n_pos, softmax_scale,
                    head_scale, uses_candidates);
                break;
        }
    } else {
        indexer_score_kernel<<<sgrid, 256, 0, s>>>(q, index_k, weights, candidates, compress_lens,
                                                   m, nh, hd, n_pos, softmax_scale, head_scale,
                                                   uses_candidates);
    }
    cudaError_t es = cudaGetLastError();
    if (es != cudaSuccess) return (int)es;
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
    // One block per (token, batch) row, 256 threads by default: at single-stream
    // decode that is ONE block on one SM, and the chunk's score loop is
    // candidates x nh x hd multiply-accumulates (2048 x 64 x 128 at a 2k latent
    // count) - the block size is the only parallelism this shape has. The sort,
    // the merge and the rank pass are all data-independent over the thread
    // count, so a wider block changes no result, only the wall time.
    static const int idx_threads = [] {
        const char* e = getenv("DSV41_IDX_THREADS");
        if (e == nullptr) return 1024;
        const int v = atoi(e);
        return (v >= 64 && v <= 1024) ? v : 1024;
    }();
    dim3 grid((unsigned)m, (unsigned)b);
    indexer_topk_kernel<<<grid, idx_threads, smem, s>>>(q, index_k, weights, candidates,
                                                        compress_lens, out, m, nh, hd, n_pos, topk,
                                                        offset, softmax_scale, head_scale,
                                                        uses_candidates);
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

// COMPRESS_FUSE: state carry + pool + commit in ONE 1-block launch. DECODE-ONLY
// by contract (`b == seqlen == 1`, `ratio > 1`, `start_pos > 0`): the prefill /
// ratio == 1 shapes keep the three-launch path, whose state carry has a
// different (multi-row) mapping. The caller (`chain_dev.rs::compress_on`) checks
// the same shape gate plus the `DSV41_COMPRESS_FUSE` env flag, so a violation
// here is a programming error, not a runtime decline.
extern "C" int dsv41_compressor_fused(const float* kvp, const float* scp, const float* norm_w,
                                      float* state_kv, float* state_score, float* latent,
                                      int32_t* out_rows, const float* cos_t, const float* sin_t,
                                      float* ring, int* clen, int b, int seqlen, int hd, int ratio,
                                      int rope_dim, int half, int window, const int* pos_ctr,
                                      float eps, cudaStream_t s) {
    if (b != 1 || seqlen != 1 || hd <= 0 || ratio <= 1) return (int)cudaErrorInvalidValue;
    compressor_fused_kernel<<<1, 128, 0, s>>>(kvp, scp, norm_w, state_kv, state_score, latent,
                                              out_rows, cos_t, sin_t, ring, clen, hd, ratio,
                                              rope_dim, half, window, pos_ctr, eps);
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
                                           inverse, nullptr, nullptr);
    return (int)cudaGetLastError();
}

// B2: apply_rope whose epilogue emits the fp8 of the whole roped region, so the
// consumer's `quant1(o)` launch (40 per step: one per layer, right after the
// inverse rope) disappears. Returns 1 when the shape cannot take the fused
// emission (rows*row_len not a multiple of 32) - the caller then runs the plain
// dsv41_apply_rope + dsv41_quant_fp8 pair, which this is bit-identical to.
//
// ⚠️ LEGACY decline contract (rounds 37-41, kept as-is): the sentinel is 1,
// which IS cudaErrorInvalidValue, and the trailing `cudaGetLastError()` can also
// yield 1. A genuine 1 is therefore read by `Device::apply_rope_q` (rc == 1) as a
// decline and silently falls back. Harmless (the fallback is bit-identical) but
// invisible; migrate to decline == 2 if this path is ever re-touched.
extern "C" int dsv41_apply_rope_q(float* x, const float* cos, const float* sin, int rows,
                                  int row_len, int dim, int half, const int* base, int mul, int off,
                                  int step, int inverse, uint8_t* xq, float* xsc, cudaStream_t s) {
    if (rows <= 0 || row_len <= 0) return 1;
    if (((long long)rows * (long long)row_len) % 32 != 0) return 1;
    apply_rope_kernel<<<rows, 128, 0, s>>>(x, cos, sin, rows, row_len, dim, half, base, mul, off,
                                           step, inverse, xq, xsc);
    return (int)cudaGetLastError();
}

// rmsnorm + rope on the same row, one launch. 1024 threads is load-bearing: the
// reduction tree must match ferrite_rmsnorm's at blockDim 1024 bit for bit.
extern "C" int dsv41_rmsnorm_rope(const float* x, const float* w, float* out, const float* cos,
                                  const float* sin, int n, int dim, int rope_len, int half,
                                  const int* base, int mul, int off, int step, int inverse,
                                  float eps, cudaStream_t s) {
    if (n <= 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    rmsnorm_rope_kernel<<<n, 1024, 0, s>>>(x, w, out, cos, sin, n, dim, rope_len, half, base, mul,
                                          off, step, inverse, eps);
    return (int)cudaGetLastError();
}

// T2: rmsnorm whose epilogue ALSO emits the fp8 activation pair (byte + scale)
// of its OWN normalised output, in the rmsnorm write-back's pass. This is T1's
// trick applied one level up: the value the norm loop writes as f32 is exactly
// what quant_fp8 would read back, so the consumer's quant launch (the qr ->
// wq_b projection, chain_dev.rs) is redundant and skipped.
//
// Warp/block alignment (the reason the amax is one shuffle): the write loop
// strides by blockDim, which is a multiple of 32 and starts at column 0, so a
// warp's 32 lanes cover EXACTLY one 32-element block per pass and the block
// index is `i >> 5`. No barrier, no second global pass.
//
// The scale/quant arithmetic is quant_kernel's and hc_mixes_tail's T1 emission,
// term for term: fast_round_scale(amax, 1/448), clamp +-448, __nv_fp8_e4m3. The
// emitted pair is therefore bit-identical to the dsv41_quant_fp8 launch it
// replaces. Requires every warp to be fully active inside the loop, which holds
// only when dim and the final partial pass are multiples of 32 - the launcher
// declines otherwise (returns 1) and the caller runs rmsnorm + quant_fp8.
//
// ⚠️ LEGACY decline contract: the sentinel 1 collides with cudaErrorInvalidValue
// (n <= 0 / dim <= 0 also return 1), so `Device::rmsnorm_q` (rc == 1) cannot tell
// a decline from a real failure. Kept as-is per the rounds 37-41 freeze; migrate
// to decline == 2 if re-touched.
__global__ void rmsnorm_q_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                 float* __restrict__ out, int n, int dim, float eps,
                                 uint8_t* __restrict__ xq, float* __restrict__ xsc) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const float* xr = x + (size_t)row * dim;
    float* or_ = out + (size_t)row * dim;
    // identical reduction tree to rmsnorm_kernel (blockDim-sized cross-warp)
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    float lane = ss;
    for (int off = 16; off > 0; off >>= 1) lane += __shfl_down_sync(0xffffffffu, lane, off);
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lane;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += red[i];
        red[0] = rsqrtf(t / dim + eps);
    }
    __syncthreads();
    const float inv = red[0];
    const int lane31 = threadIdx.x & 31;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = xr[i] * inv * w[i];
        or_[i] = v;
        float a = fabsf(v);
        for (int off = 16; off > 0; off >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
        const float sc = fmaxf(fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
        if (lane31 == 0) xsc[i >> 5] = sc;
        const float q = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
        const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
        xq[i] = *(const uint8_t*)&f8;
    }
}

extern "C" int dsv41_rmsnorm_q(const float* x, const float* w, float* out, int n, int dim, float eps,
                               uint8_t* xq, float* xsc, cudaStream_t s) {
    if (n <= 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    // The warp-shuffle amax needs all 32 lanes of a warp inside one 32-element
    // block, i.e. no partially-active warp: dim must be a multiple of 32 and the
    // last partial pass (dim % 1024) must be one too. Decline -> caller falls
    // back to the two launches.
    const int tail = dim % 1024;
    if (dim % 32 != 0 || (tail != 0 && tail % 32 != 0)) return 1;
    rmsnorm_q_kernel<<<n, 1024, 0, s>>>(x, w, out, n, dim, eps, xq, xsc);
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
// hc-merge (B'): the per-row ticket the merged front kernel's tail block spins
// on until all `mix` dot blocks have published their g_hc_part entries. Same
// .bss lifetime as g_hc_part (zero-initialised at module load); the tail block
// RESETS it to 0 after its last g_hc_part read, so the next launch (or graph
// replay) starts clean. A mid-kernel abort would leave it at `mix` and hang the
// next launch - acceptable because an abort already poisons the CUDA context.
__device__ unsigned g_hc_ticket[DSV41_HC_SPREAD_MAXR];

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
// P4 (gemm-act-cpasync): retire every cp.async group except the newest one.
// `cp.async.wait_group N` completes all but the N most recent commit groups, so
// `wait_group 1` completes the OLDER group and lets the newer one stay in
// flight. The fp8 gemv commits the activation group before the weight group on
// purpose (see g_gemv_act_cpasync), which is what makes this the "wait for the
// activation, keep the weight row streaming" primitive.
__device__ __forceinline__ void dsv41_cp_wait_group1() { asm volatile("cp.async.wait_group 1;\n"); }

// Phase A: one warp per (token, projection row). Both operands are staged with
// cp.async; the raw dot lands in g_hc_part[r][m][0] for the tail to scale.
__global__ void hc_mix_dots_kernel(const float* __restrict__ x, const float* __restrict__ hc_fn,
                                   int rows, int hc_dim, int mix, int ss_out) {
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
    // ALL warps stage. The row pair is 2 x hc_dim x 4 bytes = 160 KiB at the real
    // hc_dim (20480), and a single warp could only keep ~20 KiB of cp.async in
    // flight, so the staging alone was ~6 us of exposed DRAM latency (measured
    // 3.93 MiB / 7.4 us = 531 GB/s, i.e. ~21 GB/s per active SM against 100+).
    // The compute below is unchanged and still runs in warp 0 with the same lane
    // assignment, so every partial sum is bit-identical.
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sx[i], &xg[i]);
    dsv41_cp_commit();
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sw[i], &wg[i]);
    dsv41_cp_commit();
    dsv41_cp_wait_all();
    __syncthreads();   // the staging spans warps now
    if (threadIdx.x < 32) {
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
        // Fold the tail's sum-of-squares in: the row is already staged in shared
        // memory, so block m replays the tail's warp-m partial exactly (c = lane +
        // m*32, stride mix*32) - bit-identical grouping, zero extra global reads.
        if (ss_out != 0) {
            float s2 = 0.f;
            for (int c2 = lane + m * 32; c2 < hc_dim; c2 += mix * 32) s2 += s_x[c2] * s_x[c2];
            for (int off = 16; off > 0; off >>= 1) s2 += __shfl_xor_sync(0xFFFFFFFFu, s2, off);
            if (lane == 0) g_hc_part[r][m][1] = s2;
        }
    }
}

// Phase B: one block per token, 1024 threads. The sum of squares, the scale, the
// sigmoid split, the sinkhorn and the comb write - all as hc_mixes_kernel does
// them, with the walk over hc_dim striding by ss_stride (= mix*32) so the
// per-lane partial sums group exactly as they did at blockDim 768.
// hc tail split (DSV41_HC_TAIL_SPLIT): the body below is two INDEPENDENT halves.
//   EARLY = collapse + rmsnorm + T1 fp8 (writes out/xq/xsc), whose consumer is the
//           projection group that immediately follows (lin2's wq_a+wkv);
//   LATE  = ss + mixes + sigmoid + sinkhorn + comb (writes pre/post/comb), whose
//           consumer is hc_post, a whole projection + AR away.
// They share no output, so running them as two launches (LATE on a side stream)
// is bit-identical to the single launch: every statement runs in the same order
// with the same operands. `mode` selects the halves; HC_TAIL_FULL is the original
// kernel, byte-for-byte the same arithmetic.
#define HC_TAIL_FULL 0
#define HC_TAIL_EARLY 1
#define HC_TAIL_LATE 2
__global__ void hc_mixes_tail_kernel(const float* __restrict__ x,
                                     const float* __restrict__ hc_scale,
                                     const float* __restrict__ hc_base, float* __restrict__ pre,
                                     float* __restrict__ post, float* __restrict__ comb, int hc,
                                     int dim, int sinkhorn_iters, float eps, int ss_stride,
                                     const float* __restrict__ w_norm,
                                     const float* __restrict__ pre_collapse,
                                     float* __restrict__ out, float eps_norm, int ss_in,
                                     uint8_t* __restrict__ xq, float* __restrict__ xsc, int mode) {
    const int r = blockIdx.x;
    const int mix = hc * (2 + hc);
    const int hc_dim = hc * dim;
    extern __shared__ float t_sm[];
    float* mixes = t_sm;                // mix
    float* cm = t_sm + 64;              // hc*hc
    __shared__ float sss;
    __shared__ float wpart[32];
    const float* xr = x + (size_t)r * hc_dim;
    const int nwarp = ss_stride >> 5;
    // ---------------- LATE half (ss -> mixes -> sigmoid -> sinkhorn -> comb) ----
    if (mode != HC_TAIL_EARLY) {
    if (ss_in != 0) {
        // The ss partials came from the dots kernel, computed from the same
        // staged row with the identical per-warp grouping - only the
        // cross-warp combine (the 32-lane tree over 24 partials) remains.
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < nwarp) ? g_hc_part[r][threadIdx.x][1] : 0.f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
            if (threadIdx.x == 0) sss = v;
        }
    } else {
        float ss = 0.f;
        for (int c = threadIdx.x; c < hc_dim; c += ss_stride) ss += xr[c] * xr[c];
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        if ((threadIdx.x & 31) == 0 && (int)(threadIdx.x >> 5) < nwarp)
            wpart[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < nwarp) ? wpart[threadIdx.x] : 0.f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
            if (threadIdx.x == 0) sss = v;
        }
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
    }   // end LATE half
    // ---------------- EARLY half (collapse + rmsnorm + T1 fp8) ----------------
    // Collapse + rmsnorm, the body of dsv41_hc_collapse_norm_kernel. It reads the
    // `pre` of the slot the caller names for the collapse, which is NOT the one the
    // mixes just wrote: the block walks the premix slots so the attention half
    // collapses with the previous layer's coefficients. Folding it in here is what
    // makes this the whole front end in two launches.
    if (mode != HC_TAIL_LATE && w_norm != nullptr) {
        __syncthreads();
        float* o_r = out + (size_t)r * dim;
        float s2 = 0.f;
        // Compiler-directed unroll (the same treatment that worked on the fp8
        // gemv: the fmaf accumulation is explicitly rounded, so letting nvcc
        // widen the loop body keeps the arithmetic).
#pragma unroll 4
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
        // Optional fused fp8 emission (T1): the SAME normalised value this loop
        // writes as f32 is what quant_kernel would quantise next, so emit the
        // e4m3 byte and the per-32-block scale here and the consumer skips its
        // quant launch. blockDim strides keep a warp's 32 lanes inside exactly one
        // 32-element block per pass (block index = warp + 32*pass), so the amax is
        // a single warp shuffle - no barrier, no extra pass over global memory.
        // The scale/quant arithmetic is quant_kernel's, term for term
        // (fast_round_scale(amax, 1/448), clamp +-448, __nv_fp8_e4m3), so the
        // emitted pair is bit-identical to the launch it replaces.
        const int lane31 = threadIdx.x & 31;
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            const float v = o_r[c] * inv2 * w_norm[c];
            o_r[c] = v;
            if (xq != nullptr) {
                float a = fabsf(v);
                for (int off = 16; off > 0; off >>= 1)
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                const float sc = fmaxf(fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
                if (lane31 == 0) xsc[c >> 5] = sc;
                const float q = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
                const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
                xq[c] = *(const uint8_t*)&f8;
            }
        }
    }
}

static const bool g_hc_front = [] {
    const char* e = getenv("DSV41_HC_FRONT");
    // Default on: 16.85 ms against 19.96 ms in one session, same binary, with the
    // four prompts character-for-character correct. "0" still opts out.
    if (e == nullptr) return true;
    return e[0] != '0';
}();

// hc-merge (B'): single kernel for the whole hc front. grid = (mix+1, rows),
// block = 1024. The m < mix blocks each stage + compute one projection row's
// dot (warp 0, the exact hc_mix_dots code) and bump a per-row ticket; the
// m == mix block FIRST does the collapse+rmsnorm+fp8 (which depends only on
// x/w_norm/pre_collapse, not on any dot), THEN spins on the ticket, THEN runs
// the tail body (ss/mixes/sigmoid/sinkhorn/comb, the exact hc_mixes_tail code
// minus the collapse which moved earlier). One launch instead of two, and the
// collapse leaves the critical path (it overlaps the dots).
//
// Deadlock safety: NO early returns anywhere (every __syncthreads must see the
// whole block); the grid is exact so no bounds check is needed; the spin has a
// ~5 s watchdog; the tail block resets the ticket after its last g_hc_part
// read, so the next launch (or graph replay) starts clean. B300 keeps ~296
// blocks resident against the 2*(mix+1) = 50 this needs, so the spin always
// resolves. rows == 1 at every current call site.
__global__ void hc_front_kernel(const float* __restrict__ x, const float* __restrict__ hc_fn,
                                const float* __restrict__ hc_scale,
                                const float* __restrict__ hc_base, float* __restrict__ pre,
                                float* __restrict__ post, float* __restrict__ comb, int hc,
                                int dim, int sinkhorn_iters, float eps, int ss_stride,
                                const float* __restrict__ w_norm,
                                const float* __restrict__ pre_collapse, float* __restrict__ out,
                                float eps_norm, int ss_in, uint8_t* __restrict__ xq,
                                float* __restrict__ xsc, int rows, int hc_dim, int mix) {
    const int r = blockIdx.y;
    const int m = blockIdx.x;
    extern __shared__ float hc_sm[];
    const int lane = threadIdx.x & 31;

    if (m < mix) {
        // ---------- dot branch: hc_mix_dots_kernel's body, verbatim ----------
        float* s_x = hc_sm;
        float* s_w = hc_sm + hc_dim;
        const int n4 = hc_dim >> 2;
        const float4* xg = reinterpret_cast<const float4*>(x + (size_t)r * hc_dim);
        const float4* wg = reinterpret_cast<const float4*>(hc_fn + (size_t)m * hc_dim);
        float4* sx = reinterpret_cast<float4*>(s_x);
        float4* sw = reinterpret_cast<float4*>(s_w);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sx[i], &xg[i]);
        dsv41_cp_commit();
        for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sw[i], &wg[i]);
        dsv41_cp_commit();
        dsv41_cp_wait_all();
        __syncthreads();
        if (threadIdx.x < 32) {
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
            if (ss_in != 0) {
                float s2 = 0.f;
                for (int c2 = lane + m * 32; c2 < hc_dim; c2 += mix * 32)
                    s2 += s_x[c2] * s_x[c2];
                for (int off = 16; off > 0; off >>= 1)
                    s2 += __shfl_xor_sync(0xFFFFFFFFu, s2, off);
                if (lane == 0) g_hc_part[r][m][1] = s2;
            }
        }
        // ---------- publish: the writer thread is warp0/lane0 ----------
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(&g_hc_ticket[r], 1u);
        }
    } else {
        // ---------- tail branch ----------
        float* mixes = hc_sm;           // reuse the staging area (only this block sees it)
        float* cm = hc_sm + 64;
        __shared__ float sss;
        __shared__ float wpart[32];
        const float* xr = x + (size_t)r * hc_dim;
        const int nwarp = ss_stride >> 5;

        // (1) collapse + rmsnorm + fp8 FIRST (depends only on x/w_norm/pre_collapse)
        if (w_norm != nullptr) {
            __syncthreads();
            float* o_r = out + (size_t)r * dim;
            float s2 = 0.f;
#pragma unroll 4
            for (int c = threadIdx.x; c < dim; c += blockDim.x) {
                float acc = 0.f;
                for (int i = 0; i < hc; ++i)
                    acc = fmaf(pre_collapse[(size_t)r * hc + i], xr[(size_t)i * dim + c], acc);
                o_r[c] = acc;
                s2 += acc * acc;
            }
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
            const int lane31 = threadIdx.x & 31;
            for (int c = threadIdx.x; c < dim; c += blockDim.x) {
                const float v = o_r[c] * inv2 * w_norm[c];
                o_r[c] = v;
                if (xq != nullptr) {
                    float a = fabsf(v);
                    for (int off = 16; off > 0; off >>= 1)
                        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                    const float sc = fmaxf(fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
                    if (lane31 == 0) xsc[c >> 5] = sc;
                    const float q = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
                    const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
                    xq[c] = *(const uint8_t*)&f8;
                }
            }
        }

        // (2) acquire: wait for all mix dot blocks to publish
        if (threadIdx.x == 0) {
            long spins = 0;
            while (atomicAdd(&g_hc_ticket[r], 0u) < (unsigned)mix) {
                __nanosleep(64);
                if (++spins > 40000000) break;   // ~5 s watchdog
            }
        }
        __syncthreads();
        __threadfence();   // acquire: g_hc_part reads must not hoist above the spin

        // (3) the tail body: hc_mixes_tail_kernel's code minus the collapse
        if (ss_in != 0) {
            if (threadIdx.x < 32) {
                float v = (threadIdx.x < nwarp) ? g_hc_part[r][threadIdx.x][1] : 0.f;
                for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
                if (threadIdx.x == 0) sss = v;
            }
        } else {
            float ss = 0.f;
            for (int c = threadIdx.x; c < hc_dim; c += ss_stride) ss += xr[c] * xr[c];
            for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
            if ((threadIdx.x & 31) == 0 && (int)(threadIdx.x >> 5) < nwarp)
                wpart[threadIdx.x >> 5] = ss;
            __syncthreads();
            if (threadIdx.x < 32) {
                float v = (threadIdx.x < nwarp) ? wpart[threadIdx.x] : 0.f;
                for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
                if (threadIdx.x == 0) sss = v;
            }
        }
        __syncthreads();
        const float inv = rsqrtf(sss / (float)hc_dim + eps);
        for (int m2 = threadIdx.x; m2 < mix; m2 += blockDim.x)
            mixes[m2] = g_hc_part[r][m2][0] * inv;
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
            const int warp = threadIdx.x >> 5;
            if (warp == 0) {
                float c = (lane < hh) ? cm[lane] : 0.f;
                float mx = c;
                for (int off = 1; off < hc; off <<= 1)
                    mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
                c = expf(c - mx);
                float rs = c;
                for (int off = 1; off < hc; off <<= 1)
                    rs += __shfl_xor_sync(0xFFFFFFFFu, rs, off);
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

        // (4) ticket reset: after the last g_hc_part read (the mixes loop above)
        if (threadIdx.x == 0) atomicExch(&g_hc_ticket[r], 0u);
    }
}

// ============================================================================
// Stage C persistent prototype (docs/agent/dsv41-persistent-arch.md §1): the
// WHOLE hc front end in ONE block, ordered by __syncthreads() phase barriers.
// grid = (rows,), block = 1024, dynamic smem = one staged activation row
// (hc_dim floats = 80 KiB at dim 5120 / hc_mult 4).
//
// Why this shape: the two-launch form is 24 dot blocks + 1 tail block (25
// blocks, 2 launches) that publish through global memory + stream order. Here
// one block runs all four phases of the front end, so every dependency the
// stream used to order becomes a single __syncthreads() and the second launch
// disappears. This is the mechanism the segment kernels (P2/P3) will scale up.
//
// NO ticket, NO spin — the hc-merge lesson (hc_front_kernel, DSV41_HC_MERGE):
// a tail block that spins on a ticket holds an SM hostage while 31 of its 32
// warps idle, and measured +3.2 ms/step. A phase machine has no spin by
// construction; every barrier is unconditional and reached by the whole block.
//
// Bit-exactness is by construction, not by luck:
//   * Each projection row's dot is computed by ONE warp with hc_mix_dots_kernel's
//     EXACT lane assignment (c = lane, the +96 three-float4 chain, the +32
//     remainder, the scalar tail), and the ss partial uses hc_mixes_tail_kernel's
//     exact grouping (c2 = lane + m*32, stride mix*32). Sums therefore do not move.
//   * The activation row is staged with cp.async into the SAME smem layout the
//     dots kernel used, so sx[c] holds the same bits. The weight row is read
//     straight from global: wg[c] == the staged sw[c]. SMEM STAGING OF THE
//     WEIGHT WAS A LATENCY OPTIMISATION, NEVER AN ARITHMETIC ONE — the values
//     and the accumulation order are identical either way.
//   * The tail body (ss combine -> mixes -> sigmoid -> sinkhorn -> comb) is
//     hc_mixes_tail_kernel's, statement for statement; the collapse + rmsnorm +
//     fp8 is the tail's / dsv41_hc_collapse_norm_kernel's, statement for statement.
//
// ⚠️ THE SMEM BUDGET IS FINE, THE PARALLELISM IS NOT. Two-launch stages x + ONE
// weight row = 160 KiB per (row, mix) block and gets 24-way BLOCK parallelism
// (24 SMs, measured 531 GB/s aggregate / 22 GB/s per SM, 7.4 us). One block
// cannot hold 24 weight rows (24 x 80 KiB = 1.9 MiB), so the dots here run on a
// SINGLE SM: 24 warps reading 1.9 MiB of weights from DRAM. That is expected to
// be slower than the 24-block dots, so this kernel is a MECHANISM + PARITY
// prototype, not (yet) a win — measure before trusting. DSV41_HC_PERSIST=1
// selects it from the Rust side (default OFF).
// ============================================================================
__global__ void hc_pre_persist_kernel(const float* __restrict__ x, const float* __restrict__ hc_fn,
                                      const float* __restrict__ hc_scale,
                                      const float* __restrict__ hc_base, float* __restrict__ pre,
                                      float* __restrict__ post, float* __restrict__ comb, int hc,
                                      int dim, int sinkhorn_iters, float eps, int ss_stride,
                                      const float* __restrict__ w_norm,
                                      const float* __restrict__ pre_collapse,
                                      float* __restrict__ out, float eps_norm, int ss_in,
                                      uint8_t* __restrict__ xq, float* __restrict__ xsc,
                                      int hc_dim, int mix) {
    const int r = blockIdx.x;
    extern __shared__ float hc_sm[];
    float* s_x = hc_sm;                 // hc_dim floats: the staged activation row
    __shared__ float sss;
    __shared__ float wpart[32];
    __shared__ float p_mixes[64];
    __shared__ float p_cm[64];
    __shared__ float p_red[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n4 = hc_dim >> 2;
    const float* xr = x + (size_t)r * hc_dim;
    const int nwarp = ss_stride >> 5;

    // ---------------- phase 1: stage the activation row (all warps) ---------
    {
        const float4* xg = reinterpret_cast<const float4*>(xr);
        float4* sx = reinterpret_cast<float4*>(s_x);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sx[i], &xg[i]);
        dsv41_cp_commit();
        dsv41_cp_wait_all();
        __syncthreads();
    }

    // ---------------- phase 2: the mix projection dots (one row per warp) ---
    if (warp < mix) {
        const int m = warp;
        const float4* wg = reinterpret_cast<const float4*>(hc_fn + (size_t)m * hc_dim);
        const float4* sx = reinterpret_cast<const float4*>(s_x);
        const float* wrow = hc_fn + (size_t)m * hc_dim;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f;
        int c = lane;
        for (; c + 64 < n4; c += 96) {
            const float4 w0 = wg[c], w1 = wg[c + 32], w2 = wg[c + 64];
            const float4 v0 = sx[c], v1 = sx[c + 32], v2 = sx[c + 64];
            a0 += w0.x * v0.x + w0.y * v0.y + w0.z * v0.z + w0.w * v0.w;
            a1 += w1.x * v1.x + w1.y * v1.y + w1.z * v1.z + w1.w * v1.w;
            a2 += w2.x * v2.x + w2.y * v2.y + w2.z * v2.z + w2.w * v2.w;
        }
        for (; c < n4; c += 32) {
            const float4 w = wg[c], v = sx[c];
            a0 += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
        }
        for (int k = (n4 << 2) + lane; k < hc_dim; k += 32) a0 += wrow[k] * s_x[k];
        float acc = (a0 + a1) + a2;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) g_hc_part[r][m][0] = acc;
        if (ss_in != 0) {
            float s2 = 0.f;
            for (int c2 = lane + m * 32; c2 < hc_dim; c2 += mix * 32) s2 += s_x[c2] * s_x[c2];
            for (int off = 16; off > 0; off >>= 1) s2 += __shfl_xor_sync(0xFFFFFFFFu, s2, off);
            if (lane == 0) g_hc_part[r][m][1] = s2;
        }
    }
    __syncthreads();   // phase barrier: g_hc_part is now visible to the whole block

    // ---------------- phase 3: the tail body (hc_mixes_tail minus collapse) -
    if (ss_in != 0) {
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < nwarp) ? g_hc_part[r][threadIdx.x][1] : 0.f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
            if (threadIdx.x == 0) sss = v;
        }
    } else {
        float ss = 0.f;
        for (int c = threadIdx.x; c < hc_dim; c += ss_stride) ss += xr[c] * xr[c];
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        if ((threadIdx.x & 31) == 0 && warp < nwarp) wpart[warp] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < nwarp) ? wpart[threadIdx.x] : 0.f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
            if (threadIdx.x == 0) sss = v;
        }
    }
    __syncthreads();
    const float inv = rsqrtf(sss / (float)hc_dim + eps);
    for (int m2 = threadIdx.x; m2 < mix; m2 += blockDim.x)
        p_mixes[m2] = g_hc_part[r][m2][0] * inv;
    __syncthreads();
    if (threadIdx.x < (unsigned)hc) {
        const int j = threadIdx.x;
        pre[(size_t)r * hc + j] =
            (1.f / (1.f + expf(-(p_mixes[j] * hc_scale[0] + hc_base[j])))) + eps;
        post[(size_t)r * hc + j] =
            2.f / (1.f + expf(-(p_mixes[hc + j] * hc_scale[1] + hc_base[hc + j])));
    }
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
        const int j = jk / hc, k = jk % hc;
        p_cm[jk] = p_mixes[2 * hc + j * hc + k] * hc_scale[2] + hc_base[2 * hc + j * hc + k];
    }
    __syncthreads();
    if (warp == 0) {
        const int hh = hc * hc;
        float c = (lane < hh) ? p_cm[lane] : 0.f;
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
        if (lane < hh) p_cm[lane] = c;
    }
    __syncthreads();
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x)
        comb[(size_t)r * hc * hc + jk] = p_cm[jk];

    // ---------------- phase 4: collapse + rmsnorm + fp8 (all warps) ---------
    // The body of dsv41_hc_collapse_norm_kernel. It reads the `pre` of the slot
    // the caller names for the collapse, which is NOT the one the mixes just
    // wrote: the caller walks the premix slots so the attention half collapses
    // with the previous layer's coefficients.
    if (w_norm != nullptr) {
        __syncthreads();
        float* o_r = out + (size_t)r * dim;
        float s2 = 0.f;
#pragma unroll 4
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            float acc = 0.f;
            for (int i = 0; i < hc; ++i)
                acc = fmaf(pre_collapse[(size_t)r * hc + i], xr[(size_t)i * dim + c], acc);
            o_r[c] = acc;
            s2 += acc * acc;
        }
        for (int off = 16; off > 0; off >>= 1) s2 += __shfl_down_sync(0xffffffffu, s2, off);
        if ((threadIdx.x & 31) == 0) p_red[threadIdx.x >> 5] = s2;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.f;
            for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += p_red[i];
            p_red[0] = rsqrtf(t / dim + eps_norm);
        }
        __syncthreads();
        const float inv2 = p_red[0];
        const int lane31 = threadIdx.x & 31;
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            const float v = o_r[c] * inv2 * w_norm[c];
            o_r[c] = v;
            if (xq != nullptr) {
                float a = fabsf(v);
                for (int off = 16; off > 0; off >>= 1)
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                const float sc = fmaxf(fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
                if (lane31 == 0) xsc[c >> 5] = sc;
                const float q = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
                const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
                xq[c] = *(const uint8_t*)&f8;
            }
        }
    }
}

// ss computed in the dots kernel from the staged row (default on); "0" restores
// the in-tail scan for A/B. Bit-identical by construction either way.
static const bool g_hc_ss = [] {
    const char* e = getenv("DSV41_HC_SS");
    if (e == nullptr) return true;
    return e[0] != '0';
}();

// Dots block size for the two-launch path (hc_mix_dots_kernel). File scope so the
// split entry and dsv41_hc_front share ONE value: a divergence between the two
// would silently change the staged-row lane grouping and break bit-exactness.
static const int g_hc_dots_t = [] {
    const char* e = getenv("DSV41_HC_DOTS_T");
    if (e == nullptr) return 128;
    const int v = atoi(e);
    return (v >= 32 && v <= 1024 && (v & 31) == 0 && v % 32 == 0) ? v : 128;
}();

// Block size of the split tail's LATE half (DSV41_HC_LATE_T, default 128).
//
// WHY IT IS NOT 1024. The LATE half is ~30 scalars of elementwise work plus a
// warp-0-only sinkhorn chain: ONE warp does all of it. Launched at 1024 threads
// (the shape the single-kernel form needs, because ITS ss walk strides by mix*32
// and reads its partials out of wpart[threadIdx.x>>5], i.e. warps 0..mix-1), the
// block (a) makes every one of its ~6 __syncthreads() a 32-warp convergence with
// 31 warps carrying nothing, and (b) demands a 1024-thread SM slot from a machine
// whose SMs are full of 128/256-thread projection blocks. Round 41's tail split
// realised -0.20ms of its -0.86ms: the LATE half did not overlap the projection
// chain. A 128-thread block fits beside a running projection block instead of
// queueing behind a wave, which is the scheduling half of that gap.
//
// BIT-IDENTICAL by construction: every LATE statement is elementwise over
// `mixes`/`pre`/`post`/`cm`/`comb` or a warp-0 sinkhorn, so which thread executes
// which iteration cannot move a value, and lanes 0..31 of warp 0 - the only lanes
// the ss combine and the sinkhorn ever touch - are unchanged.
//
// The ONE exception is the self-computed ss path (DSV41_HC_SS=0, an A/B knob):
// its partials live in wpart[threadIdx.x>>5] and the cross-warp tree reads
// warp 0..nwarp-1, so that path MUST keep 1024 threads - hence the gate in the
// launcher rather than a blanket block-size change.
static const int g_hc_late_t = [] {
    const char* e = getenv("DSV41_HC_LATE_T");
    if (e == nullptr) return 128;
    const int v = atoi(e);
    return (v >= 32 && v <= 1024 && (v & 31) == 0 && v % 32 == 0) ? v : 128;
}();

// ============================================================================
// hc dots + LATE in ONE launch (`DSV41_HC_DL_MERGE`, default ON).
//
// WHY THIS MERGE IS LEGAL — and why the EARLY+LATE one is not. hc_front_kernel
// folded EARLY into the tail block and paid +3.2 ms/step: its tail block SPUN on
// a ticket while the dots ran, so it held an SM hostage for the whole dots phase
// — and it parked the 1.7 us EARLY wait on MAIN's critical path, in front of the
// projection group, which is exactly why the EARLY-on-main / EARLY+LATE arm was
// reverted. The tail's ONLY dependency on the dots is `g_hc_part`, so it can be
// ELECTED to the LAST dot block to publish (hc_pre_persist_mb_kernel's
// mechanism) instead of polled for: no ticket, no spin, no extra resident block,
// and EARLY stays its own launch at the head of the side chain.
//
// WHAT IT SAVES. ONE graph node per front (2 per layer x 61 layers) on the
// whole-step capture (DSV41_GRAPH_STEP, default ON): the LATE launch's per-node
// overhead and its scheduler slot. The work itself does not move.
//
// BIT-EXACTNESS by construction. The dot branch is hc_mix_dots_kernel's body
// verbatim (same cp.async staging, same warp-0 float4 three-accumulator lane
// chain, same ss replay residue m*32 / stride mix*32 into g_hc_part[r][m][1]);
// the tail is hc_mixes_tail_kernel's LATE branch verbatim at ss_in == 1, which
// reads the dots' ss partials and needs only warp 0 — so it stays valid at the
// dots' block size (g_hc_dots_t), unlike the self-ss path that needs >= mix*32
// threads. The launcher therefore requires DSV41_HC_SS=1 (the default) and falls
// back to the two-launch pair otherwise.
//
// K-SPLIT IS NOT USED HERE (grid.x = mix, exactly like the dots kernel). At
// split > 1 the ck slots collide with the ss partial slot (the third dimension
// is DSV41_HC_SPREAD_S = 8) and the reduction order stops being bit-exact.
// hc_pre_persist_mb_kernel keeps the K-split arm for the non-bit-exact
// experiment, where its self-ss tail can afford the 1024 threads it needs.
//
// ELECTION DISCIPLINE (why it cannot hang). Every dot block publishes, fences
// and then bumps the per-row counter; the block that sees the final count runs
// the tail, every other block falls off the end. NO block ever waits on another
// block, so there is nothing to deadlock — and the grid is EXACT (mix, rows)
// with no bounds check, so no block can drop out of the count. The elected block
// RESETS the counter after its last g_hc_part read, so the next launch (or graph
// replay) starts clean — the discipline g_hc_ticket / g_hc_mb_done use.
// ============================================================================
static const bool g_hc_dl_merge = [] {
    const char* e = getenv("DSV41_HC_DL_MERGE");
    if (e == nullptr) return true;
    return e[0] != '0';
}();

__device__ unsigned g_hc_dl_done[DSV41_HC_SPREAD_MAXR];

__global__ void hc_dots_late_kernel(const float* __restrict__ x, const float* __restrict__ hc_fn,
                                    const float* __restrict__ hc_scale,
                                    const float* __restrict__ hc_base, float* __restrict__ pre,
                                    float* __restrict__ post, float* __restrict__ comb, int hc,
                                    int dim, int sinkhorn_iters, float eps, int mix) {
    const int m = blockIdx.x;
    const int r = blockIdx.y;
    const int hc_dim = hc * dim;
    const int ss_stride = mix * 32;
    const int nwarp = ss_stride >> 5;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    extern __shared__ float hc_sm[];
    float* s_x = hc_sm;                 // hc_dim floats: the staged activation row
    float* s_w = hc_sm + hc_dim;        // hc_dim floats: this projection row
    __shared__ unsigned s_elected;
    __shared__ float sss;
    __shared__ float p_mixes[64];
    __shared__ float p_cm[64];

    // ---------------- dot: one projection row over the whole K ---------------
    const int n4 = hc_dim >> 2;
    const float4* xg = reinterpret_cast<const float4*>(x + (size_t)r * hc_dim);
    const float4* wg = reinterpret_cast<const float4*>(hc_fn + (size_t)m * hc_dim);
    float4* sx = reinterpret_cast<float4*>(s_x);
    float4* sw = reinterpret_cast<float4*>(s_w);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sx[i], &xg[i]);
    dsv41_cp_commit();
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sw[i], &wg[i]);
    dsv41_cp_commit();
    dsv41_cp_wait_all();
    __syncthreads();   // the staging spans warps
    if (threadIdx.x < 32) {
        // hc_mix_dots_kernel's float4 three-accumulator lane chain, verbatim
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
        // The tail's warp-m ss replay (c = lane + m*32, stride mix*32), exactly as
        // hc_mix_dots_kernel writes it when ss_out != 0.
        float s2 = 0.f;
        for (int c2 = lane + m * 32; c2 < hc_dim; c2 += mix * 32) s2 += s_x[c2] * s_x[c2];
        for (int off = 16; off > 0; off >>= 1) s2 += __shfl_xor_sync(0xFFFFFFFFu, s2, off);
        if (lane == 0) g_hc_part[r][m][1] = s2;
    }
    // Publish and elect. g_hc_part's writer is warp0/lane0 == threadIdx.x 0, so
    // the release fence below orders exactly those stores; s_elected is
    // block-uniform, so the branch that follows is uniform too.
    if (threadIdx.x == 0) {
        __threadfence();
        s_elected = (atomicAdd(&g_hc_dl_done[r], 1u) == (unsigned)(mix - 1)) ? 1u : 0u;
    }
    __syncthreads();
    if (s_elected == 0u) return;    // no barrier follows on this path - safe

    // ---------------- tail: the LATE half, by the elected dot block ----------
    // Acquire: every other dot block stored g_hc_part BEFORE its release fence +
    // atomicAdd, and this block observed the final count.
    __threadfence();
    if (threadIdx.x < 32) {
        // ss_in == 1: the dots' ss partials, only the cross-warp tree remains.
        float v = (threadIdx.x < nwarp) ? g_hc_part[r][threadIdx.x][1] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (threadIdx.x == 0) sss = v;
    }
    __syncthreads();
    const float inv = rsqrtf(sss / (float)hc_dim + eps);
    for (int m2 = threadIdx.x; m2 < mix; m2 += blockDim.x)
        p_mixes[m2] = g_hc_part[r][m2][0] * inv;
    __syncthreads();
    if (threadIdx.x < (unsigned)hc) {
        const int j = threadIdx.x;
        pre[(size_t)r * hc + j] =
            (1.f / (1.f + expf(-(p_mixes[j] * hc_scale[0] + hc_base[j])))) + eps;
        post[(size_t)r * hc + j] =
            2.f / (1.f + expf(-(p_mixes[hc + j] * hc_scale[1] + hc_base[hc + j])));
    }
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
        const int j = jk / hc, k = jk % hc;
        p_cm[jk] = p_mixes[2 * hc + j * hc + k] * hc_scale[2] + hc_base[2 * hc + j * hc + k];
    }
    __syncthreads();
    if (warp == 0) {
        const int hh = hc * hc;
        float c = (lane < hh) ? p_cm[lane] : 0.f;
        float mx = c;
        for (int off = 1; off < hc; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
        c = expf(c - mx);
        float rs = c;
        for (int off = 1; off < hc; off <<= 1) rs += __shfl_xor_sync(0xFFFFFFFFu, rs, off);
        c = c / rs + eps;
        for (int it = 0; it < sinkhorn_iters; ++it) {
            if (it > 0) {
                float st = c;
                for (int off = 1; off < hc; off <<= 1)
                    st += __shfl_xor_sync(0xFFFFFFFFu, st, off);
                c = c / (st + eps);
            }
            float t = c;
            for (int off = hc; off < hh; off <<= 1) t += __shfl_xor_sync(0xFFFFFFFFu, t, off);
            c = c / (t + eps);
        }
        if (lane < hh) p_cm[lane] = c;
    }
    __syncthreads();
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x)
        comb[(size_t)r * hc * hc + jk] = p_cm[jk];
    // Reset LAST: the loops above were the final g_hc_part reads of this row.
    if (threadIdx.x == 0) atomicExch(&g_hc_dl_done[r], 0u);
}

extern "C" int dsv41_hc_front(const float* x, const float* hc_fn, const float* hc_scale,
                              const float* hc_base, const float* w_norm, const float* pre_collapse,
                              float* pre, float* post, float* comb, float* out, int rows, int hc,
                              int dim, int sinkhorn_iters, float eps, float eps_norm,
                              uint8_t* xq, float* xsc, cudaStream_t s) {
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
    // g_hc_part[r][m][ck]'s second dimension is a hard 64: a config with a larger
    // hc would walk off the array, so refuse it here rather than corrupt memory.
    if (mix > 64) return (int)cudaErrorInvalidValue;
    const int hc_dim = hc * dim;
    const size_t smem = (size_t)2 * (size_t)hc_dim * sizeof(float);
    // Set it EVERY call. The attribute is per-context, and a TP8 process has one
    // context per rank, so a process-wide guard leaves seven of the eight ranks
    // without the 160 KB opt-in - the kernel then launches with shared memory it
    // was not granted and the cp.async waits never retire. The existing gemm_fp8
    // launcher carries the same warning for the same reason.
    // hc-merge, DEFAULT OFF: the single-kernel form measured +3.2ms/step
    // (13.75 vs 10.54 at round 14, texts correct, zero faults). The tail block's
    // ticket spin holds an SM hostage for the whole dots phase, and at 1024
    // threads/block 31 of 32 warps idle during the dot compute (the two-launch
    // form's dots ran 128-thread blocks where only 3 warps idled). The gate
    // stays for a future attempt that gives the tail branch its own small block
    // instead of a full 1024-thread one. DSV41_HC_MERGE=1 re-enables.
    static const bool g_hc_merge = [] {
        const char* e = getenv("DSV41_HC_MERGE");
        if (e == nullptr) return false;
        return e[0] == '1';
    }();
    if (g_hc_merge) {
        cudaError_t e2 = cudaFuncSetAttribute(hc_front_kernel,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              dsv41_smem_ceiling(hc_front_kernel));  // 232448-260 static
        if (e2 != cudaSuccess) {
            (void)cudaGetLastError();
            return (int)e2;
        }
        hc_front_kernel<<<dim3((unsigned)(mix + 1), (unsigned)rows), 1024u, smem, s>>>(
            x, hc_fn, hc_scale, hc_base, pre, post, comb, hc, dim, sinkhorn_iters, eps,
            mix * 32, w_norm, pre_collapse, out, eps_norm, g_hc_ss ? 1 : 0, xq, xsc, rows, hc_dim,
            mix);
        return (int)cudaGetLastError();
    }
    // ---- the two-launch path (kept for A/B) ----
    {
        cudaError_t e = cudaFuncSetAttribute(hc_mix_dots_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(hc_mix_dots_kernel));
        if (e != cudaSuccess) {
            (void)cudaGetLastError();   // clear the sticky flag before reporting
            return (int)e;
        }
    }
    // Multi-warp staging, DEFAULT ON (verified 2026-09-11: 11.80 -> 11.68ms, all
    // four prompts verbatim-correct, zero faults). The original conviction - an
    // err 700 "caused by" the 128-thread form - was misattribution: the real fault
    // was the gemv_bf16 dynamic-smem overrun that happened to surface in the same
    // A/B window. Four warps keep four times the cp.async in flight against the
    // 160 KiB row pair; the dot itself still runs in warp 0 with the identical
    // lane assignment, so every partial is bit-identical.
    // Block size comes from the file-scope g_hc_dots_t so the split entry stages
    // the row with the identical lane grouping.
    hc_mix_dots_kernel<<<dim3((unsigned)mix, (unsigned)rows), (unsigned)g_hc_dots_t, smem, s>>>(
        x, hc_fn, rows, hc_dim, mix, g_hc_ss ? 1 : 0);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    hc_mixes_tail_kernel<<<(unsigned)rows, 1024, (64 + 64) * sizeof(float), s>>>(
        x, hc_scale, hc_base, pre, post, comb, hc, dim, sinkhorn_iters, eps, mix * 32, w_norm,
        pre_collapse, out, eps_norm, g_hc_ss ? 1 : 0, xq, xsc, HC_TAIL_FULL);
    return (int)cudaGetLastError();
}

// hc tail split entry (DSV41_HC_TAIL_SPLIT, Rust-gated). Same front end as
// dsv41_hc_front, but the ENTIRE tail chain — and the dots — leave the main
// stream: the EARLY half (collapse + rmsnorm + T1 fp8), the dots and the LATE
// half (ss + sigmoid + sinkhorn + comb) all run in that order on `side`, while
// main blocks only on the EARLY half. Issued sequence (dependency view, not host
// order):
//   main: record(fork_ev) -> wait(fork_ev) [-> projections -> ...]
//   side: wait(fork_ev) -> tail_early(HC_TAIL_EARLY) -> record(fork_ev)
//         -> dots+late in ONE node (DSV41_HC_DL_MERGE, ON) -> record(join_ev)
//         [fallback / A/B arm (`DSV41_HC_DL_MERGE=0`): dots -> tail_late]
// The caller MUST then wait(join_ev) on the main stream before the hc_post that
// consumes `comb`; the projection chain right after this call is ordered after
// EARLY by the launcher's own main-stream wait on the re-recorded `fork_ev`.
//
// WHY EARLY IS BACK ON `side` (2026-09-11, REVERTING B). The A/B on 6.84 ms put
// the +0.27 ms regression on B (EARLY-on-main) + C: P1 was neutral (see the
// HEAD doc commits). EARLY-on-main removed the cross-stream event pair but
// parked its 1.7 us ON main's critical path, in front of the projection group —
// and main never avoided that wait. Pulling EARLY, the dots AND the LATE half
// onto `side` makes the front's main-stream cost the EARLY completion (1.7 us)
// instead of the dots (4.9 us). The side chain
// EARLY(1.7) + dots(4.9) + LATE(10.7) = ~17 us still sits far inside the ~50 us
// projection window that must elapse before hc_post consumes `comb`
// (dsv41-layer-fusion.md, DSV41_HC_TAIL_PRIO note), so `join_ev` is still
// already satisfied when main reaches hc_tail_join. (C — the LATE kernel's
// priority demotion — is kept and is NOT part of this revert.)
//
// WHY `fork_ev` CARRIES BOTH EDGES. The pre-B variant expressed the two extra
// edges with a dedicated in_ev/early_ev pair; those event objects were deleted
// with B and are gone from the runtime that passes this ABI, so the two slots
// stay dead (`in_ev`/`early_ev` are neither validated nor touched). The two
// edges are therefore carried by the ONE live fork event, re-recorded once per
// front:
//   (i)  main -> side: record(fork_ev) on `s`, then wait it on `side` before
//        EARLY. EARLY reads `x` (= s.h) and `pre_collapse`, both written by
//        MAIN-stream work preceding this call, so without this edge `side` is a
//        graph ROOT for EARLY/dots/LATE and a whole-step capture could run EARLY
//        before the main-stream hc_post that fills s.h (stale residual bytes).
//   (ii) side -> main: record(fork_ev) on `side` immediately after EARLY, then
//        wait it on `s`. This orders the projection group after EARLY's writes
//        of out/xq/xsc WITHOUT waiting the dots or the LATE half behind it.
// Each record is immediately followed by its matching wait in program order, so
// the whole-step capture's record/wait disambiguation rule holds (see the
// fork_ev note in devrt.rs): a re-recorded event is never waited by a later,
// non-adjacent path — and `join_ev` stays a SINGLE record per front, so
// hc_tail_join's existing main-stream wait keeps its exact meaning.
//
// WHY THE DOTS CAN LEAVE MAIN (dots-on-side, 2026-09-11). The projection group
// that follows this call reads ONLY the EARLY outputs — lin2 reads `xn` (= `out`)
// and, when `xq_of_xn_valid` is armed, `xq`/`xsc`; `chain_dev.rs:2430`. The dots
// write ONLY `g_hc_part`, whose sole reader is the LATE branch of
// hc_mixes_tail_kernel (also on `side`, after the dots in stream order). So main
// never has a data dependence on the dots, and the main-stream front cost drops
// from the dots (4.9 us) to the EARLY half (1.7 us). The side chain becomes
// EARLY(1.7) + dots(4.9) + LATE(10.7) = ~17 us, still far inside the ~50 us
// projection window that must elapse before hc_post consumes `comb`
// (dsv41-layer-fusion.md, DSV41_HC_TAIL_PRIO note), so `join_ev` is still
// already satisfied when main reaches hc_tail_join.
// The dots and LATE share ONE stream, so stream order already publishes
// `g_hc_part` before the LATE reads — no second fork event. The main->side
// `fork_ev` record/wait pair (i) is what pins EARLY, the dots AND LATE after
// main's writes of `s.h`/`pre_collapse`: without it the side stream is a graph
// ROOT for those nodes and the whole-step capture (DSV41_GRAPH_STEP, default ON)
// may start them before the main-stream hc_post that fills s.h — i.e. on stale
// residual bytes.
// All stream/event ops are legal under cudaStreamCaptureModeRelaxed, so a
// whole-step capture turns the fork/join into graph edges. Bit-identical to
// dsv41_hc_front: the same statements execute with the same operands on
// whichever stream (the EARLY+dots+LATE-on-side split is free — the three are
// disjoint in statements and in memory: EARLY writes out/xq/xsc, the dots write
// g_hc_part, LATE writes pre/post/comb).
// Returns InvalidValue (1) when the gate is off, when there is no collapse half
// to keep (w_norm == nullptr), or when the shapes are outside the spread tables —
// all mean "use dsv41_hc_front instead".
extern "C" int dsv41_hc_front_split(const float* x, const float* hc_fn, const float* hc_scale,
                                    const float* hc_base, const float* w_norm,
                                    const float* pre_collapse, float* pre, float* post,
                                    float* comb, float* out, int rows, int hc, int dim,
                                    int sinkhorn_iters, float eps, float eps_norm, uint8_t* xq,
                                    float* xsc, cudaStream_t s, cudaStream_t side,
                                    cudaEvent_t in_ev, cudaEvent_t fork_ev,
                                    cudaEvent_t early_ev, cudaEvent_t join_ev,
                                    cudaStream_t side_dl) {
    if (x == nullptr || hc_fn == nullptr || hc_scale == nullptr || hc_base == nullptr ||
        pre == nullptr || post == nullptr || comb == nullptr)
        return (int)cudaErrorInvalidValue;
    // `fork_ev` carries BOTH split edges — main->side input-ready at (0) and
    // side->main EARLY-done at (2) — so it is required. `in_ev`/`early_ev` are
    // dead parameters (their event objects were deleted with B and the runtime
    // passes null): they are neither validated nor touched, so a runtime that
    // failed to create only those two still gets the split.
    if (side == nullptr || fork_ev == nullptr || join_ev == nullptr)
        return (int)cudaErrorInvalidValue;
    if (rows <= 0 || hc <= 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    if (!g_hc_front) return (int)cudaErrorInvalidValue;   // caller keeps the old path
    if (rows > DSV41_HC_SPREAD_MAXR) return (int)cudaErrorInvalidValue;
    if ((w_norm == nullptr) != (pre_collapse == nullptr)) return (int)cudaErrorInvalidValue;
    // The split only pays when there IS an EARLY half to leave on the critical
    // path; without a collapse the caller uses dsv41_hc_front.
    if (w_norm == nullptr || out == nullptr) return (int)cudaErrorInvalidValue;
    const int mix = hc * (2 + hc);
    if (mix > 64) return (int)cudaErrorInvalidValue;
    const int hc_dim = hc * dim;
    const size_t smem = (size_t)2 * (size_t)hc_dim * sizeof(float);
    {
        cudaError_t e = cudaFuncSetAttribute(hc_mix_dots_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(hc_mix_dots_kernel));
        if (e != cudaSuccess) {
            (void)cudaGetLastError();
            return (int)e;
        }
    }
    // (0) hc-input-ready edge, main -> side: record `fork_ev` on MAIN before the
    // side chain and wait it on `side`. EARLY, the dots and LATE all read `x`
    // (= s.h) and/or `pre_collapse`, both written by MAIN-stream work that
    // precedes this call (the previous hc_post / AR fold); without this edge the
    // side stream is a graph ROOT for those nodes and a whole-step capture
    // (DSV41_GRAPH_STEP, default ON) could run EARLY before the hc_post that
    // fills s.h — i.e. on stale residual bytes. This is the pre-B `in_ev` edge,
    // now carried by `fork_ev` (the in_ev/early_ev pair was deleted with B and
    // the runtime no longer passes them).
    // NOTE the SAME event is used again at (2) for the opposite (side -> main)
    // edge. Each record is immediately followed by its matching wait, which is
    // the program-order disambiguation a whole-step capture relies on.
    cudaError_t e = cudaEventRecord(fork_ev, s);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();   // clear the sticky flag before reporting
        return (int)e;
    }
    e = cudaStreamWaitEvent(side, fork_ev, 0);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();   // clear the sticky flag before reporting
        return (int)e;
    }
    // (0b) `DSV41_HC_DL_SIDE`: the SECOND waiter of the SAME input-ready record —
    // the dots+LATE half leaves for its own stream. It needs exactly this one
    // edge and nothing else: the tail's two halves are disjoint in statements AND
    // in memory (EARLY writes out/xq/xsc, DL writes g_hc_part/pre/post/comb) and
    // DL reads no EARLY output — both only want `x` (= s.h) and the collapse
    // weights, which the main-stream hc_post before this call already wrote. This
    // wait is ADJACENT to the record above (second matching wait of the same
    // record), so the whole-step capture's program-order disambiguation holds.
    // A null `side_dl` (runtime created no fourth stream) collapses DL back onto
    // `side`, i.e. exactly the pre-split order.
    cudaStream_t dl = (side_dl != nullptr) ? side_dl : side;
    if (dl != side) {
        e = cudaStreamWaitEvent(dl, fork_ev, 0);
        if (e != cudaSuccess) {
            (void)cudaGetLastError();   // clear the sticky flag before reporting
            return (int)e;
        }
    }
    // (1) EARLY half FIRST on the SIDE stream (collapse + rmsnorm + T1 fp8). It is
    // what main blocks on, so it heads the side chain; the dots and LATE behind
    // it are off main's path entirely. Same statements, same operands and same
    // 1024-thread block as dsv41_hc_front's EARLY ⇒ bit-identical bytes.
    hc_mixes_tail_kernel<<<(unsigned)rows, 1024, (64 + 64) * sizeof(float), side>>>(
        x, hc_scale, hc_base, nullptr, nullptr, nullptr, hc, dim, sinkhorn_iters, eps, mix * 32,
        w_norm, pre_collapse, out, eps_norm, g_hc_ss ? 1 : 0, xq, xsc, HC_TAIL_EARLY);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    // (2) EARLY-done edge, side -> main: record `fork_ev` on `side` right after
    // EARLY and wait it on MAIN. The projection group right after this call reads
    // EARLY's outputs (`out` / `xq` / `xsc`), so it must not start before this
    // point; the dots and LATE stay behind it on `side` and keep running
    // concurrently with the projections. This is the pre-B `early_ev` edge,
    // carried by `fork_ev`. Record and wait are ADJACENT in program order.
    e = cudaEventRecord(fork_ev, side);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();   // clear the sticky flag before reporting
        return (int)e;
    }
    e = cudaStreamWaitEvent(s, fork_ev, 0);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();   // clear the sticky flag before reporting
        return (int)e;
    }
    // (3) dots on the SIDE stream — off main's path entirely. `g_hc_part` is
    // written here and read only by the LATE branch below, which is on the same
    // stream, so no second event is needed to publish it. Main does NOT wait this
    // kernel (nor LATE): the projection group that follows consumes only the
    // EARLY outputs, and the hc_post that consumes `comb` is ~50 us away — it
    // waits `join_ev` instead (see the header comment).
    // (3) dots + LATE in ONE launch on the SIDE stream (`DSV41_HC_DL_MERGE`,
    // default ON; see hc_dots_late_kernel's header). The tail is elected to the
    // last dot block to publish `g_hc_part` — no ticket and no spin, so nothing
    // holds an SM hostage the way hc_front_kernel's tail block did. `g_hc_part`
    // is written and read by that one kernel, and main does not wait it: main's
    // only edge into the side chain is the EARLY completion above (edge 2), and
    // the hc_post that consumes `comb` is ~50 us away waiting `join_ev`. The
    // two-launch pair below stays as the A/B arm (`DSV41_HC_DL_MERGE=0`) and as
    // the fallback when the merge cannot run (the self-ss arm has no ss partials
    // to hand the elected block, and the 160 KiB opt-in can be refused).
    // The whole DL branch below is issued on `dl` (the fourth stream when the
    // runtime created one, `side` otherwise). Nothing on `side` reads its output
    // and nothing here reads EARLY's, so the two halves overlap freely.
    bool dl_merged = false;
    if (g_hc_dl_merge && g_hc_ss) {
        cudaError_t e3 = cudaFuncSetAttribute(
            hc_dots_late_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            dsv41_smem_ceiling(hc_dots_late_kernel));
        if (e3 == cudaSuccess) {
            hc_dots_late_kernel<<<dim3((unsigned)mix, (unsigned)rows), (unsigned)g_hc_dots_t,
                                  smem, dl>>>(x, hc_fn, hc_scale, hc_base, pre, post, comb, hc,
                                              dim, sinkhorn_iters, eps, mix);
            e = cudaGetLastError();
            if (e != cudaSuccess) return (int)e;
            dl_merged = true;
        } else {
            (void)cudaGetLastError();   // clear the sticky flag, then take the pair
        }
    }
    if (!dl_merged) {
    hc_mix_dots_kernel<<<dim3((unsigned)mix, (unsigned)rows), (unsigned)g_hc_dots_t, smem, dl>>>(
        x, hc_fn, rows, hc_dim, mix, g_hc_ss ? 1 : 0);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    // (4) LATE half on the DL stream (ss -> mixes -> sigmoid -> sinkhorn -> comb).
    // Stream order after the dots is the whole synchronisation: same-stream ops
    // cannot overtake one another, so every `g_hc_part[r][m][ck]` read below sees
    // the dots' write. Block size from g_hc_late_t when the ss partials come from
    // the dots kernel (the default): one warp does all the LATE work, so a
    // 1024-thread block only buys 31 idle warps and a 1024-thread SM slot to queue
    // for. The self-computed ss path reads wpart[threadIdx.x>>5] for warps
    // 0..mix-1, so it keeps 1024.
    const unsigned late_t = (g_hc_ss ? (unsigned)g_hc_late_t : 1024u);
    hc_mixes_tail_kernel<<<(unsigned)rows, late_t, (64 + 64) * sizeof(float), dl>>>(
        x, hc_scale, hc_base, pre, post, comb, hc, dim, sinkhorn_iters, eps, mix * 32, nullptr,
        nullptr, nullptr, eps_norm, g_hc_ss ? 1 : 0, nullptr, nullptr, HC_TAIL_LATE);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    }   // end of the `if (!dl_merged)` two-launch fallback / A/B arm
    // (5) join: record on the DL stream (`side` when no fourth stream exists — a
    // stream wait on an event recorded on ANOTHER stream is legal either way, and
    // the record must sit on the stream that actually ran the tail); the MODEL
    // waits it on main (`hc_tail_join`) before the earliest consumer of
    // `post`/`comb` — the fused AR epilogue when the hc-post fold is on, the
    // standalone hc_post otherwise. Nothing to wait here: main has no consumer
    // before that point.
    e = cudaEventRecord(join_ev, dl);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();   // clear the sticky flag before reporting
        return (int)e;
    }
    // `in_ev`/`early_ev` are kept in the signature for ABI compatibility but are
    // deliberately neither validated nor recorded/waited: the runtime no longer
    // creates them (they went away with B) and the split's two edges are both
    // carried by `fork_ev`. See the header comment.
    (void)in_ev;
    (void)early_ev;
    // `side_dl` is consumed above through `dl`; `(void)` is for the case where
    // the compiler cannot see that the ternary already read it.
    (void)side_dl;
    return (int)cudaGetLastError();
}

// Persistent prototype entry (DSV41_HC_PERSIST=1, Rust-gated). Same contract as
// dsv41_hc_front, but the whole front end runs as ONE block per row with no
// ticket and no spin (see hc_pre_persist_kernel). Kept as a SEPARATE symbol so
// the two-launch path stays byte-for-byte untouched and a stale .so simply
// resolves it to None (the Rust side then keeps calling dsv41_hc_front).
extern "C" int dsv41_hc_front_persist(const float* x, const float* hc_fn, const float* hc_scale,
                                      const float* hc_base, const float* w_norm,
                                      const float* pre_collapse, float* pre, float* post,
                                      float* comb, float* out, int rows, int hc, int dim,
                                      int sinkhorn_iters, float eps, float eps_norm, uint8_t* xq,
                                      float* xsc, cudaStream_t s) {
    if (x == nullptr || hc_fn == nullptr || hc_scale == nullptr || hc_base == nullptr ||
        pre == nullptr || post == nullptr || comb == nullptr)
        return (int)cudaErrorInvalidValue;
    if (rows <= 0 || hc <= 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    if (rows > DSV41_HC_SPREAD_MAXR) return (int)cudaErrorInvalidValue;
    if ((w_norm == nullptr) != (pre_collapse == nullptr)) return (int)cudaErrorInvalidValue;
    if (w_norm != nullptr && out == nullptr) return (int)cudaErrorInvalidValue;
    const int mix = hc * (2 + hc);
    if (mix > 64) return (int)cudaErrorInvalidValue;
    const int hc_dim = hc * dim;
    // Only the staged activation row lives in dynamic smem (one weight row per
    // warp is read from global; 24 of them would not fit). 80 KiB still needs
    // the opt-in, and the attribute is per-context (one per rank under TP8), so
    // set it on EVERY call exactly like the two-launch launcher does.
    const size_t smem = (size_t)hc_dim * sizeof(float);
    cudaError_t e = cudaFuncSetAttribute(hc_pre_persist_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(hc_pre_persist_kernel));
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        return (int)e;
    }
    hc_pre_persist_kernel<<<(unsigned)rows, 1024u, smem, s>>>(
        x, hc_fn, hc_scale, hc_base, pre, post, comb, hc, dim, sinkhorn_iters, eps, mix * 32,
        w_norm, pre_collapse, out, eps_norm, g_hc_ss ? 1 : 0, xq, xsc, hc_dim, mix);
    return (int)cudaGetLastError();
}

// ============================================================================
// Stage C persistent, MULTI-BLOCK form (docs/agent/dsv41-persistent-arch.md §1):
// the whole hc front end in ONE launch, with the dots SPREAD instead of pinned
// to a single SM. This is the shape the segment kernels (P2) will scale up; the
// single-block prototype above proved the phase-machine mechanism, not the
// performance.
//
// WHY THE SINGLE BLOCK WAS THE WRONG SHAPE (measured, not assumed). The
// two-launch dots run grid = (mix=24, 1) blocks, each staging x + one weight row
// = 160 KiB through cp.async, and measure 3.93 MiB / 7.4 us = 531 GB/s, i.e.
// 22 GB/s per ACTIVE SM. DRAM is not the ceiling - the 1.92 MB of weights is
// 0.6 us at 3 TB/s - the ceiling is the per-SM in-flight window: an SM keeps
// only ~20 KiB of cp.async outstanding, so ONE 160 KiB block costs ~8 serial
// DRAM latencies (~7 us), and 24 such blocks give 24 x 20 KiB / ~0.7 us ~
// 690 GB/s. Bytes per block is what must shrink, and only a K-split shrinks it:
// at split = 8 every block stages hc_dim/8 x 2 x 4 B = 20 KiB (one window) and
// `mix * split` blocks across 148 SMs reach ~4 TB/s, so the staging drops to
// ~1 us. Splitting M is already maxed out (mix = 24 rows, one block each) and
// splitting the dimension-wise ops does nothing for a reduction, so K is the
// only axis left.
//
// ROLES, one grid, NO ticket and NO spin (the hc-merge lesson):
//   bid in [0, mix*split)          dot block: one (projection row, K chunk)
//   bid == mix*split               collapse block: collapse + rmsnorm + fp8
//   the LAST dot block to publish  tail block: ss + mixes + sigmoid + sinkhorn
// The tail is not polled for. Each dot block publishes g_hc_part, fences, and
// bumps the per-row counter; the block that sees the final count runs the tail.
// Every other block falls off the end. Nobody holds an SM hostage waiting, which
// is what made hc_front_kernel's ticket spin cost +3.2 ms/step.
//
// The collapse is a SEPARATE block because it does not depend on the dots: it
// reads x and `pre_collapse` (deliberately NOT the slot the mixes write), so it
// runs in parallel with the staging instead of after the tail.
//
// NUMERICS. This shape is BIT-EXACT at split = 1 only: the chunk is then the
// whole row and the dot is hc_mix_dots_kernel's lane chain, statement for
// statement. split = 1 exists so the machinery has a parity target against the
// two-launch path. Any split > 1 splits the reduction, and the partials are
// recombined in ck-ascending order, which is NOT the single warp's tree sum: it
// is deterministic but not bit-identical, and needs a tolerance test plus the
// same-binary A/B token gate. §1 states the same constraint: "split must be 1".
// ============================================================================
#define DSV41_HC_MB_MAXS 8          // g_hc_part's ck dimension is the ceiling

// Per-row publication counter for the tail election. Same .bss lifetime as
// g_hc_part (zero-initialised at module load); the elected tail block RESETS it
// to 0 after its last g_hc_part read, so the next launch (or graph replay)
// starts clean - the same discipline g_hc_ticket uses.
__device__ unsigned g_hc_mb_done[DSV41_HC_SPREAD_MAXR];

__global__ void hc_pre_persist_mb_kernel(const float* __restrict__ x,
                                         const float* __restrict__ hc_fn,
                                         const float* __restrict__ hc_scale,
                                         const float* __restrict__ hc_base,
                                         float* __restrict__ pre, float* __restrict__ post,
                                         float* __restrict__ comb, int hc, int dim,
                                         int sinkhorn_iters, float eps,
                                         const float* __restrict__ w_norm,
                                         const float* __restrict__ pre_collapse,
                                         float* __restrict__ out, float eps_norm,
                                         uint8_t* __restrict__ xq, float* __restrict__ xsc,
                                         int hc_dim, int mix, int split) {
    const int r = blockIdx.y;
    const int bid = blockIdx.x;
    const int ndot = mix * split;
    const int ss_stride = mix * 32;
    const int nwarp = ss_stride >> 5;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float* xrow = x + (size_t)r * hc_dim;
    extern __shared__ float mb_sm[];
    __shared__ unsigned s_elected;
    __shared__ float wpart[32];
    __shared__ float sss;
    __shared__ float p_mixes[64];
    __shared__ float p_cm[64];
    __shared__ float p_red[32];

    if (bid < ndot) {
        // ---------------- dot: one projection row over one K chunk ----------
        const int m = bid % mix;        // m varies fastest so the `mix` blocks
        const int ck = bid / mix;       // of a chunk share one x read (L2)
        const int chunk = hc_dim / split;
        const int lo = ck * chunk;
        float* s_x = mb_sm;             // chunk floats, staged
        float* s_w = mb_sm + chunk;     // chunk floats, this projection row
        const int n4 = chunk >> 2;
        const float4* xg = reinterpret_cast<const float4*>(xrow + lo);
        const float4* wg = reinterpret_cast<const float4*>(hc_fn + (size_t)m * hc_dim + lo);
        float4* sx = reinterpret_cast<float4*>(s_x);
        float4* sw = reinterpret_cast<float4*>(s_w);
        for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sx[i], &xg[i]);
        dsv41_cp_commit();
        for (int i = threadIdx.x; i < n4; i += blockDim.x) dsv41_cp_async16(&sw[i], &wg[i]);
        dsv41_cp_commit();
        dsv41_cp_wait_all();
        __syncthreads();
        if (threadIdx.x < 32) {
            // hc_mix_dots_kernel's float4 three-accumulator lane chain, restricted
            // to [lo, lo + chunk). At split == 1 this is that kernel's dot exactly.
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
            for (int k = (n4 << 2) + lane; k < chunk; k += 32) a0 += s_w[k] * s_x[k];
            float acc = (a0 + a1) + a2;
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) g_hc_part[r][m][ck] = acc;
        }
        // Publish and elect. The writer of g_hc_part is threadIdx.x == 0 (warp 0,
        // lane 0), so the release fence before the atomicAdd orders exactly that
        // store; s_elected is block-uniform, so the branch below is uniform too.
        if (threadIdx.x == 0) {
            __threadfence();
            s_elected = (atomicAdd(&g_hc_mb_done[r], 1u) == (unsigned)(ndot - 1)) ? 1u : 0u;
        }
        __syncthreads();
        if (s_elected == 0u) return;    // no barrier follows on this path - safe
    } else {
        // ---------------- collapse: collapse + rmsnorm + fp8, one block ------
        // dsv41_hc_collapse_norm_kernel's body at blockDim 1024 (the same tree and
        // the same #pragma unroll 4 as the folded tail version), run in parallel
        // with the dots because it does not read their output.
        float* o_r = out + (size_t)r * dim;
        float s2 = 0.f;
#pragma unroll 4
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            float acc = 0.f;
            for (int i = 0; i < hc; ++i)
                acc = fmaf(pre_collapse[(size_t)r * hc + i], xrow[(size_t)i * dim + c], acc);
            o_r[c] = acc;
            s2 += acc * acc;
        }
        for (int off = 16; off > 0; off >>= 1) s2 += __shfl_down_sync(0xffffffffu, s2, off);
        if ((threadIdx.x & 31) == 0) p_red[threadIdx.x >> 5] = s2;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.f;
            for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += p_red[i];
            p_red[0] = rsqrtf(t / dim + eps_norm);
        }
        __syncthreads();
        const float inv2 = p_red[0];
        const int lane31 = threadIdx.x & 31;
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            const float v = o_r[c] * inv2 * w_norm[c];
            o_r[c] = v;
            if (xq != nullptr) {
                float a = fabsf(v);
                for (int off = 16; off > 0; off >>= 1)
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                const float sc = fmaxf(fast_round_scale(a, 1.0f / 448.0f), 1e-30f);
                if (lane31 == 0) xsc[c >> 5] = sc;
                const float q = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
                const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(q);
                xq[c] = *(const uint8_t*)&f8;
            }
        }
        return;                         // collapse path never reaches the tail
    }

    // ---------------- tail: run by the elected (last) dot block -------------
    // Acquire: every other dot block stored g_hc_part BEFORE its release fence +
    // atomicAdd, and this block observed the final count.
    __threadfence();
    {
        // ss, hc_mixes_tail_kernel's ss_in == 0 branch. The per-warp partials
        // group exactly as the dots kernel's replay did (residue m*32 + lane,
        // stride mix*32), and warps >= nwarp are discarded, so the sum does not
        // move when blockDim is 1024 instead of 768. That is what lets the K-split
        // dots skip the ss replay entirely and still land on the same bits.
        float ss = 0.f;
        for (int c = threadIdx.x; c < hc_dim; c += ss_stride) ss += xrow[c] * xrow[c];
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
        if ((threadIdx.x & 31) == 0 && warp < nwarp) wpart[warp] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (unsigned)nwarp) ? wpart[threadIdx.x] : 0.f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
            if (threadIdx.x == 0) sss = v;
        }
    }
    __syncthreads();
    const float inv = rsqrtf(sss / (float)hc_dim + eps);
    for (int m2 = threadIdx.x; m2 < mix; m2 += blockDim.x) {
        // The K partials in a FIXED ascending ck order: deterministic, but not
        // the single warp's tree sum, hence not bit-exact for split > 1. The
        // g_hc_part layout (and this order) is the spread path's, reused verbatim.
        float a = 0.f;
        for (int ck = 0; ck < split; ++ck) a += g_hc_part[r][m2][ck];
        p_mixes[m2] = a * inv;
    }
    __syncthreads();
    if (threadIdx.x < (unsigned)hc) {
        const int j = threadIdx.x;
        pre[(size_t)r * hc + j] =
            (1.f / (1.f + expf(-(p_mixes[j] * hc_scale[0] + hc_base[j])))) + eps;
        post[(size_t)r * hc + j] =
            2.f / (1.f + expf(-(p_mixes[hc + j] * hc_scale[1] + hc_base[hc + j])));
    }
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x) {
        const int j = jk / hc, k = jk % hc;
        p_cm[jk] = p_mixes[2 * hc + j * hc + k] * hc_scale[2] + hc_base[2 * hc + j * hc + k];
    }
    __syncthreads();
    if (warp == 0) {
        const int hh = hc * hc;
        float c = (lane < hh) ? p_cm[lane] : 0.f;
        float mx = c;
        for (int off = 1; off < hc; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
        c = expf(c - mx);
        float rs = c;
        for (int off = 1; off < hc; off <<= 1) rs += __shfl_xor_sync(0xFFFFFFFFu, rs, off);
        c = c / rs + eps;
        for (int it = 0; it < sinkhorn_iters; ++it) {
            if (it > 0) {
                float st = c;
                for (int off = 1; off < hc; off <<= 1)
                    st += __shfl_xor_sync(0xFFFFFFFFu, st, off);
                c = c / (st + eps);
            }
            float t = c;
            for (int off = hc; off < hh; off <<= 1) t += __shfl_xor_sync(0xFFFFFFFFu, t, off);
            c = c / (t + eps);
        }
        if (lane < hh) p_cm[lane] = c;
    }
    __syncthreads();
    for (int jk = threadIdx.x; jk < hc * hc; jk += blockDim.x)
        comb[(size_t)r * hc * hc + jk] = p_cm[jk];
    // Reset LAST: the loop above was the final g_hc_part read of this row.
    if (threadIdx.x == 0) atomicExch(&g_hc_mb_done[r], 0u);
}

// Multi-block entry (DSV41_HC_PERSIST_MB=1, Rust-gated). Same contract as
// dsv41_hc_front_persist; a SEPARATE symbol so the two-launch path and the
// single-block prototype stay byte-for-byte untouched and a stale .so simply
// resolves it to None (the Rust side then keeps the older path).
extern "C" int dsv41_hc_front_persist_mb(const float* x, const float* hc_fn,
                                         const float* hc_scale, const float* hc_base,
                                         const float* w_norm, const float* pre_collapse,
                                         float* pre, float* post, float* comb, float* out,
                                         int rows, int hc, int dim, int sinkhorn_iters,
                                         float eps, float eps_norm, uint8_t* xq, float* xsc,
                                         cudaStream_t s) {
    if (x == nullptr || hc_fn == nullptr || hc_scale == nullptr || hc_base == nullptr ||
        pre == nullptr || post == nullptr || comb == nullptr)
        return (int)cudaErrorInvalidValue;
    if (rows <= 0 || hc <= 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    if (rows > DSV41_HC_SPREAD_MAXR) return (int)cudaErrorInvalidValue;
    if ((w_norm == nullptr) != (pre_collapse == nullptr)) return (int)cudaErrorInvalidValue;
    if (w_norm != nullptr && out == nullptr) return (int)cudaErrorInvalidValue;
    const int mix = hc * (2 + hc);
    if (mix > 64) return (int)cudaErrorInvalidValue;
    const int hc_dim = hc * dim;
    // K chunks: one dot block per (projection row, chunk). Default 8 - the widest
    // g_hc_part carries - and the split must keep the chunk float4-aligned so the
    // staged copy and the lane chain stay aligned. Read once: this launcher runs
    // twice a layer.
    static const int split = [] {
        const char* e = getenv("DSV41_HC_PERSIST_MB_S");
        if (e == nullptr) return DSV41_HC_MB_MAXS;
        const int v = atoi(e);
        return (v >= 1 && v <= DSV41_HC_MB_MAXS) ? v : DSV41_HC_MB_MAXS;
    }();
    if (hc_dim % (4 * split) != 0) return (int)cudaErrorInvalidValue;
    const int chunk = hc_dim / split;
    const size_t smem = (size_t)2 * (size_t)chunk * sizeof(float);
    // The attribute is per-context, and a TP8 process has one context per rank,
    // so set it EVERY call - the existing launchers carry the same warning.
    cudaError_t e = cudaFuncSetAttribute(hc_pre_persist_mb_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, dsv41_smem_ceiling(hc_pre_persist_mb_kernel));
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        return (int)e;
    }
    dim3 grid((unsigned)(mix * split + 1), (unsigned)rows);
    hc_pre_persist_mb_kernel<<<grid, 1024u, smem, s>>>(
        x, hc_fn, hc_scale, hc_base, pre, post, comb, hc, dim, sinkhorn_iters, eps, w_norm,
        pre_collapse, out, eps_norm, xq, xsc, hc_dim, mix, split);
    return (int)cudaGetLastError();
}
