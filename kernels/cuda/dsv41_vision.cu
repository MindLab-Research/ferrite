// =====================================================================
// dsv41_vision.cu — DeepSeek-V4.1-Flash vision-tower device kernels.
//
// Device counterpart of `crates/ferrite-dsv41/src/vision.rs` (the CPU
// golden): the ViT encoder's GEMMs, its bidirectional attention with a
// standard online softmax, RMSNorm / 2D-RoPE / activations and the
// Aligner's 3x3 `unfold` pack.
//
// Data contract (non-negotiable):
//   * The vision tower is bf16 *natively*. Nothing is dequantised: weights
//     and activations stay bf16 end to end; every large matmul is a
//     tensor-core `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
//     with an f32 accumulator, rounded back to bf16 at the store.
//   * Norm gains are f32 (the reference `vision.py::RMSNorm.weight` is an
//     f32 parameter); everything else is bf16.
//
// Layout conventions (match the reference's `vision.py` / `image_processor.py`):
//   patches   [n_vit, 3*14*14]  row-major, patch grid (h, w) row-major
//   q/k/v/out [n_tokens, n_heads, 64]
//   mlp       gate/up interleaved as the reference's `w1` chunk(2, -1):
//             `gu[row, 0..inter) = gate`, `gu[row, inter..2*inter) = up`
//   Aligner pack: out[l, c*r*r + i*r + j] = x_pad[c, oh*r+i, ow*r+j] with
//             l = oh*n_llm_w + ow (the `F.unfold` channel-major order), zero
//             padded at the right/bottom edges.
//
// Kernel inventory (all shapes on the released checkpoint):
//   gemm_bf16       C[M,N] = A[M,K] @ W[N,K]^T + bias   (patch_embed 1024x588,
//                   wqkv 3072x1024, wo 1024x1024, mlp 5632x1024 & 1024x2816,
//                   aligner 5120x9216 & 5120x5120)
//   attn_bf16       full bidirectional attention, head_dim 64, online softmax
//   rms_norm_bf16   one CTA per row
//   silu_mul_bf16   SwiGLU: silu(gate)*up, `w1` output shape [rows, 2*inter]
//   gelu_bf16       exact erf GELU (the Aligner's activation)
//   rope_tables / rope2d_apply   2D RoPE ([n_tokens, head_dim/2] cos/sin)
//   aligner_pack_bf16   patch grid -> 3x3 folded rows
//
// Tuning state: BM 128 x BN 64 x BK 32 for the GEMM (8 warps, ping-pong
// smem, scalar staging loads) and 64x64 tiles for attention (4 warps).
// cp.async staging and warp-specialised pipelining are deliberate
// follow-ups, not correctness requirements.
// =====================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

#define WARP_FULL 0xffffffffu

// ---------------------------------------------------------------- helpers

__device__ __forceinline__ bf16 bf16_zero() { return __float2bfloat16(0.f); }

// Pack two floats as bf16 into one 32-bit register (lo half = `lo`).
__device__ __forceinline__ unsigned bf16_pack2(float lo, float hi) {
    const unsigned l = (unsigned)__bfloat16_as_ushort(__float2bfloat16(lo));
    const unsigned h = (unsigned)__bfloat16_as_ushort(__float2bfloat16(hi));
    return l | (h << 16);
}

// 4-byte shared-memory load (both halves must be 4B aligned).
__device__ __forceinline__ unsigned smem_ld32(const bf16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

// One m16n8k16 bf16 MMA: c[4] += a[4] x b[2] (f32 accumulate).
__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(WARP_FULL, v, 1);
    v += __shfl_xor_sync(WARP_FULL, v, 2);
    v += __shfl_xor_sync(WARP_FULL, v, 4);
    v += __shfl_xor_sync(WARP_FULL, v, 8);
    v += __shfl_xor_sync(WARP_FULL, v, 16);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(WARP_FULL, v, 1));
    v = fmaxf(v, __shfl_xor_sync(WARP_FULL, v, 2));
    v = fmaxf(v, __shfl_xor_sync(WARP_FULL, v, 4));
    v = fmaxf(v, __shfl_xor_sync(WARP_FULL, v, 8));
    v = fmaxf(v, __shfl_xor_sync(WARP_FULL, v, 16));
    return v;
}

// =====================================================================
// 1. GEMM: C[M, N] = A[M, K] @ W[N, K]^T (+ bias), bf16 in, bf16 out.
//
//    CTA tile 128 (M) x 64 (N), K-block 32, 256 threads = 8 warps laid
//    out 4 (M) x 2 (N); each warp owns a 32x32 patch = 2x4 m16n8 tiles.
//    Shared memory is double buffered (ping-pong); the K loop is
//         load(next) ; compute(current) ; __syncthreads()
//    with zero-filled tiles at the M/N/K edges, so any shape is legal.
// =====================================================================

constexpr int GBM = 128;
constexpr int GBN = 64;
constexpr int GBK = 32;
constexpr int GTHREADS = 256;

__device__ __forceinline__ void gemm_load_tile(bf16* as, bf16* bs,
                                               const bf16* __restrict__ a,
                                               const bf16* __restrict__ w,
                                               int m0, int k0, int m, int n, int k) {
    // A tile [128 x 32]
    for (int i = threadIdx.x; i < GBM * GBK; i += GTHREADS) {
        const int r = i / GBK;
        const int c = i - r * GBK;
        const int gr = m0 + r;
        const int gc = k0 + c;
        as[i] = (gr < m && gc < k) ? a[(size_t)gr * k + gc] : bf16_zero();
    }
    // B tile: W rows [blockIdx.x*64, +64) x K columns [k0, k0+32)
    for (int i = threadIdx.x; i < GBN * GBK; i += GTHREADS) {
        const int r = i / GBK;
        const int c = i - r * GBK;
        const int gr = blockIdx.x * GBN + r;
        const int gc = k0 + c;
        bs[i] = (gr < n && gc < k) ? w[(size_t)gr * k + gc] : bf16_zero();
    }
}

__device__ __forceinline__ void gemm_store(bf16* c, const bf16* bias, int m, int n,
                                           int r, int col, float v) {
    if (r < m && col < n) {
        const float bb = bias ? __bfloat162float(bias[col]) : 0.f;
        c[(size_t)r * n + col] = __float2bfloat16(v + bb);
    }
}

__global__ void __launch_bounds__(GTHREADS) dsv41v_gemm_bf16_kernel(
    const bf16* __restrict__ a, const bf16* __restrict__ w,
    const bf16* __restrict__ bias, bf16* __restrict__ c,
    int m, int n, int k) {
    __shared__ __align__(16) bf16 smem[2][GBM * GBK + GBN * GBK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int wm = warp & 3;   // M-direction warp (0..3)
    const int wn = warp >> 2;  // N-direction warp (0..1)

    const int m0 = blockIdx.y * GBM;
    const int n0 = blockIdx.x * GBN;
    const int nk = (k + GBK - 1) / GBK;

    float acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[mt][nt][j] = 0.f;

    gemm_load_tile(smem[0], smem[0] + GBM * GBK, a, w, m0, 0, m, n, k);
    __syncthreads();

    for (int kb = 0; kb < nk; kb++) {
        const int cur = kb & 1;
        if (kb + 1 < nk) {
            gemm_load_tile(smem[cur ^ 1], smem[cur ^ 1] + GBM * GBK, a, w, m0,
                           (kb + 1) * GBK, m, n, k);
        }
        const bf16* as = smem[cur];
        const bf16* bs = smem[cur] + GBM * GBK;

#pragma unroll
        for (int kc = 0; kc < GBK; kc += 16) {
            unsigned af[2][4];
            unsigned wf[4][2];
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                const int r = wm * 32 + mt * 16;
                af[mt][0] = smem_ld32(&as[(r + group) * GBK + kc + tig * 2]);
                af[mt][1] = smem_ld32(&as[(r + group + 8) * GBK + kc + tig * 2]);
                af[mt][2] = smem_ld32(&as[(r + group) * GBK + kc + tig * 2 + 8]);
                af[mt][3] = smem_ld32(&as[(r + group + 8) * GBK + kc + tig * 2 + 8]);
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int r = wn * 32 + nt * 8;
                wf[nt][0] = smem_ld32(&bs[(r + group) * GBK + kc + tig * 2]);
                wf[nt][1] = smem_ld32(&bs[(r + group) * GBK + kc + tig * 2 + 8]);
            }
#pragma unroll
            for (int mt = 0; mt < 2; mt++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) mma_bf16(acc[mt][nt], af[mt], wf[nt]);
        }
        __syncthreads();
    }

    // epilogue: f32 accumulators (+ bias) -> bf16 rows [m, n)
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int r0 = m0 + wm * 32 + mt * 16 + group;
            const int c0 = n0 + wn * 32 + nt * 8 + tig * 2;
            gemm_store(c, bias, m, n, r0, c0, acc[mt][nt][0]);
            gemm_store(c, bias, m, n, r0, c0 + 1, acc[mt][nt][1]);
            gemm_store(c, bias, m, n, r0 + 8, c0, acc[mt][nt][2]);
            gemm_store(c, bias, m, n, r0 + 8, c0 + 1, acc[mt][nt][3]);
        }
    }
}

extern "C" cudaError_t dsv41v_gemm_bf16(const void* a, const void* w, const void* bias,
                                        void* out, int m, int n, int k, cudaStream_t s) {
    if (!a || !w || !out || m <= 0 || n <= 0 || k <= 0) return cudaErrorInvalidValue;
    dim3 grid((unsigned)((n + GBN - 1) / GBN), (unsigned)((m + GBM - 1) / GBM));
    dsv41v_gemm_bf16_kernel<<<grid, GTHREADS, 0, s>>>(
        (const bf16*)a, (const bf16*)w, (const bf16*)bias, (bf16*)out, m, n, k);
    return cudaGetLastError();
}

// =====================================================================
// 2. RMSNorm: y[r, :] = w * (x[r, :] * rsqrt(mean(x^2) + eps))
//    One CTA per row; f32 mean-square reduction, bf16 store; `w` is f32.
// =====================================================================

__global__ void dsv41v_rmsnorm_bf16_kernel(const bf16* __restrict__ x,
                                           const float* __restrict__ weight,
                                           bf16* __restrict__ y, int dim, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const bf16* xr = x + (size_t)row * dim;

    float ss = 0.f;
    for (int c = tid; c < dim; c += blockDim.x) {
        const float v = __bfloat162float(xr[c]);
        ss = fmaf(v, v, ss);
    }

    __shared__ float red[32];
    ss = warp_reduce_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        const int nwarps = (blockDim.x + 31) >> 5;
        float s2 = (tid < nwarps) ? red[tid] : 0.f;
        s2 = warp_reduce_sum(s2);
        if (tid == 0) red[0] = s2;
    }
    __syncthreads();

    const float inv = rsqrtf(red[0] / (float)dim + eps);
    for (int c = tid; c < dim; c += blockDim.x) {
        const float v = __bfloat162float(xr[c]);
        y[(size_t)row * dim + c] = __float2bfloat16(weight[c] * (v * inv));
    }
}

extern "C" cudaError_t dsv41v_rmsnorm_bf16(const void* x, const float* weight, void* y,
                                           int rows, int dim, float eps, cudaStream_t s) {
    if (!x || !weight || !y || rows <= 0 || dim <= 0) return cudaErrorInvalidValue;
    dsv41v_rmsnorm_bf16_kernel<<<rows, 128, 0, s>>>((const bf16*)x, weight, (bf16*)y,
                                                    dim, eps);
    return cudaGetLastError();
}

// =====================================================================
// 3. SwiGLU: out[i] = silu(gate) * up, gate/up interleaved per row of a
//    [rows, 2*inter] buffer (the reference's `w1(x).chunk(2, -1)`).
// =====================================================================

__global__ void dsv41v_silu_mul_bf16_kernel(const bf16* __restrict__ gu,
                                            bf16* __restrict__ out, long long total,
                                            int inter) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const long long row = i / inter;
        const int j = (int)(i - row * inter);
        const float g = __bfloat162float(gu[row * 2 * inter + j]);
        const float u = __bfloat162float(gu[row * 2 * inter + inter + j]);
        out[i] = __float2bfloat16((g / (1.f + expf(-g))) * u);
    }
}

extern "C" cudaError_t dsv41v_silu_mul_bf16(const void* gu, void* out, long long rows,
                                            int inter, cudaStream_t s) {
    if (!gu || !out || rows <= 0 || inter <= 0) return cudaErrorInvalidValue;
    const long long total = rows * (long long)inter;
    dim3 grid((unsigned)((total + 255) / 256));
    dsv41v_silu_mul_bf16_kernel<<<grid, 256, 0, s>>>((const bf16*)gu, (bf16*)out, total,
                                                     inter);
    return cudaGetLastError();
}

// =====================================================================
// 4. Exact (erf) GELU — the Aligner's activation (`F.gelu`).
// =====================================================================

__global__ void dsv41v_gelu_bf16_kernel(const bf16* __restrict__ x, bf16* __restrict__ y,
                                        long long total) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const float v = __bfloat162float(x[i]);
        y[i] = __float2bfloat16(0.5f * v * (1.f + erff(v * 0.7071067811865476f)));
    }
}

extern "C" cudaError_t dsv41v_gelu_bf16(const void* x, void* y, long long total,
                                        cudaStream_t s) {
    if (!x || !y || total <= 0) return cudaErrorInvalidValue;
    dim3 grid((unsigned)((total + 1023) / 1024));
    dsv41v_gelu_bf16_kernel<<<grid, 256, 0, s>>>((const bf16*)x, (bf16*)y, total);
    return cudaGetLastError();
}

// =====================================================================
// 5. 2D RoPE (vision.py::get_vision_cos_sin + apply_rotary).
//    Table [n_tokens, rope_dim], rope_dim = head_dim/2; the first half of
//    each row multiplies the row position, the second half the column.
//    `rope2d_apply` rotates q or k in place, [n_tokens, n_heads, head_dim].
// =====================================================================

__global__ void dsv41v_rope_tables_kernel(float* __restrict__ cos_t,
                                          float* __restrict__ sin_t, long long total,
                                          int n_w, int rope_dim, float theta) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    const int half = rope_dim / 2;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const int k = (int)(i % rope_dim);
        const int row = (int)(i / rope_dim);
        const int h = row / n_w;
        const int w = row - h * n_w;
        const int hh = (k < half) ? k : k - half;
        const float freq = powf(theta, -2.0f * (float)hh / (float)rope_dim);
        const float ang = ((k < half) ? (float)h : (float)w) * freq;
        cos_t[i] = cosf(ang);
        sin_t[i] = sinf(ang);
    }
}

extern "C" cudaError_t dsv41v_rope_tables(float* cos_t, float* sin_t, int n_tokens,
                                          int n_h, int n_w, int rope_dim, float theta,
                                          cudaStream_t s) {
    if (!cos_t || !sin_t || n_tokens <= 0 || n_w <= 0 || rope_dim <= 0) {
        return cudaErrorInvalidValue;
    }
    if (n_h * n_w != n_tokens || (rope_dim & 1)) return cudaErrorInvalidValue;
    const long long total = (long long)n_tokens * rope_dim;
    dim3 grid((unsigned)((total + 255) / 256));
    dsv41v_rope_tables_kernel<<<grid, 256, 0, s>>>(cos_t, sin_t, total, n_w, rope_dim,
                                                   theta);
    return cudaGetLastError();
}

__global__ void dsv41v_rope2d_apply_kernel(bf16* __restrict__ x,
                                           const float* __restrict__ cos_t,
                                           const float* __restrict__ sin_t, long long total,
                                           int n_heads, int head_dim) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    const int half = head_dim / 2;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const int k = (int)(i % half);
        const long long t = i / half;
        const int h = (int)(t % n_heads);
        const long long tok = t / n_heads;
        const size_t base = (size_t)(tok * n_heads + h) * head_dim;
        const float x1 = __bfloat162float(x[base + k]);
        const float x2 = __bfloat162float(x[base + half + k]);
        const float c = cos_t[tok * half + k];
        const float s = sin_t[tok * half + k];
        x[base + k] = __float2bfloat16(x1 * c - x2 * s);
        x[base + half + k] = __float2bfloat16(x2 * c + x1 * s);
    }
}

extern "C" cudaError_t dsv41v_rope2d_apply(void* x, const float* cos_t, const float* sin_t,
                                           int n_tokens, int n_heads, int head_dim,
                                           cudaStream_t s) {
    if (!x || !cos_t || !sin_t || n_tokens <= 0 || n_heads <= 0) return cudaErrorInvalidValue;
    if (head_dim <= 0 || (head_dim & 1)) return cudaErrorInvalidValue;
    const long long total = (long long)n_tokens * n_heads * (head_dim / 2);
    dim3 grid((unsigned)((total + 255) / 256));
    dsv41v_rope2d_apply_kernel<<<grid, 256, 0, s>>>((bf16*)x, cos_t, sin_t, total, n_heads,
                                                    head_dim);
    return cudaGetLastError();
}

// =====================================================================
// 6. Full bidirectional attention (no mask) with a standard online
//    softmax, head_dim fixed at 64 (the ViT's width).
//
//    CTA = 1 head x 64 query rows; 4 warps, one 16-row m-tile each.
//    Keys/values stream in 64-wide blocks; K is staged row-major, V
//    transposed ([feature][key], stride 66 to break bank conflicts).
//    Per warp and block:
//      S[8 n-tiles][4] = Q @ K^T              (16 MMAs, f32)
//      m_new = max(m, rowmax(S)); l = l*exp(m-m_new) + rowsum(exp(S-m_new))
//      o = o*exp(m-m_new) + P @ V             (16 MMAs, P rounded to bf16)
//    The row statistics are reduced across the 4 lanes of a quad with
//    __shfl_xor_sync (all four lanes hold the same rows).
//    Right/bottom key padding is masked to -inf so it never enters the
//    softmax; padded query rows are never written out.
// =====================================================================

constexpr int AH = 64;   // head_dim
constexpr int ABM = 64;  // query rows per CTA
constexpr int ABN = 64;  // keys per step
constexpr int ATHREADS = 128;

__global__ void __launch_bounds__(ATHREADS) dsv41v_attn_bf16_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, bf16* __restrict__ out, int n_tokens, int n_heads) {
    __shared__ __align__(16) bf16 qs[ABM * AH];
    __shared__ __align__(16) bf16 ks[ABN * AH];
    __shared__ __align__(16) bf16 vt[AH * (ABN + 2)];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int tig = lane & 3;

    const int head = blockIdx.x;
    const int q0 = blockIdx.y * ABM;
    const size_t hstride = (size_t)n_heads * AH;

    // stage this CTA's Q rows [64 x 64]
    for (int i = tid; i < ABM * AH; i += ATHREADS) {
        const int r = i >> 6;
        const int c = i & 63;
        const int gr = q0 + r;
        qs[i] = (gr < n_tokens) ? q[(size_t)gr * hstride + (size_t)head * AH + c]
                                : bf16_zero();
    }
    __syncthreads();

    // Q fragments: 4 k-chunks x 4 registers for this warp's 16 rows
    unsigned qf[4][4];
    const int r0 = warp * 16;
#pragma unroll
    for (int kc = 0; kc < 4; kc++) {
        qf[kc][0] = smem_ld32(&qs[(r0 + group) * AH + kc * 16 + tig * 2]);
        qf[kc][1] = smem_ld32(&qs[(r0 + group + 8) * AH + kc * 16 + tig * 2]);
        qf[kc][2] = smem_ld32(&qs[(r0 + group) * AH + kc * 16 + tig * 2 + 8]);
        qf[kc][3] = smem_ld32(&qs[(r0 + group + 8) * AH + kc * 16 + tig * 2 + 8]);
    }

    float o[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int j = 0; j < 4; j++) o[nt][j] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int nk = (n_tokens + ABN - 1) / ABN;
    for (int kb = 0; kb < nk; kb++) {
        const int k0 = kb * ABN;
        for (int i = tid; i < ABN * AH; i += ATHREADS) {
            const int r = i >> 6;
            const int c = i & 63;
            const int gr = k0 + r;
            bf16 kk = bf16_zero();
            bf16 vv = bf16_zero();
            if (gr < n_tokens) {
                kk = k[(size_t)gr * hstride + (size_t)head * AH + c];
                vv = v[(size_t)gr * hstride + (size_t)head * AH + c];
            }
            ks[i] = kk;
            vt[c * (ABN + 2) + r] = vv;  // transpose on the way in
        }
        __syncthreads();

        // ---- S = Q @ K^T, [8 n-tiles][4] f32 ----
        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int j = 0; j < 4; j++) s[nt][j] = 0.f;
            unsigned wf[4][2];
#pragma unroll
            for (int kc = 0; kc < 4; kc++) {
                wf[kc][0] = smem_ld32(&ks[(nt * 8 + group) * AH + kc * 16 + tig * 2]);
                wf[kc][1] = smem_ld32(&ks[(nt * 8 + group) * AH + kc * 16 + tig * 2 + 8]);
            }
#pragma unroll
            for (int kc = 0; kc < 4; kc++) mma_bf16(s[nt], qf[kc], wf[kc]);
        }

        // ---- scale + mask padded keys, then online softmax ----
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int col = k0 + nt * 8 + tig * 2;
            s[nt][0] *= 0.125f;
            s[nt][1] *= 0.125f;
            s[nt][2] *= 0.125f;
            s[nt][3] *= 0.125f;
            if (col >= n_tokens) {
                s[nt][0] = -INFINITY;
                s[nt][1] = -INFINITY;
                s[nt][2] = -INFINITY;
                s[nt][3] = -INFINITY;
            } else if (col + 1 >= n_tokens) {
                s[nt][1] = -INFINITY;
                s[nt][3] = -INFINITY;
            }
        }

        float lm0 = -INFINITY, lm1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            lm0 = fmaxf(lm0, fmaxf(s[nt][0], s[nt][1]));
            lm1 = fmaxf(lm1, fmaxf(s[nt][2], s[nt][3]));
        }
        lm0 = fmaxf(lm0, __shfl_xor_sync(WARP_FULL, lm0, 1));
        lm0 = fmaxf(lm0, __shfl_xor_sync(WARP_FULL, lm0, 2));
        lm1 = fmaxf(lm1, __shfl_xor_sync(WARP_FULL, lm1, 1));
        lm1 = fmaxf(lm1, __shfl_xor_sync(WARP_FULL, lm1, 2));

        const float mn0 = fmaxf(m0, lm0);
        const float mn1 = fmaxf(m1, lm1);
        const float a0 = expf(m0 - mn0);
        const float a1 = expf(m1 - mn1);
        m0 = mn0;
        m1 = mn1;

        float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            s[nt][0] = expf(s[nt][0] - mn0);
            ls0 += s[nt][0];
            s[nt][1] = expf(s[nt][1] - mn0);
            ls0 += s[nt][1];
            s[nt][2] = expf(s[nt][2] - mn1);
            ls1 += s[nt][2];
            s[nt][3] = expf(s[nt][3] - mn1);
            ls1 += s[nt][3];
        }
        ls0 += __shfl_xor_sync(WARP_FULL, ls0, 1);
        ls0 += __shfl_xor_sync(WARP_FULL, ls0, 2);
        ls1 += __shfl_xor_sync(WARP_FULL, ls1, 1);
        ls1 += __shfl_xor_sync(WARP_FULL, ls1, 2);
        l0 = l0 * a0 + ls0;
        l1 = l1 * a1 + ls1;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            o[nt][0] *= a0;
            o[nt][1] *= a0;
            o[nt][2] *= a1;
            o[nt][3] *= a1;
        }

        // ---- P (bf16) as the next A operand; k-chunk c = columns 16c..16c+16 ----
        unsigned pf[4][4];
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            pf[kc][0] = bf16_pack2(s[2 * kc][0], s[2 * kc][1]);
            pf[kc][1] = bf16_pack2(s[2 * kc][2], s[2 * kc][3]);
            pf[kc][2] = bf16_pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pf[kc][3] = bf16_pack2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
        }

        // ---- O += P @ V ----
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned vf[4][2];
#pragma unroll
            for (int kc = 0; kc < 4; kc++) {
                vf[kc][0] =
                    smem_ld32(&vt[(nt * 8 + group) * (ABN + 2) + kc * 16 + tig * 2]);
                vf[kc][1] =
                    smem_ld32(&vt[(nt * 8 + group) * (ABN + 2) + kc * 16 + tig * 2 + 8]);
            }
#pragma unroll
            for (int kc = 0; kc < 4; kc++) mma_bf16(o[nt], pf[kc], vf[kc]);
        }
        __syncthreads();
    }

    // ---- epilogue: o / l, only for real query rows ----
    const float inv0 = (l0 > 0.f) ? 1.f / l0 : 0.f;
    const float inv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const int rr0 = q0 + warp * 16 + group;
        const int cc = nt * 8 + tig * 2;
        if (rr0 < n_tokens) {
            out[(size_t)rr0 * hstride + (size_t)head * AH + cc] =
                __float2bfloat16(o[nt][0] * inv0);
            out[(size_t)rr0 * hstride + (size_t)head * AH + cc + 1] =
                __float2bfloat16(o[nt][1] * inv0);
        }
        const int rr1 = rr0 + 8;
        if (rr1 < n_tokens) {
            out[(size_t)rr1 * hstride + (size_t)head * AH + cc] =
                __float2bfloat16(o[nt][2] * inv1);
            out[(size_t)rr1 * hstride + (size_t)head * AH + cc + 1] =
                __float2bfloat16(o[nt][3] * inv1);
        }
    }
}

extern "C" cudaError_t dsv41v_attn_bf16(const void* q, const void* k, const void* v,
                                        void* out, int n_tokens, int n_heads,
                                        cudaStream_t s) {
    if (!q || !k || !v || !out || n_tokens <= 0 || n_heads <= 0) return cudaErrorInvalidValue;
    dim3 grid((unsigned)n_heads, (unsigned)((n_tokens + ABM - 1) / ABM));
    dsv41v_attn_bf16_kernel<<<grid, ATHREADS, 0, s>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)out, n_tokens, n_heads);
    return cudaGetLastError();
}

// =====================================================================
// 7. Aligner pack (vision.py::Aligner's view/permute/pad/unfold):
//    patch grid [n_h, n_w, vit_dim] -> [n_llm_h*n_llm_w, r*r*vit_dim],
//    out[l, c*r*r + i*r + j] = x_pad[c, oh*r+i, ow*r+j], l = oh*n_llm_w+ow,
//    zero at the right/bottom padding. (w1/gelu/w2 then run on `gemm_bf16`
//    / `gelu_bf16`.)
// =====================================================================

__global__ void dsv41v_aligner_pack_bf16_kernel(const bf16* __restrict__ x,
                                                bf16* __restrict__ out, long long total,
                                                int n_h, int n_w, int vit_dim, int r) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    const int n_llm_w = (n_w + r - 1) / r;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const int j = (int)(i % r);
        long long t = i / r;
        const int ii = (int)(t % r);
        t /= r;
        const int c = (int)(t % vit_dim);
        const int l = (int)(t / vit_dim);
        const int ow = l % n_llm_w;
        const int oh = l / n_llm_w;
        const int rr = oh * r + ii;
        const int cc = ow * r + j;
        bf16 val = bf16_zero();
        if (rr < n_h && cc < n_w) val = x[((size_t)rr * n_w + cc) * vit_dim + c];
        out[i] = val;
    }
}

extern "C" cudaError_t dsv41v_aligner_pack_bf16(const void* x, void* out, int n_h, int n_w,
                                                int vit_dim, int r, cudaStream_t s) {
    if (!x || !out || n_h <= 0 || n_w <= 0 || vit_dim <= 0 || r <= 0) {
        return cudaErrorInvalidValue;
    }
    const long long n_llm_h = (n_h + r - 1) / r;
    const long long n_llm_w = (n_w + r - 1) / r;
    const long long total = n_llm_h * n_llm_w * r * r * vit_dim;
    dim3 grid((unsigned)((total + 255) / 256));
    dsv41v_aligner_pack_bf16_kernel<<<grid, 256, 0, s>>>((const bf16*)x, (bf16*)out, total,
                                                         n_h, n_w, vit_dim, r);
    return cudaGetLastError();
}
