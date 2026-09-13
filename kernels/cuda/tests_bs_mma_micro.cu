// =============================================================================
// tests_bs_mma_micro.cu — ISOLATED micro-test for the tcgen05 block-scaled
// mxf8f6f4 MMA driven by kernels/cuda/tilelang_gen/moe_bs_handwritten.cu.
//
// WHAT IT ISOLATES
// -----------------------------------------------------------------------------
// ONE CTA (128 threads) and exactly ONE k-iteration of the production kernel:
//     M = 128, N = 128, K = 128        (4 x K-block of 32 == scale_vec::1X granularity)
//     A : e4m3,  1 byte/element, [128][128]
//     B : e2m1 UNPACKED, 1 byte/element, [128][128] (low nibble = even K index)
//     SFA/SFB: one uint32 per row, byte j = ue8m0 scale of K-block j (j = 0..3)
//     C[m][n] = sum_k e4m3(A[m][k]) * e2m1(B[n][k]) * sa[m][k/32] * sb[n][k/32]
// The reference is computed in double on the HOST, from VARYING data (LCG-drawn
// e4m3/e2m1 codes + powers-of-two scales that DIFFER per 32-block), so a smem
// layout / descriptor / idesc mistake cannot cancel out (constant data hides it).
//
// WHY 8 CASES
// -----------------------------------------------------------------------------
// case index c:   sf_path = c >> 2,   layout = (c >> 1) & 1,   sv1x = c & 1
//
//   layout 0 = SW128  (production default, g_canon == 0)
//       addr(r,kk) = (r/8)*1024 + (r%8)*128 + (((kk/16) ^ (r%8))*16) + (kk%16)
//       descriptor lbo=1 sbo=64 layout_type=2 ; K-block advance = ki*2   (16B units)
//   layout 1 = canonical (DSV41_MOE_BS_CANON=1)
//       addr(r,kk) = (kk>>5)*4096 + (r>>3)*256 + (r&7)*16 + (((kk&31)>>4)*128) + (kk&15)
//       descriptor lbo=8 sbo=16 layout_type=0 ; K-block advance = ki*256 (16B units)
//   sv1x 0/1  = the MMA asm WITHOUT / WITH the `.scale_vec::1X` suffix
//               (both spellings are compiled into one kernel and selected at
//                runtime, exactly like the production g_sv1x switch)
//   sf_path 0 = the PRODUCTION scale-factor path
//               (warp-2 hw_sf_transpose -> tcgen05.cp.cta_group::1.32x128b.warpx4)
//   sf_path 1 = the VERIFIED path (tests_tcgen05_mxf8f6f4_1x.cu: tcgen05.st
//               32x32b.x4 with the SF duplicated into all four lane partitions)
// Both sf_paths target the SAME TMEM columns (SFA at +0..3, SFB at +4..7 — one
// column per 32-row group, 4 K-block bytes packed in each word, idesc sf_id = ki).
// On sf_path 0 the SF columns are POISONED with a tcgen05.st of zeros immediately
// before the cp, so "the cp silently wrote nothing" cannot be masked by the
// previous case's still-correct TMEM content.
//
// BUILD (compile-only, no GPU — no device is touched by this step):
//   nvcc -c kernels/cuda/tests_bs_mma_micro.cu -o /tmp/bsmma.o \
//        -I$HOME/ferrite/kernels/cuda/tilelang_gen \
//        -I$HOME/ferrite/kernels/cuda/tilelang_inc -arch=sm_103a
//   (the -I flags are not actually needed — this file includes nothing from the
//    tree — they are passed to keep the command identical to the task's.)
// RUN (GPU — single explicit card, ~0.6 MB of device memory, ONE launch):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//        -o /tmp/bsmma kernels/cuda/tests_bs_mma_micro.cu && \
//   CUDA_VISIBLE_DEVICES=6 /tmp/bsmma
//
// NOTE on -arch: tests_tcgen05_mxf8f6f4_1x.cu lines 22-31 warn that with SOME
// nvcc builds `-arch=sm_103a` alone is silently dropped to sm_103 and every
// tcgen05 instruction is then rejected. If ptxas ever complains about a tcgen05
// instruction, use the explicit `-gencode arch=compute_103a,code=sm_103a` form.
// =============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------- geometry
constexpr int BM = 128;          // M tile
constexpr int BN = 128;          // N tile
constexpr int BK = 128;          // K per k-iteration
constexpr int NKB = BK / 32;     // 4 K-blocks of 32 (scale_vec::1X granularity)
constexpr int NCASE = 8;         // 2 layouts x 2 spellings x 2 SF paths
constexpr int TOL_EXP = 3;       // PASS if max|diff|/max(1,|ref|) < 1e-3

constexpr int LAYOUT_SW128 = 0;
constexpr int LAYOUT_CANON = 1;

// =============================================================================
// device primitives — copied from tilelang_gen/moe_bs_handwritten.cu (which in
// turn copied them from the round-trip-verified tests_tcgen05_mxf8f6f4_1x.cu),
// with the two runtime globals (g_canon / g_sv1x) turned into plain arguments.
// =============================================================================

__device__ __forceinline__ void hw_tc_alloc(uint32_t *dst, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "r"(ncols));
}

__device__ __forceinline__ void hw_tc_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void hw_tc_dealloc(uint32_t tmem, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(ncols));
}

// smem index for element (row, kk) of a 128-row x 128-K operand tile, in the
// family selected by `layout` (byte index into the 16384-B tile).
__device__ __forceinline__ int hw_smem_idx(int row, int kk, int layout) {
    if (layout == LAYOUT_CANON) {
        // canonical UMMA K-major interleave, SWIZZLE_NONE:
        //   unit16(row,kb) = (row%8) + 8*kb + 16*(row/8), atom(K=32) = 4096 B
        return (kk >> 5) * 4096 + (row >> 3) * 256 + (row & 7) * 16 +
               (((kk & 31) >> 4) * 128) + (kk & 15);
    }
    // SW128 (CU_TENSOR_MAP_SWIZZLE_128B):
    //   addr(r,c) = (r/8)*1024 + (r%8)*128 + (((c/16) ^ (r%8))*16) + (c%16)
    return (row >> 3) * 1024 + (row & 7) * 128 + ((((kk >> 4) ^ (row & 7))) << 4) + (kk & 15);
}

// The MMA under test. Both spellings are compiled in; sv1x picks at runtime
// (production uses one compiled variant selected by its g_sv1x global).
// Operand order: %0 d_tmem, %1 a_desc, %2 b_desc, %3 idesc, %4 sfa_tmem,
//                %5 sfb_tmem, %6 enable_input_d.
__device__ __forceinline__ void hw_tc_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                          uint32_t idesc, uint32_t sfa_tmem,
                                          uint32_t sfb_tmem, uint32_t enable_d, int sv1x) {
    if (sv1x) {
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t}"
            ::"r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
              "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
            : "memory");
    } else {
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t}"
            ::"r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
              "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
            : "memory");
    }
}

__device__ __forceinline__ void hw_tc_commit(void *mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(mbar)) : "memory");
}

// SMEM descriptor for tcgen05 MMA. NOTE: the LBO/SBO arguments are in 16-byte
// UNITS (unlike tl::make_desc which takes bytes and shifts) — same convention as
// the production kernel's hw_make_desc.
// layout: start_addr[0:14) | lbo[16:30) | sbo[32:46) | version=1[46] | layout[61:64)
__device__ __forceinline__ uint64_t hw_make_desc(const void *smem_ptr, uint32_t lbo_16B,
                                                 uint32_t sbo_16B, uint32_t layout) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);      // start_address (16B units)
    d |= (uint64_t)(lbo_16B & 0x3FFF) << 16;    // leading_byte_offset
    d |= (uint64_t)(sbo_16B & 0x3FFF) << 32;    // stride_byte_offset
    d |= (uint64_t)1 << 46;                     // version = SM100
    d |= (uint64_t)(layout & 0x7) << 61;        // layout_type
    return d;
}

// IDESC for mxf8f6f4 block-scaled MMA. Same bit map as
// tests_tcgen05_mxf8f6f4_1x.cu make_idesc_mxf8f6f4 and the constant 144708608
// (0x08A01400) emitted by TileLang for M=128/N=128.
// layout: b_sf[4:6) | a_fmt[7:10) | b_fmt[10:13) | n_dim[17:23) | sf_fmt[23]
//         | m_dim[24:29) | a_sf[29:31)   (k_size[31] = 0 for dense K32)
__device__ __forceinline__ uint32_t hw_make_idesc(int m, int n, int a_fmt, int b_fmt,
                                                  int sf_id) {
    uint32_t d = 0;
    d |= (uint32_t)(sf_id & 3) << 4;            // b_sf_id
    d |= (uint32_t)(a_fmt & 7) << 7;            // a_format (0 = E4M3)
    d |= (uint32_t)(b_fmt & 7) << 10;           // b_format (5 = E2M1)
    d |= (uint32_t)((n >> 3) & 63) << 17;       // n_dim = N/8
    d |= (uint32_t)1 << 23;                     // scale_format = UE8M0
    d |= (uint32_t)((m >> 4) & 31) << 24;       // m_dim = M/16
    d |= (uint32_t)(sf_id & 3) << 29;           // a_sf_id
    return d;
}

// SF in-place transpose (from TileLang tcgen05_sf_warp_transpose): the 4x32
// uint32 block becomes 32 lanes x 4 words, the layout tcgen05.cp.32x128b wants.
// MUST be executed by exactly ONE warp (all four warps running it on the same
// buffer would race -- the production kernel restricts it to warp 2).
__device__ __forceinline__ void hw_sf_transpose(uint32_t *smem_ptr) {
    const uint32_t lane = threadIdx.x % 32;
    uint32_t values[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) values[i] = smem_ptr[(i ^ (lane >> 3)) * 32 + lane];
    __syncwarp();
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) smem_ptr[lane * 4 + (i ^ (lane >> 3))] = values[i];
}

// SF copy to TMEM (TileLang tcgen05_cp — 32x128b.warpx4 shape).
__device__ __forceinline__ void hw_tc_cp(uint64_t smem_desc, uint32_t tmem_col) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_col), "l"(smem_desc));
}

// SF smem descriptor (TileLang make_sf_smem_desc: SBO>>4 = 8, version 1, layout 0).
__device__ __forceinline__ uint64_t hw_make_sf_desc(void *smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= (uint64_t)(addr >> 4) & 0x3FFF;     // start_address
    desc |= (uint64_t)8u << 32;                 // stride_byte_offset >> 4 = 8
    desc |= (uint64_t)1u << 46;                 // version = 1
    return desc;
}

// ---- TMEM access helpers (verbatim shapes from the verified probe) ----
__device__ __forceinline__ void tc_st_x4(uint32_t taddr, uint32_t w0, uint32_t w1, uint32_t w2,
                                         uint32_t w3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr),
                 "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                 : "memory");
}

__device__ __forceinline__ void tc_ld_x8(uint32_t taddr, uint32_t *v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7])
        : "r"(taddr)
        : "memory");
}

__device__ __forceinline__ void tc_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(bar)),
                 "r"(cnt)
                 : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t phase) {
    asm volatile(
        "{\n\t.reg .pred p;\n"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_%=;\n\t}" ::"r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(phase)
        : "memory");
}

// =============================================================================
// The kernel: NCASE sequential cases, one MMA chain (4 x K32) each.
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 1) bs_mma_micro_kernel(
    const uint8_t *__restrict__ A,     // [128][128] e4m3 codes
    const uint8_t *__restrict__ B,     // [128][128] e2m1 codes (unpacked, low nibble)
    const uint32_t *__restrict__ SFA,  // [128] byte j = ue8m0 of K-block j
    const uint32_t *__restrict__ SFB,  // [128]
    float *__restrict__ D,             // [NCASE][128][128]
    uint64_t *__restrict__ dbg)        // [NCASE][4]
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(1024) uint8_t s_a[BM * BK];  // 16384 B e4m3 tile
    __shared__ __align__(1024) uint8_t s_b[BN * BK];  // 16384 B e2m1(unpacked) tile
    __shared__ __align__(128) uint32_t s_sfa[128];    // production SF staging (+ transpose)
    __shared__ __align__(128) uint32_t s_sfb[128];
    __shared__ __align__(8) uint64_t s_mbar;
    __shared__ uint32_t s_tmem_d;
    __shared__ uint32_t s_tmem_sf;

    if (warp == 0) {
        hw_tc_alloc(&s_tmem_d, 128);   // D: 128 columns (M=128 lanes x N=128 cols)
        hw_tc_alloc(&s_tmem_sf, 32);   // SF: 8 columns used (SFA +0..3, SFB +4..7)
        hw_tc_relinquish();
    }
    if (tid == 0) mbar_init(&s_mbar, 1);
    __syncthreads();
    const uint32_t D_tmem = s_tmem_d;
    const uint32_t SF_tmem = s_tmem_sf;

    for (int c = 0; c < NCASE; ++c) {
        const int sf_path = c >> 2;          // 0 = cp(prod), 1 = st(verified)
        const int layout = (c >> 1) & 1;     // 0 = SW128, 1 = canonical
        const int sv1x = c & 1;              // 0 = no suffix, 1 = .scale_vec::1X

        // ---- (1) fill A and B in this family's smem layout (same data both times)
        for (int i = tid; i < BM * BK; i += 128) {
            const int m = i >> 7, kk = i & 127;
            s_a[hw_smem_idx(m, kk, layout)] = A[m * BK + kk];
        }
        for (int i = tid; i < BN * BK; i += 128) {
            const int n = i >> 7, kk = i & 127;
            s_b[hw_smem_idx(n, kk, layout)] = B[n * BK + kk];
        }
        // SF smem staging (source of the cp path; also the linear-order source the
        // st path reads straight from global — kept for symmetry/printing only).
        if (tid < 128) {
            s_sfa[tid] = SFA[tid];
            s_sfb[tid] = SFB[tid];
        }
        __syncthreads();

        // ---- (2) scale factors -> TMEM columns SF_tmem+0..3 (SFA) / +4..7 (SFB)
        if (sf_path == 0) {
            // production path: poison first, then warp-2 transpose + tcgen05.cp
            for (int r = 0; r < 2; ++r) {
                const uint32_t z = 0u;
                tc_st_x4(((uint32_t)(warp * 32) << 16) | (SF_tmem + 4 * r), z, z, z, z);
            }
            tc_wait_st();
            tc_fence_before_thread_sync();
            __syncthreads();
            tc_fence_after_thread_sync();

            if (warp == 2) {  // exactly one warp (an all-warp transpose would race)
                hw_sf_transpose(s_sfa);
                hw_sf_transpose(s_sfb);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (warp == 1 && lane == 0) {  // one elected thread, as production does
                hw_tc_cp(hw_make_sf_desc(s_sfa), SF_tmem + 0);
                hw_tc_cp(hw_make_sf_desc(s_sfb), SF_tmem + 4);
            }
        } else {
            // verified path: tcgen05.st with the SF word duplicated into all four
            // lane partitions; register j -> column base+j (row group j).
            uint32_t wa[4], wb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wa[j] = SFA[32 * j + lane];
                wb[j] = SFB[32 * j + lane];
            }
            tc_st_x4(((uint32_t)(warp * 32) << 16) | (SF_tmem + 0), wa[0], wa[1], wa[2], wa[3]);
            tc_st_x4(((uint32_t)(warp * 32) << 16) | (SF_tmem + 4), wb[0], wb[1], wb[2], wb[3]);
            tc_wait_st();
        }

        // ---- (3) make the generic smem stores (A/B, and the transposed SF)
        //          visible to the ASYNC PROXY the MMA/UTCCP read through
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        tc_fence_before_thread_sync();
        __syncthreads();
        tc_fence_after_thread_sync();

        // ---- (4) the MMA chain: 4 x K32, one per scale-vector block
        if (warp == 1 && lane == 0) {
            const bool sw128 = (layout == LAYOUT_SW128);
            const uint64_t a_desc_base = sw128
                                             ? hw_make_desc(s_a, 1, 64, 2)   // TileLang SW128
                                             : hw_make_desc(s_a, 8, 16, 0);  // canonical
            const uint64_t b_desc_base = sw128
                                             ? hw_make_desc(s_b, 1, 64, 2)
                                             : hw_make_desc(s_b, 8, 16, 0);
            dbg[c * 4 + 0] = a_desc_base;
            dbg[c * 4 + 1] = b_desc_base;
            for (int ki = 0; ki < NKB; ++ki) {
                const uint32_t idesc = hw_make_idesc(BM, BN, /*a_fmt=*/0, /*b_fmt=*/5, ki);
                // K-block descriptor advance (16B units):
                //   canonical: 4096 B per K32 atom -> 256 units
                //   SW128    : 32 B per K32 block   -> 2 units
                //   (TileLang's `desc + (ki*32)` is in BYTES and its operator+
                //    shifts right by 4, i.e. the same ki*2 units.)
                const uint64_t a_desc =
                    a_desc_base + (uint64_t)(ki * ((layout == LAYOUT_CANON) ? 256 : 2));
                const uint64_t b_desc =
                    b_desc_base + (uint64_t)(ki * ((layout == LAYOUT_CANON) ? 256 : 2));
                const uint32_t enable_d = (ki == 0) ? 0u : 1u;  // ki=0 clears the acc
                if (ki == 0) dbg[c * 4 + 2] = idesc;
                if (ki == NKB - 1) dbg[c * 4 + 3] = idesc;
                hw_tc_mma(D_tmem, a_desc, b_desc, idesc, SF_tmem + 0, SF_tmem + 4, enable_d, sv1x);
            }
            hw_tc_commit(&s_mbar);
        }
        // one commit per case -> the mbarrier parity alternates per case
        mbar_wait(&s_mbar, (uint32_t)(c & 1));

        // ---- (5) read D back: D[m][n] lives at TMEM lane m, column D_tmem+n
        tc_fence_before_thread_sync();
        __syncthreads();
        tc_fence_after_thread_sync();
        {
            const int row = warp * 32 + lane;
            for (int q = 0; q < 16; ++q) {
                uint32_t v[8];
                tc_ld_x8(((uint32_t)(warp * 32) << 16) | (D_tmem + 8 * q), v);
                tc_wait_ld();
                for (int i = 0; i < 8; ++i)
                    D[((size_t)c * BM + row) * BN + 8 * q + i] = __uint_as_float(v[i]);
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        hw_tc_dealloc(D_tmem, 128);
        hw_tc_dealloc(SF_tmem, 32);
    }
}

// =============================================================================
// host side: data generation, exact double reference, comparison
// =============================================================================
namespace {

uint32_t g_rng = 0x12345678u;
uint32_t xrand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

// quant.rs FP4_TABLE (e2m1 code -> value), indexed by the low nibble.
const float kTabE2M1[16] = {0.f,  0.5f, 1.f,  1.5f, 2.f,  3.f,  4.f,  6.f,
                            0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// e4m3 decode, exact (S EEEE MMM).
double e4m3_to_d(uint8_t b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    double v = (e == 0) ? std::ldexp((double)m / 8.0, -6) : std::ldexp(1.0 + (double)m / 8.0, e - 7);
    return s ? -v : v;
}

// e4m3 encode (round-to-nearest, saturating at 448) — same convention as
// __nv_cvt_float_to_fp8(__NV_SATFINITE) and ph0::e4m3_encode.
uint8_t e4m3_encode(float x) {
    const uint32_t sign = (x < 0.f) ? 0x80u : 0u;
    const float a = std::fabs(x);
    if (!(a <= 448.f)) return (uint8_t)(sign | 0x7Eu);
    int efield, mant;
    if (a < 0.015625f) {
        mant = (int)std::lrint(a * 512.0f);
        efield = 0;
        if (mant >= 8) { efield = 1; mant = 0; }
    } else {
        int e;
        std::frexp(a, &e);
        int E = e - 1;
        int m8 = (int)std::lrint(std::ldexp(a, 3 - E));
        if (m8 >= 16) { ++E; m8 = 8; }
        efield = E + 7;
        mant = m8 - 8;
        if (E > 8) { efield = 15; mant = 6; }
    }
    return (uint8_t)(sign | (uint32_t)(efield << 3) | (uint32_t)mant);
}

struct CaseInfo {
    const char *layout;
    const char *sv;
    const char *sfpath;
};

// case index c: sf_path = c>>2, layout = (c>>1)&1, sv1x = c&1  (must match kernel)
const CaseInfo kCases[NCASE] = {
    {"SW128",     "plain", "cp(prod)"},
    {"SW128",     "::1X",  "cp(prod)"},
    {"canonical", "plain", "cp(prod)"},
    {"canonical", "::1X",  "cp(prod)"},
    {"SW128",     "plain", "st(verified)"},
    {"SW128",     "::1X",  "st(verified)"},
    {"canonical", "plain", "st(verified)"},
    {"canonical", "::1X",  "st(verified)"},
};

// probe points (m, n) — chosen to separate an M-dimension error from an
// N-dimension error from a uniform (K/descriptor/SF) error.
const int kProbes[][2] = {{0, 0}, {1, 0}, {0, 1}, {2, 3}, {32, 0}, {0, 32}, {64, 64}, {127, 127}};
const int kNumProbes = (int)(sizeof(kProbes) / sizeof(kProbes[0]));

}  // namespace

int main(int argc, char **argv) {
    const int only = (argc > 1) ? std::atoi(argv[1]) : -1;  // optional: single case

    // ---- host data: varying codes + powers-of-two scales that differ per block
    std::vector<uint8_t> a((size_t)BM * BK), b((size_t)BN * BK);
    std::vector<double> av((size_t)BM * BK), bv((size_t)BN * BK);
    std::vector<uint32_t> sfa(BM, 0), sfb(BN, 0);
    std::vector<int> sae((size_t)BM * NKB), sbe((size_t)BN * NKB);

    g_rng = 0x12345678u;
    int enc_mismatch = 0;
    for (int m = 0; m < BM; ++m) {
        for (int k = 0; k < BK; ++k) {
            // exactly representable in e4m3: multiples of 0.25 in [-3, 3]
            const int t = (int)(xrand() % 25u) - 12;      // -12..12
            const float v = (float)t * 0.25f;
            const uint8_t code = e4m3_encode(v);
            a[(size_t)m * BK + k] = code;
            av[(size_t)m * BK + k] = e4m3_to_d(code);     // reference uses the DECODED value
            if (e4m3_to_d(code) != (double)v) ++enc_mismatch;
        }
    }
    for (int n = 0; n < BN; ++n) {
        for (int k = 0; k < BK; ++k) {
            const uint8_t code = (uint8_t)(xrand() & 0xF);
            b[(size_t)n * BK + k] = code;
            bv[(size_t)n * BK + k] = (double)kTabE2M1[code];
        }
    }
    for (int m = 0; m < BM; ++m)
        for (int j = 0; j < NKB; ++j) {
            sae[(size_t)m * NKB + j] = (int)((m * 7 + j * 5) % 7) - 3;  // -3..3, distinct per j
            sfa[m] |= (uint32_t)(uint8_t)(sae[(size_t)m * NKB + j] + 127) << (8 * j);
        }
    for (int n = 0; n < BN; ++n)
        for (int j = 0; j < NKB; ++j) {
            sbe[(size_t)n * NKB + j] = (int)((n * 11 + j * 13) % 7) - 3;
            sfb[n] |= (uint32_t)(uint8_t)(sbe[(size_t)n * NKB + j] + 127) << (8 * j);
        }

    // ---- double reference
    std::vector<double> ref((size_t)BM * BN, 0.0);
    for (int m = 0; m < BM; ++m)
        for (int n = 0; n < BN; ++n) {
            double acc = 0.0;
            for (int k = 0; k < BK; ++k) {
                const int j = k >> 5;
                const double sa = std::ldexp(1.0, sae[(size_t)m * NKB + j]);
                const double sb = std::ldexp(1.0, sbe[(size_t)n * NKB + j]);
                acc += av[(size_t)m * BK + k] * bv[(size_t)n * BK + k] * sa * sb;
            }
            ref[(size_t)m * BN + n] = acc;
        }

    double maxabs_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) maxabs_ref = std::fmax(maxabs_ref, std::fabs(ref[i]));

    printf("== tcgen05 mxf8f6f4.block_scale micro-test ==\n");
    printf("   M=%d N=%d K=%d (%d K-blocks of 32), one CTA of 128 threads, one launch\n", BM, BN,
           BK, NKB);
    printf("   A=e4m3 1B/elem, B=e2m1-unpacked 1B/elem, SFA/SFB = uint32/row (byte j = block j)\n");
    printf("   data: LCG codes (varying with m,k / n,k), scales 2^-3..2^3 distinct per K-block\n");
    printf("   e4m3 encode/decode non-exact elements: %d (0 expected)\n", enc_mismatch);
    printf("   CPU reference in double, max|ref| = %.6g\n", maxabs_ref);
    printf("   PASS criterion: max|diff| / max(1,|ref|) < 1e-%d\n\n", TOL_EXP);
    fflush(stdout);

    // ---- device
    uint8_t *dA = nullptr, *dB = nullptr;
    uint32_t *dSFA = nullptr, *dSFB = nullptr;
    float *dD = nullptr;
    uint64_t *ddbg = nullptr;
    const size_t dbytes = (size_t)NCASE * BM * BN * sizeof(float);
    if (cudaMalloc(&dA, a.size()) != cudaSuccess || cudaMalloc(&dB, b.size()) != cudaSuccess ||
        cudaMalloc(&dSFA, sfa.size() * 4) != cudaSuccess ||
        cudaMalloc(&dSFB, sfb.size() * 4) != cudaSuccess || cudaMalloc(&dD, dbytes) != cudaSuccess ||
        cudaMalloc(&ddbg, NCASE * 4 * sizeof(uint64_t)) != cudaSuccess) {
        printf("[FAIL] cudaMalloc: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 2;
    }
    cudaMemcpy(dA, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), b.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA, sfa.data(), sfa.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB, sfb.data(), sfb.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(dD, 0, dbytes);
    cudaMemset(ddbg, 0, NCASE * 4 * sizeof(uint64_t));

    bs_mma_micro_kernel<<<1, 128>>>(dA, dB, dSFA, dSFB, dD, ddbg);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[FAIL] launch/exec: %s\n", cudaGetErrorString(err));
        return 2;
    }

    std::vector<float> got(dbytes / sizeof(float));
    std::vector<uint64_t> hdbg(NCASE * 4, 0);
    cudaMemcpy(got.data(), dD, dbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdbg.data(), ddbg, NCASE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // ---- compare
    int fails = 0;
    double worst_global = 0.0;
    for (int c = 0; c < NCASE; ++c) {
        if (only >= 0 && c != only) continue;
        const float *g = got.data() + (size_t)c * BM * BN;
        double maxdiff = 0.0, rabs = 0.0, rrel = 0.0;
        size_t first_bad = (size_t)-1, worst = 0;
        long nbad = 0;
        std::vector<char> badrow(BM, 0), badcol(BN, 0);
        for (int m = 0; m < BM; ++m)
            for (int n = 0; n < BN; ++n) {
                const size_t i = (size_t)m * BN + n;
                const double d = std::fabs((double)g[i] - ref[i]);
                const double sc = d / std::fmax(1.0, std::fabs(ref[i]));
                if (d > maxdiff) { maxdiff = d; worst = i; }
                if (sc > rabs) rabs = sc;
                if (std::fabs(ref[i]) > 1e-30) rrel = std::fmax(rrel, d / std::fabs(ref[i]));
                if (sc > std::pow(10.0, -TOL_EXP)) {
                    ++nbad;
                    badrow[m] = 1;
                    badcol[n] = 1;
                    if (first_bad == (size_t)-1) first_bad = i;
                }
            }
        int nbr = 0, nbc = 0;
        for (int m = 0; m < BM; ++m) nbr += badrow[m];
        for (int n = 0; n < BN; ++n) nbc += badcol[n];

        printf("---- case %d: layout=%-9s scale_vec=%s sf_path=%s\n", c, kCases[c].layout,
               kCases[c].sv, kCases[c].sfpath);
        printf("     a_desc0 = 0x%016llx  b_desc0 = 0x%016llx\n",
               (unsigned long long)hdbg[c * 4 + 0], (unsigned long long)hdbg[c * 4 + 1]);
        printf("     idesc(ki=0) = 0x%08llx  idesc(ki=3) = 0x%08llx   (expected base 0x%08x)\n",
               (unsigned long long)hdbg[c * 4 + 2], (unsigned long long)hdbg[c * 4 + 3],
               (unsigned)((16u << 17) | (1u << 23) | (8u << 24) | (5u << 10)));
        printf("     max|diff|=%.4e  relerr(scaled)=%.4e  relerr(raw)=%.4e  bad=%ld/16384  "
               "bad_rows=%d/128 bad_cols=%d/128\n",
               maxdiff, rabs, rrel, nbad, nbr, nbc);
        printf("     worst at (m=%d,n=%d): mma=%.6f ref=%.6f\n", (int)(worst / BN),
               (int)(worst % BN), (double)g[worst], ref[worst]);
        for (int p = 0; p < kNumProbes; ++p) {
            const int m = kProbes[p][0], n = kProbes[p][1];
            const size_t i = (size_t)m * BN + n;
            printf("       probe (m=%3d,n=%3d) mma=%14.6f ref=%14.6f diff=%11.4e\n", m, n,
                   (double)g[i], ref[i], std::fabs((double)g[i] - ref[i]));
        }
        const bool ok = (rabs < std::pow(10.0, -TOL_EXP));
        printf("     %s\n\n", ok ? "[PASS]" : "[FAIL]");
        if (!ok) ++fails;
        worst_global = std::fmax(worst_global, rabs);
        fflush(stdout);
    }

    printf("=== SUMMARY (tol = 1e-%d * max(1,|ref|)) ===\n", TOL_EXP);
    printf("%-4s %-10s %-4s %-14s %-12s %s\n", "case", "layout", "sv", "sf_path", "err(scaled)",
           "verdict");
    for (int c = 0; c < NCASE; ++c) {
        if (only >= 0 && c != only) continue;
        const float *g = got.data() + (size_t)c * BM * BN;
        double rabs = 0.0;
        for (int i = 0; i < BM * BN; ++i)
            rabs = std::fmax(rabs, std::fabs((double)g[i] - ref[i]) / std::fmax(1.0, std::fabs(ref[i])));
        printf("%-4d %-10s %-4s %-14s %-12.4e %s\n", c, kCases[c].layout, kCases[c].sv,
               kCases[c].sfpath, rabs, (rabs < std::pow(10.0, -TOL_EXP)) ? "PASS" : "FAIL");
    }
    printf("worst scaled error over all cases = %.4e\n", worst_global);
    printf("%s (%d failing case(s))\n", fails ? "[OVERALL FAIL]" : "[OVERALL PASS]", fails);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dSFA);
    cudaFree(dSFB);
    cudaFree(dD);
    cudaFree(ddbg);
    return fails ? 1 : 0;
}
