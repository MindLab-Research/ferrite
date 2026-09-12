// =============================================================================
// tcgen05_probe.cu — ptxas + layout probe for the sm_103a mixed fp8 x fp4
// block-scaled tcgen05 MMA:
//
//   tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
//       [d_tmem], a_desc, b_desc, idesc, [scale_a_tmem], [scale_b_tmem], p;
//
// WHY THIS FILE EXISTS (expert-tcgen05 pre-study, 0.5-day first step)
// ---------------------------------------------------------------------------
// The pre-study (docs/agent/perf-roadmap.md, "expert 侧 tcgen05 fp4 swapAB
// 预研") concluded from headers + CUTLASS + DeepGEMM reading only (the analysis
// host had no nvcc) that this spelling exists on sm_103a and that the swapAB
// mapping (A = fp4 e2m1 weight, B = e4m3 activation) is correct. This file turns
// those claims into facts:
//   1. does ptxas ASSEMBLE the instruction for sm_103a?      -> see STATUS below
//   2. is the 64-bit SMEM descriptor built the way the hardware wants?
//   3. do tcgen05.alloc / tcgen05.st / tcgen05.ld round-trip?
//   4. does D match a CPU reference (i.e. is the operand LAYOUT right)?
//
// STATUS (2026-09-12, checked locally with the CUDA 13.3 toolkit found at
// /tmp/nvccx/nvidia/cu13 — this host has nvcc+ptxas but NO GPU):
//   (1) VERIFIED. `nvcc -gencode arch=compute_103a,code=sm_103a` accepts the
//       file; the emitted PTX carries
//         .target sm_103a
//         tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
//             [%r182], %rd19, %rd20, %r183, [%r184], [%r185], p;
//       and `ptxas -arch=sm_103a` assembles it clean (ptxas -v: 30 registers,
//       4364 bytes smem). NOTE: `-arch=sm_103a` alone is NOT enough with this
//       nvcc — its `--list-gpu-arch` has no sm_103a entry, so it silently drops
//       the suffix and targets sm_103, where every tcgen05 instruction is
//       rejected. Use the explicit `-gencode arch=compute_103a,code=sm_103a`.
//   (2)(3)(4) NEED A GPU. No sm_103a device on this host, so the numeric check
//       (descriptor correctness + scale/TMEM mapping) is still open. Run it on a
//       B300:
//         nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//              -o /tmp/tcgen05_probe /tmp/tcgen05_probe.cu && /tmp/tcgen05_probe
//
// WHAT IS ALREADY PROVEN AND IS THEREFORE REUSED VERBATIM
// ---------------------------------------------------------------------------
// The descriptor bit layout, the TMEM row/column mapping and the SF-word
// mechanics below are the ones that kernels/cuda/dsv41_experts_mxf4.cu already
// validates numerically for tcgen05.mma.cta_group::1.kind::mxf4
// .block_scale.scale_vec::2X (see its header comment lines 24-70 and the
// self-test tests_tcgen05_mxf4.cu). Only what genuinely differs for the
// mxf8f6f4 path is changed, and every change is marked [CHANGED].
//
// THE ONE LAYOUT FACT THAT IS EASY TO GET WRONG
// ---------------------------------------------------------------------------
// [CHANGED] In the mxf8f6f4 path the fp4 operand is UNPACKED in smem: one e2m1
// element per BYTE (low nibble used). Evidence:
//   * CUTLASS cute/arch/mma_sm100_desc.hpp: `float_e2m1_unpacksmem_t` ->
//     MXF8F6F4Format::E2M1, while the packed `float_e2m1_t` -> MXF4Format::E2M1.
//     I.e. only kind::mxf4 consumes packed nibbles; mxf8f6f4 wants the unpacked
//     ("unpacksmem") representation.
//   * DeepGEMM include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:86,87 sizes
//     A/B smem as LOAD_BLOCK_M*BLOCK_K*sizeof(a_dtype_t) with a 1-byte dtype,
//     i.e. one element per byte.
// Consequence for K=32: A/B are 32 BYTES per row = 2 x 16-byte chunks, which is
// exactly the canonical UMMA Major-K SWIZZLE_NONE atom
//     ((8,n),(2,1)) : ((1,SBO),LBO)        [units of uint128_t]
// (cute/atom/mma_traits_sm100.hpp:275). A packed fp4 A would be 16 B/row and
// would NOT fit that canonical form. So A and B share the same strides:
//     LBO = 8 u128 = 128 B   (stride between the atom's two K-chunks)
//     SBO = 16 u128 = 256 B  (stride between 8-row groups)
// If the remote numeric check fails, this packing assumption is suspect #1:
// flip A to packed nibbles (16 B/row, K-chunks = 1, LBO unused) and re-run.
//
// [CHANGED] scale_vec::1X = one e8m0 scale per 32-element K-block (mxf8f6f4 has
// no 2X/4X). A 32-bit TMEM SF word holds 4 consecutive blocks' bytes and the
// idesc 2-bit SFA_ID/SFB_ID selects which byte. Ground truth: DeepGEMM
// sm100_fp8_fp4_gemm_1d1d.cuh sizes SF smem as rows*sizeof(uint32_t) and uses
// kNumSFAStagesPerLoad = 1 when kGranKA == 32, advancing the TMEM column by 4
// per 128-row group (SM100_UTCCP_4x32dp128bit). This probe has K=32 => exactly
// ONE block => SFA_ID = SFB_ID = 0 => the meaningful byte is byte 0. Suspect #2
// if the numeric check fails: the byte position inside the SF word.
//
// [CHANGED] idesc formats: mxf8f6f4 a_format/b_format use MXF8F6F4Format
// (E4M3=0, E5M2=1, E2M3=3, E3M2=4, E2M1=5) -- NOT the MXF4Format::E2M1 = 1 that
// dsv41_experts_mxf4.cu uses. Here a_format = 5 (E2M1, the weight) and
// b_format = 0 (E4M3, the activation). k_size = 0 = dense K32 for mxf8f6f4
// (whereas kind::mxf4's k_size = 0 means dense K64). All other idesc bits are
// unchanged. Expected idesc for this probe = 0x08820280.
//
// [CHANGED] M is pinned at 128 (CUTLASS SM100_MMA_MXF8F6F4_SS static_asserts
// M == 128 for the 1-CTA form; N must be a multiple of 8 in [8,256]). The brief
// said "A: 16 rows"; in the canonical layout 128 rows is 16 row-groups of 8, so
// A is [128, K] here.
//
// ISOLATION SWITCHES (for "ptxas rejected X" triage)
// ---------------------------------------------------------------------------
//   -DPROBE_MMA_SPELLING=1   emit `...kind::mxf8f6f4.block_scale` with no
//                            explicit `.scale_vec::1X` (the CUTLASS / DeepGEMM
//                            spelling) and the otherwise identical operands, to
//                            tell "the qualifier is the problem" apart from
//                            "the whole instruction is unsupported".
//                            Both spellings assemble on sm_103a (verified).
//   -DPROBE_ASM_ONLY=1       compile ONLY a minimal kernel holding the asm, so a
//                            rejection is attributable to the instruction alone
//                            and not to the surrounding file.
//   -DPROBE_SF_BYTE0_ONLY=1  write the SF byte into byte 0 of the SF word only,
//                            instead of replicating it to all 4 bytes. The
//                            default replication is proof-safe for a single
//                            K-block (whichever byte the SF-id picks carries the
//                            same value) and removes a byte-position ambiguity
//                            from the numeric check.
// =============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// --------------------------------------------------------------------- knobs
#define PROBE_M 128  // pinned by the instruction (CUTLASS static_assert)
#define PROBE_N 8    // minimum legal N for mxf8f6f4 (multiples of 8 in [8,256])
#define PROBE_K 32   // dense K for mxf8f6f4, = one 32-element scale block (1X)

#define PROBE_TMEM_COLS 32  // power of two, >= 32 (tcgen05.alloc requirement)

// TMEM column map: D first (as in dsv41_experts_mxf4.cu), then SFA, then SFB.
//   D   : M=128 -> lanes (m%32) x 4 partitions, N=8 columns.
//   SFA : 4 columns, one per 32-row group of A's M=128 rows.
//   SFB : 4 columns are written (proven x4 store form); at N=8 the hardware
//         reads only the first one (B rows n<32 -> lane n, column +0).
#define PROBE_D_OFF 0
#define PROBE_D_COLS PROBE_N
#define PROBE_SFA_OFF (PROBE_D_OFF + PROBE_D_COLS)      // 8
#define PROBE_SFA_COLS 4
#define PROBE_SFB_OFF (PROBE_SFA_OFF + PROBE_SFA_COLS)  // 12
#define PROBE_SFB_COLS 4

// Canonical Major-K SWIZZLE_NONE strides (identical for A and B here; see the
// header comment). In units of 16 bytes: LBO = 8, SBO = 16.
#define PROBE_LBO_BYTES 128
#define PROBE_SBO_BYTES 256

#define PROBE_A_BYTES (PROBE_M * PROBE_K)  // 4096: 1 byte per fp4 element
#define PROBE_B_BYTES (PROBE_N * PROBE_K)  //  256: 1 byte per e4m3 element

#ifndef PROBE_MMA_SPELLING
#define PROBE_MMA_SPELLING 0
#endif

namespace {

// ------------------------------------------------------------- small helpers
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ------------------------------------------------------- tcgen05 primitives
__device__ __forceinline__ void tc_alloc(uint32_t* dst, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                     smem_addr(dst)),
                 "r"(ncols));
}
__device__ __forceinline__ void tc_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols)
                 : "memory");
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
            smem_addr(bar))
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
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(cnt)
                 : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n\t.reg .pred p;\n"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_%=;\n\t}" ::"r"(smem_addr(bar)),
        "r"(phase)
        : "memory");
}

// ------------------------------------------------------------- the MMA under test
// [CHANGED vs dsv41_experts_mxf4.cu] kind::mxf8f6f4.block_scale.scale_vec::1X
// instead of kind::mxf4.block_scale.scale_vec::2X. The operand list is
// identical: [d_tmem], a_desc, b_desc, idesc, [sf_a_tmem], [sf_b_tmem], pred.
// Operand numbering used by both branches: %0 d_tmem, %1 a_desc, %2 b_desc,
// %3 idesc, %4 sfa_tmem, %5 sfb_tmem, %6 enable_input_d.
__device__ __forceinline__ void tc_mma_mxf8f6f4_1x(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                                   uint32_t idesc, uint32_t sfa_tmem,
                                                   uint32_t sfb_tmem, uint32_t enable_d) {
#if PROBE_MMA_SPELLING == 0
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
        : "memory");
#else
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
        : "memory");
#endif
}

__device__ __forceinline__ void tc_st_x4(uint32_t taddr, uint32_t w0, uint32_t w1, uint32_t w2,
                                         uint32_t w3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr),
                 "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                 : "memory");
}
__device__ __forceinline__ void tc_ld_x8(uint32_t taddr, uint32_t* v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7])
        : "r"(taddr)
        : "memory");
}

// ------------------------------------------------------------- descriptors
// CUTLASS UMMA::SmemDescriptor / DeepGEMM make_smem_desc bit layout:
//   [ 0,14) start_address >> 4
//   [16,30) leading_byte_offset >> 4    (LBO)
//   [32,46) stride_byte_offset  >> 4    (SBO)
//   [46,48) version = 1 (Blackwell)
//   [49,52) base_offset = 0, [52] lbo_mode = 0 (legacy)
//   [61,64) layout_type = 0 (SWIZZLE_NONE / INTERLEAVE)
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_base, uint32_t lbo_bytes,
                                              uint32_t sbo_bytes) {
    const uint64_t start = (uint64_t)((smem_base >> 4) & 0x3FFFu);
    const uint64_t lbo = (uint64_t)((lbo_bytes >> 4) & 0x3FFFu);
    const uint64_t sbo = (uint64_t)((sbo_bytes >> 4) & 0x3FFFu);
    return start | (lbo << 16) | (sbo << 32) | ((uint64_t)1 << 46);  // version = 1
}

// Instruction descriptor, block-scaled form (CUTLASS
// UMMA::InstrDescriptorBlockScaled). [CHANGED] formats come from
// MXF8F6F4Format: E4M3 = 0, E5M2 = 1, E2M3 = 3, E3M2 = 4, E2M1 = 5.
// Bit map: [0,2) sparse_id2 [2] sparse [4,6) b_sf_id [7,10) a_format
// [10,13) b_format [13] a_neg [14] b_neg [15] a_major [16] b_major
// [17,23) n_dim = N>>3 [23] scale_format (1 = UE8M0) [24,29) m_dim = M>>4
// [29,31) a_sf_id [31] k_size (0 = dense K32 for mxf8f6f4)
__device__ __forceinline__ uint32_t make_idesc_mxf8f6f4(uint32_t m_dim, uint32_t n_dim,
                                                        uint32_t a_fmt, uint32_t b_fmt,
                                                        uint32_t a_sf_id, uint32_t b_sf_id,
                                                        uint32_t k_size) {
    uint32_t d = 0;
    d |= (b_sf_id & 0x3u) << 4;
    d |= (a_fmt & 0x7u) << 7;
    d |= (b_fmt & 0x7u) << 10;
    d |= (n_dim & 0x3Fu) << 17;
    d |= 1u << 23;  // scale_format = UE8M0
    d |= (m_dim & 0x1Fu) << 24;
    d |= (a_sf_id & 0x3u) << 29;
    d |= (k_size & 0x1u) << 31;
    return d;
}

}  // namespace

// =============================================================================
#ifdef PROBE_ASM_ONLY
// ---------------------------------------------------------------------------
// Minimal translation unit: ONE kernel holding just the instruction under
// test, so that a ptxas rejection is attributable to the MMA alone.
//   nvcc -gencode arch=compute_103a,code=sm_103a -DPROBE_ASM_ONLY=1 -c file.cu
// ---------------------------------------------------------------------------
namespace {
__global__ void probe_asm_only_kernel(const uint8_t* a, const uint8_t* b) {
    __shared__ uint32_t s_tmem;
    __shared__ __align__(8) uint64_t s_mbar;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    if (warp == 0) {
        tc_alloc(&s_tmem, PROBE_TMEM_COLS);
        tc_relinquish();
    }
    if (tid == 0) mbar_init(&s_mbar, 1);
    __syncthreads();
    if (tid == 0) {
        const uint32_t tb = s_tmem;
        const uint64_t da = make_desc(smem_addr(a), PROBE_LBO_BYTES, PROBE_SBO_BYTES);
        const uint64_t db = make_desc(smem_addr(b), PROBE_LBO_BYTES, PROBE_SBO_BYTES);
        const uint32_t idesc = make_idesc_mxf8f6f4(PROBE_M >> 4, PROBE_N >> 3, 5, 0, 0, 0, 0);
        tc_mma_mxf8f6f4_1x(tb, da, db, idesc, tb + PROBE_SFA_OFF, tb + PROBE_SFB_OFF, 0u);
        tc_commit(&s_mbar);
    }
    __syncthreads();
    if (warp == 0) tc_dealloc(s_tmem, PROBE_TMEM_COLS);
}
}  // namespace

int main() {
    return 0;  // nothing to run: this build only answers "does ptxas take it?"
}

#else  // !PROBE_ASM_ONLY
// =============================================================================
namespace {

// ------------------------------------------------------------------- kernel
__global__ void __launch_bounds__(128) probe_kernel(
    const uint8_t* __restrict__ a,    // [128][32] fp4 e2m1, UNPACKED (1 element/byte)
    const uint8_t* __restrict__ sfa,  // [128] e8m0, one 32-element block per row
    const uint8_t* __restrict__ b,    // [  8][32] e4m3 (1 element/byte)
    const uint8_t* __restrict__ sfb,  // [  8] e8m0
    float* __restrict__ d_out,        // [128][8]
    uint64_t* __restrict__ dbg)       // [4] {a_desc, b_desc, idesc, tmem_base}
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // -------- smem: canonical Major-K SWIZZLE_NONE --------
    //   unit16(r, kb) = (r % 8) + 8*kb + 16*(r / 8)      [16-byte units], kb in {0,1}
    //   A: 128 rows * 32 B = 256 units = 4096 B
    //   B:   8 rows * 32 B =  16 units =  256 B
    __shared__ __align__(1024) uint8_t s_a[PROBE_A_BYTES];
    __shared__ __align__(1024) uint8_t s_b[PROBE_B_BYTES];
    __shared__ __align__(8) uint64_t s_mbar;
    __shared__ uint32_t s_tmem_base;

    // -------- tmem alloc --------
    if (warp == 0) {
        tc_alloc(&s_tmem_base, PROBE_TMEM_COLS);
        tc_relinquish();
    }
    if (tid == 0) mbar_init(&s_mbar, 1);
    __syncthreads();

    const uint32_t tmem_base = s_tmem_base;
    const uint32_t d_col = tmem_base + PROBE_D_OFF;
    const uint32_t sfa_col = tmem_base + PROBE_SFA_OFF;
    const uint32_t sfb_col = tmem_base + PROBE_SFB_OFF;

    // -------- stage A (one row per thread: 128 threads, 128 rows) --------
    for (int m = tid; m < PROBE_M; m += 128) {
        for (int kb = 0; kb < 2; ++kb) {
            const int unit = (m & 7) + 8 * kb + 16 * (m >> 3);
            uint4 v;
            __builtin_memcpy(&v, a + (size_t)m * PROBE_K + kb * 16, 16);
            *reinterpret_cast<uint4*>(s_a + unit * 16) = v;
        }
    }
    // -------- stage B (8 rows * 2 chunks = 16 uint4) --------
    for (int c = tid; c < PROBE_N * 2; c += 128) {
        const int n = c >> 1, kb = c & 1;
        const int unit = (n & 7) + 8 * kb + 16 * (n >> 3);
        uint4 v;
        __builtin_memcpy(&v, b + (size_t)n * PROBE_K + kb * 16, 16);
        *reinterpret_cast<uint4*>(s_b + unit * 16) = v;
    }

    // make the smem writes visible to the async proxy (the MMA)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // -------- stage the scale factors into TMEM --------
    // A: row m -> lane (m%32), column (sfa_col + m/32).
    // [CHANGED] scale_vec::1X + K=32 => one block => the scale byte lives in
    // byte 0 (SF-id 0). The default writes it to all four bytes of the word,
    // which is provably safe while only one K-block exists.
    {
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = 32 * j + lane;
            const uint32_t sf = (uint32_t)sfa[m];
#if PROBE_SF_BYTE0_ONLY
            w[j] = sf;
#else
            w[j] = sf | (sf << 8) | (sf << 16) | (sf << 24);
#endif
        }
        // x4 store: register j -> column sfa_col + j = sfa_col + m/32. Every warp
        // writes the identical content into its own 32-lane partition, which is
        // the PTX requirement that SF be duplicated to all lane partitions.
        tc_st_x4(((uint32_t)(warp * 32) << 16) | sfa_col, w[0], w[1], w[2], w[3]);
    }
    // B: B row n -> lane (n%32), column (sfb_col + n/32) = sfb_col for n < 32.
    {
        uint32_t w0 = 0;
        if (lane < PROBE_N) {
            const uint32_t sf = (uint32_t)sfb[lane];
#if PROBE_SF_BYTE0_ONLY
            w0 = sf;
#else
            w0 = sf | (sf << 8) | (sf << 16) | (sf << 24);
#endif
        }
        tc_st_x4(((uint32_t)(warp * 32) << 16) | sfb_col, w0, 0u, 0u, 0u);
    }
    tc_wait_st();
    tc_fence_before_thread_sync();
    __syncthreads();
    tc_fence_after_thread_sync();

    // -------- issue the MMA --------
    if (tid == 0) {
        const uint64_t da = make_desc(smem_addr(s_a), PROBE_LBO_BYTES, PROBE_SBO_BYTES);
        const uint64_t db = make_desc(smem_addr(s_b), PROBE_LBO_BYTES, PROBE_SBO_BYTES);
        // a_format = 5 (E2M1 weight), b_format = 0 (E4M3 activation),
        // scale_format = UE8M0, a_major = b_major = K, k_size = 0 (dense K32).
        const uint32_t idesc = make_idesc_mxf8f6f4(PROBE_M >> 4, PROBE_N >> 3,
                                                   /*a_fmt=*/5, /*b_fmt=*/0,
                                                   /*a_sf_id=*/0, /*b_sf_id=*/0,
                                                   /*k_size=*/0);
        dbg[0] = da;
        dbg[1] = db;
        dbg[2] = (uint64_t)idesc;
        dbg[3] = (uint64_t)tmem_base;
        // enable_input_d = 0: the accumulator is cleared by this single MMA.
        tc_mma_mxf8f6f4_1x(d_col, da, db, idesc, sfa_col, sfb_col, /*enable_d=*/0);
        tc_commit(&s_mbar);
    }
    mbar_wait(&s_mbar, 0);

    // -------- read D back: D[m][n] at lane (m%32), column (d_col + n) --------
    {
        uint32_t v[PROBE_N];
        tc_ld_x8(((uint32_t)(warp * 32) << 16) | d_col, v);
        tc_wait_ld();
#pragma unroll
        for (int n = 0; n < PROBE_N; ++n) {
            const int m = warp * 32 + lane;
            d_out[(size_t)m * PROBE_N + n] = __uint_as_float(v[n]);
        }
    }

    __syncthreads();
    if (warp == 0) tc_dealloc(tmem_base, PROBE_TMEM_COLS);
}

// ---------------------------------------------------------------------- host
const float kE2M1[16] = {0.f,  0.5f, 1.f,   1.5f,  2.f,  3.f,  4.f,  6.f,
                         0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// e4m3 decode: S EEEE MMM (no infinities; 0x7F/0xFF are NaN and are not generated)
double e4m3_to_d(uint8_t b) {
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    double v;
    if (e == 0)
        v = std::ldexp((double)m / 8.0, -6);  // subnormal
    else
        v = std::ldexp(1.0 + (double)m / 8.0, e - 7);
    return s ? -v : v;
}

uint32_t g_rng = 12345u;
uint32_t xrand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

}  // namespace

// =============================================================================
int main() {
    std::vector<uint8_t> a((size_t)PROBE_M * PROBE_K), sfa(PROBE_M);
    std::vector<uint8_t> b((size_t)PROBE_N * PROBE_K), sfb(PROBE_N);

    // A: random e2m1 codes, one per byte (the low nibble; the high nibble is
    // ignored by the unpacked fp4 layout).
    for (size_t i = 0; i < a.size(); ++i) a[i] = (uint8_t)(xrand() & 0xF);
    // B: random e4m3 bytes, avoiding the two NaN encodings.
    for (size_t i = 0; i < b.size(); ++i) {
        uint8_t v;
        do {
            v = (uint8_t)xrand();
        } while (v == 0x7F || v == 0xFF);
        b[i] = v;
    }
    // scales: small powers of two, e8m0 = e + 127.
    for (int m = 0; m < PROBE_M; ++m) sfa[m] = (uint8_t)(127 + (int)(xrand() % 5) - 2);
    for (int n = 0; n < PROBE_N; ++n) sfb[n] = (uint8_t)(127 + (int)(xrand() % 5) - 2);

    // ---- CPU reference: D[m][n] = sum_k A[m][k]*2^(sfa[m]-127)*B[n][k]*2^(sfb[n]-127)
    std::vector<double> ref((size_t)PROBE_M * PROBE_N, 0.0);
    for (int m = 0; m < PROBE_M; ++m) {
        const double sam = std::ldexp(1.0, (int)sfa[m] - 127);
        for (int n = 0; n < PROBE_N; ++n) {
            const double sbn = std::ldexp(1.0, (int)sfb[n] - 127);
            double acc = 0.0;
            for (int k = 0; k < PROBE_K; ++k)
                acc += (double)kE2M1[a[(size_t)m * PROBE_K + k] & 0xF] *
                       e4m3_to_d(b[(size_t)n * PROBE_K + k]);
            ref[(size_t)m * PROBE_N + n] = acc * sam * sbn;
        }
    }

    uint8_t *da = nullptr, *dsfa = nullptr, *db = nullptr, *dsfb = nullptr;
    float* dd = nullptr;
    uint64_t* ddbg = nullptr;
    cudaMalloc(&da, a.size());
    cudaMalloc(&dsfa, sfa.size());
    cudaMalloc(&db, b.size());
    cudaMalloc(&dsfb, sfb.size());
    cudaMalloc(&dd, ref.size() * sizeof(float));
    cudaMalloc(&ddbg, 4 * sizeof(uint64_t));
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dsfa, sfa.data(), sfa.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dsfb, sfb.data(), sfb.size(), cudaMemcpyHostToDevice);
    cudaMemset(dd, 0, ref.size() * sizeof(float));

    printf("== tcgen05 mxf8f6f4 block_scale.scale_vec::1X probe ==\n");
    printf("   M=%d N=%d K=%d (swapAB: A = fp4 e2m1 weight [M,K], B = e4m3 act [N,K])\n",
           PROBE_M, PROBE_N, PROBE_K);
    printf("   smem layout: K-major SWIZZLE_NONE, LBO=%d B, SBO=%d B; fp4 UNPACKED (1 B/elem)\n",
           PROBE_LBO_BYTES, PROBE_SBO_BYTES);
    printf("   expected idesc = 0x%08X\n\n",
           (unsigned)((5u << 7) | ((PROBE_N >> 3) << 17) | (1u << 23) | ((PROBE_M >> 4) << 24)));
    fflush(stdout);

    probe_kernel<<<1, 128>>>(da, dsfa, db, dsfb, dd, ddbg);
    cudaError_t err = cudaDeviceSynchronize();
    uint64_t hdbg[4] = {0, 0, 0, 0};
    cudaMemcpy(hdbg, ddbg, sizeof(hdbg), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[FAIL] launch/sync: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("   a_desc   = 0x%016llx\n", (unsigned long long)hdbg[0]);
    printf("   b_desc   = 0x%016llx\n", (unsigned long long)hdbg[1]);
    printf("   idesc    = 0x%08llx\n", (unsigned long long)hdbg[2]);
    printf("   tmem_base= 0x%08llx\n\n", (unsigned long long)hdbg[3]);

    std::vector<float> got(ref.size());
    cudaMemcpy(got.data(), dd, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double maxabs = 0.0, maxdiff = 0.0;
    size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < ref.size(); ++i) {
        maxabs = std::fmax(maxabs, std::fabs(ref[i]));
        const double dev = std::fabs((double)got[i] - ref[i]);
        if (dev > maxdiff) maxdiff = dev;
        if (dev > 1e-3 * std::fmax(1.0, std::fabs(ref[i])) && first_bad == (size_t)-1)
            first_bad = i;
    }
    printf("   max|ref|=%.6g  max|diff|=%.6g  (tol = 1e-3 * max(1,|ref|))\n", maxabs, maxdiff);
    printf("   D[0][0..7] =");
    for (int n = 0; n < PROBE_N; ++n) printf(" %10.5f", (double)got[n]);
    printf("\n   ref[0][0..7]=");
    for (int n = 0; n < PROBE_N; ++n) printf(" %10.5f", ref[n]);
    printf("\n");
    if (first_bad != (size_t)-1) {
        const size_t m = first_bad / PROBE_N, n = first_bad % PROBE_N;
        printf("   first mismatch at (m=%zu, n=%zu): got %.6f expect %.6f\n", m, n,
               (double)got[first_bad], ref[first_bad]);
        printf("[FAIL] layout/scale mismatch\n");
    } else {
        printf("[PASS] D matches the CPU reference within tolerance\n");
    }
    printf("        (max|diff| should be tiny rounding only; a large value means the\n"
           "         SMEM descriptor or the TMEM/scale mapping is wrong, not ptxas)\n");

    cudaFree(da);
    cudaFree(dsfa);
    cudaFree(db);
    cudaFree(dsfb);
    cudaFree(dd);
    cudaFree(ddbg);
    return first_bad == (size_t)-1 ? 0 : 1;
}

#endif  // PROBE_ASM_ONLY
