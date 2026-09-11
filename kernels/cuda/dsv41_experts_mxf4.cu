// dsv41_experts_mxf4.cu — DeepSeek-V4.1-Flash fp4 expert GEMMs on tcgen05 (MXFP4).
//
// ============================================================================
// WHY THIS FILE EXISTS (all facts below were verified on sm_103a / CUDA 13.2)
// ============================================================================
//  * The routed experts are fp4 in the checkpoint (58% of the weights) and MUST
//    run on fp4 tensor cores. There is no fp8 expert path anywhere.
//  * ptxas rejects every `mma.sync` fp4 spelling for sm_103a:
//        "Instruction 'mma with FP6/FP4 floating point type' not supported
//         on .target 'sm_103a'"
//    (probed for kind::f8f6f4 on sm_100a/sm_103a/sm_103f; it assembles only on
//    sm_120a, i.e. it does not exist on this part at all).
//  * The only fp4 tensor-core entry on this part is
//        tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
//            [d_tmem], a_desc, b_desc, idesc, [scale_a_tmem], [scale_b_tmem], p;
//    with the scale type FIXED to ue8m0 == the checkpoint's expert scale format
//    (per-row x k-block-32 e8m0), zero conversion. M=128 (1-CTA), N in [8,256]
//    step 8, K=64 fp4 elements per instruction (dense), scale-vector size 32.
//
// ============================================================================
// LAYOUTS (provenance noted; every one of them is validated by the small-matrix
// numerical self-test in tests_tcgen05_mxf4.cu)
// ============================================================================
// 1. SMEM operand (A and B, both K-major), canonical UMMA form. Per MMA atom the
//    operand block is a K-major "interleaved" layout of 16-byte core chunks:
//        unit16(m, kb) = (m % 8) + 8*kb + 16*(m / 8)      [units of 16 bytes]
//    i.e. 8 rows per group with unit stride, the two K-chunks (16 B = 32 fp4
//    elements each -> K=64 per atom) at stride LBO, the next 8-row group at
//    stride SBO. Descriptor fields (CUTLASS UMMA::SmemDescriptor bit layout):
//        bits [ 0,14) start_address >> 4
//        bits [16,30) leading_byte_offset >> 4   (= 8,  the K-chunk stride)
//        bits [32,46) stride_byte_offset  >> 4   (= 16, the 8-row-group stride)
//        bits [46,48) version_ = 1 (Blackwell; CUTLASS make_umma_desc sets this)
//        bits [49,52) base_offset = 0, bit [52] lbo_mode = 0
//        bits [61,64) layout_type = 0 (SWIZZLE_NONE / INTERLEAVE)
//    CUTLASS cross-check: make_umma_desc<Major::K> accepts exactly
//        SWIZZLE_NONE : ((8,n),(2,1)) : ((1,SBO),LBO)   [uint128 units]
//    which is what the constants above encode.
//
// 2. Scale factors in TMEM. The hardware reads, for a 1-CTA M=128 instruction:
//        row m -> lane (m % 32), column (sf_base + m/32), 2 scale bytes starting
//        at the sub-column selected by the 2-bit SFA_ID field of the idesc.
//    Per the PTX ISA figures "Layout of scale factor A matrix with
//    scale_vec::2X/block32 with K=64/K=128" and the B twin:
//        word bytes = [SF0, SF1, SF0, SF1]  (the pair replicated in both
//        half-words; SFA_ID/SFB_ID pick which half - 00 = low, 10 = high),
//        and the scale factors are DUPLICATED to all four 32-lane TMEM
//        partitions (PTX: "Scale factors for A and B matrices need to be
//        duplicated to all 32 lane partitions of tensor memory").
//    This kernel therefore writes the same words from all four warps (each
//    warp owns one 32-lane partition) and always uses SFA_ID = SFB_ID = 0.
//    Atom placement: SFA occupies 4 columns per atom (one per 32-row group;
//    M=128), SFB occupies ceil(N/32) columns per atom; consecutive atoms take
//    consecutive column blocks (sf_mode 0) - see DSV41_SF_ID_ALT below for the
//    alternative packing that shares a column between two atoms via
//    SFA_ID = 0/2 (kept switchable because only the numeric test can settle it).
//
// 3. Instruction descriptor (32-bit, block-scaled form). Bits (CUTLASS
//    UMMA::InstrDescriptorBlockScaled, cross-checked against the PTX ISA
//    "Instruction descriptor" tables):
//        [ 0, 2) sparse_id2 = 0        [ 2, 3) sparse = 0
//        [ 4, 6) b_sf_id               [ 7,10) a_format (1 = E2M1 for mxf4)
//        [10,13) b_format (1)          [13,15) negate a/b = 0
//        [15,16) a_major = 0 (K)       [16,17) b_major = 0 (K)
//        [17,23) n_dim = N >> 3        [23,24) scale_format = 1 (UE8M0)
//        [24,29) m_dim = M >> 4        [29,31) a_sf_id
//        [31,32) k_size = 0 (dense K64)
//
// 4. TMEM: 512 columns per CTA, address = (lane << 16) | column. The fp32
//    accumulator D[m][n] occupies lane m, column d_base + n.
//
// ============================================================================
// ORGANISATION
// ============================================================================
// tcgen05 is a CTA-level op (M=128 here), while a decode step has a handful of
// rows per expert. The kernel is therefore written as a MASKED M=128 tile GEMM
// (the DeepGEMM m_grouped_gemm_nt_masked shape): `m_valid` rows of the tile are
// real, the rest are zero-filled on load and skipped on store. The A row block
// of one tile belongs to ONE expert - the caller passes that expert's W1/W3 (or
// W2) pointers, so the weight traffic is 0.5 byte/param read exactly once.
// The N dimension is tiled in kNTile columns per CTA; a call loops the K
// dimension in stages of up to four K=64 atoms.
//
// NOTE on the multi-expert grouped dispatch: batching several experts into one
// launch needs a per-row expert id (DeepGEMM's `m_indices`) that the current
// `dsv41_expert_*_fp4` ABI in crates/ferrite-dsv41/src/kernels.rs does not
// carry; the tile machinery here is the part that a grouped launcher would
// reuse (one (expert, m-tile, n-tile) job per CTA).
//
// ============================================================================
#include <cuda_runtime.h>
#include <cstdint>

namespace {

// ---------------------------------------------------------------- constants
constexpr int kThreads = 128;      // 4 warps; warp w owns tmem lanes 32w..32w+31
constexpr int kMTile = 128;        // MMA M (1-CTA kind::mxf4 is fixed at 128)
constexpr int kNTile = 64;         // output columns per CTA (N of the MMA)
constexpr int kAtomK = 64;         // fp4 elements consumed by one MMA (dense mxf4)
constexpr int kAtomBytes = kAtomK / 2;   // 32 bytes per row per atom
constexpr int kStageAtoms = 4;     // K atoms staged per smem round (256 elements)
constexpr int kATileBytes = kMTile * kAtomBytes;         // 4096 per atom
constexpr int kBTileBytes = kNTile * kAtomBytes;         // 2048 per atom
constexpr int kABytes = kStageAtoms * kATileBytes;       // 16384
constexpr int kBBytes = kStageAtoms * kBTileBytes;       //  8192
constexpr int kSfaCols = kStageAtoms * (kMTile / 32);    //  4 per atom -> 16
[[maybe_unused]] constexpr int kSfbCols = kStageAtoms * (kNTile / 32);  // 2 per atom -> 8
constexpr int kDCols = kNTile;                           // 64
constexpr int kTmemCols = 128;     // power-of-two alloc >= kDCols + kSfaCols + kSfbCols

// ------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// e8m0 -> f32 (2^(b-127); 0xFF is NaN, mirroring quant.rs).
[[maybe_unused]] __device__ __forceinline__ float ue8m0_to_f(uint8_t b) {
    return __uint_as_float(((uint32_t)b) << 23);
}

// f32 power-of-two -> e8m0 byte (the caller's scales are fast_round_scale
// outputs, i.e. powers of two; a non-power-of-two is truncated to its exponent).
// ue8m0(b) = 2^(b-127), so the byte is the BIASED exponent: e + 127.
__device__ __forceinline__ uint8_t f_pow2_to_ue8m0(float s) {
    if (!(s > 0.f)) return 0;
    int e = (int)((__float_as_uint(s) >> 23) & 0xFFu) - 127;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (uint8_t)(e + 127);
}

// 2^ceil(log2(amax / 6)) — the reference fast_round_scale for fp4 (quant.rs).
__device__ __forceinline__ float fast_round_scale6(float amax) {
    if (!(amax > 0.f)) return __uint_as_float((uint32_t)1 << 23);  // 2^-126
    const float r = amax * (1.0f / 6.0f);
    const uint32_t bits = __float_as_uint(r);
    const int e = (int)((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) ? 1 : 0);
    const int ec = e < -126 ? -126 : (e > 127 ? 127 : e);
    return __uint_as_float((uint32_t)(ec + 127) << 23);
}

// Nearest e2m1 code (magnitudes {0,.5,1,1.5,2,3,4,6}); ties -> smaller magnitude
// (matches quant.rs e2m1_encode, which keeps the first minimum-distance slot).
__device__ __forceinline__ uint8_t e2m1_encode(float v) {
    const float a = fminf(fabsf(v), 6.0f);
    uint8_t c;
    if (a <= 0.25f)      c = 0;
    else if (a <= 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a <= 1.75f) c = 3;
    else if (a <= 2.5f)  c = 4;
    else if (a <= 3.5f)  c = 5;
    else if (a <= 5.0f)  c = 6;
    else                 c = 7;
    return (uint8_t)(c | (v < 0.f ? 8u : 0u));
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

// The MXFP4 MMA. enable_input_d = 0 clears the accumulator (first atom).
__device__ __forceinline__ void tc_mma_mxf4(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                            uint32_t idesc, uint32_t sfa_tmem, uint32_t sfb_tmem,
                                            uint32_t enable_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
        : "memory");
}

__device__ __forceinline__ void tc_st_x4(uint32_t taddr, uint32_t w0, uint32_t w1, uint32_t w2,
                                         uint32_t w3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr),
                 "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                 : "memory");
}
__device__ __forceinline__ void tc_st_x2(uint32_t taddr, uint32_t w0, uint32_t w1) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};" ::"r"(taddr), "r"(w0),
                 "r"(w1)
                 : "memory");
}
__device__ __forceinline__ void tc_ld_x16(uint32_t taddr, uint32_t* v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]),
          "=r"(v[14]), "=r"(v[15])
        : "r"(taddr)
        : "memory");
}

// ------------------------------------------------------------- descriptors
// K-major SWIZZLE_NONE canonical layout, one atom block (16-byte units):
//   unit16(m, kb) = (m % 8) + 8*kb + 16*(m / 8)      LBO = 8, SBO = 16
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_base) {
    const uint64_t start = (uint64_t)((smem_base >> 4) & 0x3FFFu);
    const uint64_t lbo = (uint64_t)8;    // K-chunk stride, 16-byte units
    const uint64_t sbo = (uint64_t)16;   // 8-row-group stride, 16-byte units
    return start | (lbo << 16) | (sbo << 32) | ((uint64_t)1 << 46);  // version = 1
}

// Block-scaled instruction descriptor for kind::mxf4 (E2M1 x E2M1, UE8M0).
__device__ __forceinline__ uint32_t make_idesc(int n_dim, int a_sf_id, int b_sf_id) {
    const uint32_t a_format = 1u;        // E2M1 (MXF4Format::E2M1)
    const uint32_t b_format = 1u;        // E2M1
    const uint32_t scale_format = 1u;    // UE8M0
    const uint32_t m_dim = (uint32_t)(kMTile >> 4);
    const uint32_t n = (uint32_t)(n_dim >> 3);
    uint32_t d = 0;
    d |= (b_format & 0x7u) << 10;
    d |= (scale_format & 0x1u) << 23;
    d |= (n & 0x3Fu) << 17;
    d |= (m_dim & 0x1Fu) << 24;
    d |= (a_format & 0x7u) << 7;
    d |= (uint32_t)(a_sf_id & 0x3u) << 29;
    d |= (uint32_t)(b_sf_id & 0x3u) << 4;
    return d;  // k_size = 0 (dense K64), majors = K, negates = 0, sparse = 0
}

// ------------------------------------------------------------------ kernels
// A is fp4 I8-packed [rows, k/2] with f32 per-(row, k/32) scales when AQ=false;
// when AQ=true A is f32 [rows, k] and is quantised to fp4 in-kernel (the down
// projection's activation, exactly like the reference's `x.to(fp4)` cast).
// B is fp4 I8-packed [n_rows, k/2] with u8 e8m0 per-(row, k/32) scales.
// The B row index is mapped through `b_virtual_split`: rows >= split read from
// (b_hi, row - split) instead (gate/up concatenation); split < 0 = no split.
template <bool AQ>
__global__ void __launch_bounds__(kThreads) mxf4_gemm_kernel(
    const uint8_t* __restrict__ a,        // [rows, k/2] fp4  (AQ=false)
    const float* __restrict__ a_scale,    // [rows, k/32] f32 (AQ=false)
    const float* __restrict__ a_f32,      // [rows, k]    f32 (AQ=true)
    const uint8_t* __restrict__ b,        // [b_rows, k/2] fp4
    const uint8_t* __restrict__ b_scale,  // [b_rows, k/32] e8m0
    const uint8_t* __restrict__ b_hi,     // second half (split >= 0), else same as b
    const uint8_t* __restrict__ b_hi_scale,
    float* __restrict__ out,              // [rows, n_out]
    int rows, int n_total, int k, int b_split, int epi_mode, float limit,
    const float* __restrict__ row_weight,
    // ---- indirect (graph-friendly) B addressing -------------------------
    // Given the per-layer pools' bases and per-expert strides plus a device
    // array of expert ids, the kernel derives its OWN B pointers. That removes
    // the host from the MoE dispatch: no per-layer routing download (a blocking
    // cudaMemcpy) and the launch arguments become independent of the routing,
    // which is what a CUDA graph needs. ids == nullptr keeps the direct path.
    const uint8_t* __restrict__ b_base, long b_stride,
    const uint8_t* __restrict__ bs_base, long bs_stride,
    const uint8_t* __restrict__ bh_base, long bh_stride,
    const uint8_t* __restrict__ bhs_base, long bhs_stride,
    const int* __restrict__ ids, int slot) {
    const uint8_t* b_use = b;
    const uint8_t* bsc_use = b_scale;
    const uint8_t* bhi_use = b_hi;
    const uint8_t* bhs_use = b_hi_scale;
    if (ids != nullptr) {
        const size_t e = (size_t)ids[slot];
        b_use = b_base + e * (size_t)b_stride;
        bsc_use = bs_base + e * (size_t)bs_stride;
        bhi_use = bh_base + e * (size_t)bh_stride;
        bhs_use = bhs_base + e * (size_t)bhs_stride;
    }
    const int m_base = blockIdx.y * kMTile;
    const int n_base = blockIdx.x * kNTile;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nk_blk = k >> 5;  // k-blocks of 32 (scale columns)

    // ------------------------------------------------------------- shared
    __shared__ __align__(1024) uint8_t s_a[kABytes];
    __shared__ __align__(1024) uint8_t s_b[kBBytes];
    __shared__ uint8_t s_aq_scale[kMTile][kStageAtoms * 2 / 2 + kStageAtoms * 2 / 2];  // [128][8]
    __shared__ __align__(8) uint64_t s_mbar;
    __shared__ uint32_t s_tmem_base;

    // ---------------------------------------------------------- tmem alloc
    if (warp == 0) {
        tc_alloc(&s_tmem_base, kTmemCols);
        tc_relinquish();
    }
    if (tid == 0) mbar_init(&s_mbar, 1);
    __syncthreads();

    const uint32_t tmem_base = s_tmem_base;
    const uint32_t d_col = tmem_base + 0;                    // kDCols columns
    const uint32_t sfa_col = tmem_base + kDCols;             // kSfaCols columns
    const uint32_t sfb_col = tmem_base + kDCols + kSfaCols;  // kSfbCols columns

    // ------------------------------------------------------------- K loop
    uint32_t phase = 0;
    for (int k0 = 0; k0 < k; k0 += kStageAtoms * kAtomK) {
        const int natoms = min(kStageAtoms, (k - k0 + kAtomK - 1) / kAtomK);
        const int nblk = natoms * 2;  // 32-element blocks in this stage

        // ---- 1. stage the A operand ------------------------------------
        if (!AQ) {
            // 16-byte chunk (atom, m, kb): src a[m][k0/2 + atom*32 + kb*16],
            // dst unit16 within the atom's block. A full 16 bytes (= 32 fp4
            // elements) per chunk; every atom of the stage must be loaded.
            for (int c = tid; c < kStageAtoms * kMTile * 2; c += kThreads) {
                const int atom = c / (kMTile * 2);
                if (atom >= natoms) continue;
                const int r = c % (kMTile * 2);
                const int m = r >> 1, kb = r & 1;
                const int row = m_base + m;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (row < rows)
                    val = *reinterpret_cast<const uint4*>(a + (size_t)row * (k >> 1) +
                                                          (k0 >> 1) + atom * kAtomBytes +
                                                          kb * 16);
                *reinterpret_cast<uint4*>(s_a + atom * kATileBytes +
                                          ((m & 7) + 8 * kb + 16 * (m >> 3)) * 16) = val;
            }
        } else {
            // f32 -> fp4 quantisation, 32 elements per (row, block)
            for (int t = tid; t < kMTile * nblk; t += kThreads) {
                const int m = t / nblk, bb = t % nblk;
                const int row = m_base + m;
                float vals[32];
                float amax = 0.f;
                const int kk = k0 + bb * 32;
                for (int i = 0; i < 32; ++i) {
                    float x = 0.f;
                    if (row < rows && kk + i < k) x = a_f32[(size_t)row * k + kk + i];
                    vals[i] = x;
                    amax = fmaxf(amax, fabsf(x));
                }
                const float sc = fast_round_scale6(amax);
                s_aq_scale[m][bb] = f_pow2_to_ue8m0(sc);
                const float inv = 1.0f / sc;
                uint8_t bytes[16];
                for (int i = 0; i < 16; ++i) {
                    const uint8_t lo = e2m1_encode(vals[2 * i] * inv);
                    const uint8_t hi = e2m1_encode(vals[2 * i + 1] * inv);
                    bytes[i] = (uint8_t)(lo | (uint8_t)(hi << 4));
                }
                // block bb -> atom bb/2, k-chunk kb = bb%2, unit16 within atom
                const int atom = bb >> 1, kb = bb & 1;
                uint4 val;
                __builtin_memcpy(&val, bytes, 16);
                *reinterpret_cast<uint4*>(s_a + atom * kATileBytes +
                                          ((m & 7) + 8 * kb + 16 * (m >> 3)) * 16) = val;
            }
        }

        // ---- 2. stage the B operand ------------------------------------
        for (int c = tid; c < kStageAtoms * kNTile * 2; c += kThreads) {
            const int atom = c / (kNTile * 2);
            const int r = c % (kNTile * 2);
            const int n = r >> 1, kb = r & 1;
            if (atom >= natoms) continue;
            const int n_glob = n_base + n;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (n_glob < n_total) {
                const uint8_t* src_base = b_use;
                int row = n_glob;
                if (b_split >= 0 && n_glob >= b_split) {
                    src_base = bhi_use;
                    row = n_glob - b_split;
                }
                if (row >= 0)
                    val = *reinterpret_cast<const uint4*>(src_base + (size_t)row * (k >> 1) +
                                                          (k0 >> 1) + atom * kAtomBytes +
                                                          kb * 16);
            }
            *reinterpret_cast<uint4*>(s_b + atom * kBTileBytes +
                                      ((n & 7) + 8 * kb + 16 * (n >> 3)) * 16) = val;
        }

        // make the smem writes visible to the async proxy (the MMA)
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        // ---- 3. stage the scale factors into TMEM -----------------------
        // Two consecutive K-atoms share ONE 32-bit SF word: the even atom's
        // pair (read at SFA_ID = 0) lives in bytes 0-1, the odd atom's pair
        // (SFA_ID = 2) in bytes 2-3. That is the PTX "scale_vec::2X" word
        // layout [SF0, SF1, SF0, SF1] with the 2-bit sub-column selector, and
        // it matches CUTLASS's 2X source layout where the second atom sits two
        // bytes after the first.
        // Row mapping: row m -> lane (m%32), column (base + m/32). Every warp
        // fills its own 32-lane partition with the full content (the factors
        // are duplicated to all four partitions).
        const int npairs = (natoms + 1) >> 1;
        const int abase = k0 >> 5;  // first 32-element block of this K stage
        for (int pr = 0; pr < npairs; ++pr) {
            uint32_t wa[4];
            uint32_t wb[kNTile / 32];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int m = 32 * j + lane;
                const int row = m_base + m;
                uint8_t v[4] = {0, 0, 0, 0};
                if (row < rows) {
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const int brel = 4 * pr + t;   // block within this stage
                        if (brel >= 2 * natoms) break; // odd tail: pair half empty
                        if (!AQ) {
                            const int bb = abase + brel;  // global block index
                            if (bb < nk_blk)
                                v[t] = f_pow2_to_ue8m0(a_scale[(size_t)row * nk_blk + bb]);
                        } else {
                            if (brel < nblk) v[t] = s_aq_scale[m][brel];
                        }
                    }
                }
                wa[j] = (uint32_t)v[0] | ((uint32_t)v[1] << 8) | ((uint32_t)v[2] << 16) |
                        ((uint32_t)v[3] << 24);
            }
#pragma unroll
            for (int j = 0; j < kNTile / 32; ++j) {
                const int n = 32 * j + lane;
                const int n_glob = n_base + n;
                uint8_t v[4] = {0, 0, 0, 0};
                if (n_glob < n_total) {
                    const uint8_t* sc = bsc_use;
                    int row = n_glob;
                    if (b_split >= 0 && n_glob >= b_split) {
                        sc = bhs_use;
                        row = n_glob - b_split;
                    }
                    if (row >= 0) {
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const int brel = 4 * pr + t;
                            if (brel >= 2 * natoms) break;
                            const int bb = abase + brel;
                            if (bb < nk_blk) v[t] = sc[(size_t)row * nk_blk + bb];
                        }
                    }
                }
                wb[j] = (uint32_t)v[0] | ((uint32_t)v[1] << 8) | ((uint32_t)v[2] << 16) |
                        ((uint32_t)v[3] << 24);
            }
            tc_st_x4(((uint32_t)(warp * 32) << 16) | (sfa_col + 4 * pr), wa[0], wa[1], wa[2], wa[3]);
            if (kNTile / 32 == 2)
                tc_st_x2(((uint32_t)(warp * 32) << 16) | (sfb_col + 2 * pr), wb[0], wb[1]);
            else
                tc_st_x4(((uint32_t)(warp * 32) << 16) | (sfb_col + 2 * pr), wb[0], wb[1], wb[2],
                         wb[3]);
        }
        tc_wait_st();
        tc_fence_before_thread_sync();
        __syncthreads();
        tc_fence_after_thread_sync();

        // ---- 4. issue the MMAs ------------------------------------------
        if (tid == 0) {
#pragma unroll 1
            for (int atom = 0; atom < kStageAtoms; ++atom) {
                if (atom >= natoms) break;
                const uint64_t da = make_desc(smem_addr(s_a) + atom * kATileBytes);
                const uint64_t db = make_desc(smem_addr(s_b) + atom * kBTileBytes);
                const int pr = atom >> 1;                  // atom pair index
                const int sf_id = (atom & 1) ? 2 : 0;      // low / high half-word
                const uint32_t sa_col = sfa_col + 4 * pr;
                const uint32_t sb_col = sfb_col + 2 * pr;
                const uint32_t id = make_idesc(kNTile, sf_id, sf_id);
                const uint32_t en = (k0 == 0 && atom == 0) ? 0u : 1u;
                tc_mma_mxf4(d_col, da, db, id, sa_col, sb_col, en);
            }
            tc_commit(&s_mbar);
        }
        mbar_wait(&s_mbar, phase);
        phase ^= 1u;
    }

    // ------------------------------------------------------------ epilogue
    // D[m][n] lives at lane m, column d_col + n.
    for (int c0 = 0; c0 < kNTile; c0 += 16) {
        uint32_t v[16];
        tc_ld_x16((((uint32_t)(warp * 32)) << 16) | (d_col + c0), v);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row = m_base + warp * 32 + lane;
            const int col = n_base + c0 + i;
            if (row >= rows || col >= n_total) continue;
            float x = __uint_as_float(v[i]);
            if (epi_mode == 1) {  // gate/up clamps (training convention)
                if (limit > 0.f) {
                    if (col < b_split) x = fminf(x, limit);                        // gate
                    else x = fminf(fmaxf(x, -limit), limit);                       // up
                }
            } else if (epi_mode == 2 || epi_mode == 3) {  // down: routing weight
                if (row_weight != nullptr) x *= row_weight[row];
            }
            // epi_mode 3 accumulates straight into the caller's MoE accumulator,
            // so the host no longer needs one add_inplace launch per expert.
            if (epi_mode == 3) {
                out[(size_t)row * n_total + col] += x;
            } else {
                out[(size_t)row * n_total + col] = x;
            }
        }
    }

    __syncthreads();
    if (warp == 0) tc_dealloc(tmem_base, kTmemCols);
}

// ---------------------------------------------------------------------------
// M=1 fp4 GEMV. The tcgen05 kind::mxf4 MMA has M pinned at 128 by the hardware
// (see kMTile), so at decode's M=1 the tensor-core path computes a 128x64 tile to
// emit ONE row and launches a grid of (n/64, 1) - five blocks for the whole
// expert. Measured 97.8 us per call for 1.64 MB of weights, i.e. 16.8 GB/s, 0.2%
// of the part. This kernel does the same arithmetic with none of that machinery:
// one warp per output row, block-scale-aware fp4 unpacking, no tmem, no MMA.
// Format (read off mxf4_gemm_kernel, must match bit for bit):
//   b       [n, k/2]  fp4, two values per byte, LOW nibble first
//   b_scale [n, k/32] e8m0, value = 2^(byte-127) = __uint_as_float(byte<<23)
//   a_f32   [1, k]    f32 activations (the AQ=true path)
__device__ __forceinline__ float dsv41_e2m1_to_f(uint8_t n) {
    // 1 sign, 2 exponent, 1 mantissa; exponent 0 is the subnormal pair {0, 0.5}
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float m = mag[n & 7u];
    return (n & 8u) ? -m : m;
}

__global__ void expert_gemv_fp4_kernel(const float* __restrict__ a_f32,
                                       const uint8_t* __restrict__ a,
                                       const float* __restrict__ a_scale,
                                       const uint8_t* __restrict__ b,
                                       const uint8_t* __restrict__ b_scale,
                                       const uint8_t* __restrict__ b_hi,
                                       const uint8_t* __restrict__ b_hi_scale,
                                       float* __restrict__ out, int n_total, int k, int b_split,
                                       int epi_mode, float limit, const float* __restrict__ row_weight,
                                       const uint8_t* __restrict__ b_base, long b_stride,
                                       const uint8_t* __restrict__ bs_base, long bs_stride,
                                       const uint8_t* __restrict__ bh_base, long bh_stride,
                                       const uint8_t* __restrict__ bhs_base, long bhs_stride,
                                       const int* __restrict__ ids, int slot) {
    const uint8_t* b_use = b;
    const uint8_t* bsc_use = b_scale;
    const uint8_t* bhi_use = b_hi;
    const uint8_t* bhs_use = b_hi_scale;
    if (ids != nullptr) {
        const size_t e = (size_t)ids[slot];
        b_use = b_base + e * (size_t)b_stride;
        bsc_use = bs_base + e * (size_t)bs_stride;
        bhi_use = bh_base + e * (size_t)bh_stride;
        bhs_use = bhs_base + e * (size_t)bhs_stride;
    }
    // The activation is ONE row shared by every output row, so stage it once per
    // block instead of letting each of the 512 output rows re-read it from global:
    // that re-reading cost 512 x 5120 x 4B = 10.5 MB per call against 0.65 MB of
    // weights, which is what pinned this kernel at 38 GB/s (0.5 percent).
    extern __shared__ float s_act[];   // k floats (20 KB at k=5120)
    const int kbytes = k >> 1;   // packed bytes per row
    const int ksc = k >> 5;      // e8m0 scales per row
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        if (a_f32 != nullptr) {
            s_act[j] = a_f32[j];
        } else {
            const uint8_t ab = a[j >> 1];
            const float asc = a_scale[j >> 5];
            s_act[j] = dsv41_e2m1_to_f((j & 1) ? (uint8_t)(ab >> 4) : (uint8_t)(ab & 0xFu)) * asc;
        }
    }
    __syncthreads();
    // One warp per output row, 8 rows per block. Two "obvious" improvements were
    // measured and both were WORSE, so this shape is the keeper: a k-split (one
    // block per row, 8 warps splitting k -> 512 blocks) gave 15.0 tok/s against this
    // 15.2, and 16-byte uint4 lanes gave 14.3. The kernel is not occupancy- or
    // request-rate-bound the way those two assumed.
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = (blockDim.x + 31) >> 5;

    for (int row = blockIdx.x * nwarps + warp; row < n_total; row += gridDim.x * nwarps) {
        // gate/up split: rows < b_split read the `b` pair, the rest the `b_hi` pair
        const bool hi = (b_split > 0) && (row >= b_split);
        const int r = hi ? (row - b_split) : row;
        const uint8_t* bb = hi ? bhi_use : b_use;
        const uint8_t* bb_s = hi ? bhs_use : bsc_use;
        const uint8_t* brow = bb + (size_t)r * kbytes;
        const uint8_t* srow = bb_s + (size_t)r * ksc;

        float acc = 0.f;
        for (int j = lane * 2; j < k; j += 64) {
            // two consecutive fp4 values share one byte; every 32 k share one scale
            const uint8_t byte = brow[j >> 1];
            const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
            const float w0 = dsv41_e2m1_to_f(byte & 0xFu) * sc;
            const float w1 = dsv41_e2m1_to_f((uint8_t)(byte >> 4)) * sc;
            acc += s_act[j] * w0;
            acc += s_act[j + 1] * w1;
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) {
            float x = acc;
            if (epi_mode == 1) {
                if (limit > 0.f) {
                    if (row < b_split) x = fminf(x, limit);
                    else x = fminf(fmaxf(x, -limit), limit);
                }
            } else if (epi_mode == 2 || epi_mode == 3) {
                // row_weight is the routing weight for the (token, slot) being
                // computed - ONE scalar, the caller passes route_w + slot (see the
                // expert loop in chain_dev.rs). It is indexed by the M row, and
                // this kernel exists only for M == 1, so the index is ALWAYS 0.
                // The original mxf4_gemm did not catch fire because its M loop is
                // bounded by rows (= 1), so it only ever touched row_weight[0];
                // indexing it by the output column `row` here read 4096 floats past
                // a one-float pointer, which is the out-of-bounds access seen when
                // the down path first used this kernel.
                if (row_weight != nullptr) x *= row_weight[0];
            }
            if (epi_mode == 3) out[(size_t)row] += x;
            else out[(size_t)row] = x;
        }
    }
}

// ---------------------------------------------------------------------------
// BATCHED M=1 fp4 expert GEMV (env-gated by DSV41_MOE_BATCH on the Rust side,
// default OFF). ONE launch per (layer, direction) covers every top-k slot:
// grid = (rows_blocks, slots) with blockIdx.y = the slot, so each block derives
// its own expert from ids[slot]. The launch COUNT is the MoE family's real
// lever - the per-call launch floor is ~3.05 us (measured, see the "空 kernel
// 启动地板实测" section of docs/agent/perf-roadmap.md) while the inner-loop
// levers were measured and are exhausted (see the note in
// expert_gemv_fp4_kernel).
//
// This is a line-for-line copy of expert_gemv_fp4_kernel with three
// substitutions, which is what makes a batched result BIT-IDENTICAL to the
// sequential loop: (a) the expert id / weight base come from ids[blockIdx.y];
// (b) the f32 activation is read from a_f32 + blockIdx.y * act_stride (the
// per-slot swiglu slice; gate/up passes a_f32 == nullptr and uses the shared
// quantised `a`); (c) the output goes to out + blockIdx.y * out_slot_stride.
// The per-row K dot order and the warp shuffle reduction are unchanged.
//
// Per-slot outputs MUST be disjoint - this kernel never accumulates across
// slots. The down direction therefore writes a [slots][n_total] scratch that
// moe_down_reduce_kernel sums in a FIXED ascending-slot order (fp addition is
// not associative, so the order is part of the numerical contract).
//
// row_weight is PER SLOT here: it is read at row_weight[slot * rw_stride],
// which is the same scalar the sequential caller passed as `route_w + slot`
// (whose kernel then read row_weight[0]).
// Four bytes (eight fp4 values) per lane per iteration instead of one byte, behind
// DSV41_EXPERT_FP4_VEC. The mx block scale covers 32 values, so eight fp4 always sit
// inside one scale block: lanes 0-3 share block 0, lanes 4-7 block 1, which is what
// (lane >> 2) selects. The per-element product keeps its original shape; only the
// order in which a lane visits its elements changes.
static const int g_expert_fp4_mode = [] {
    const char* e = getenv("DSV41_EXPERT_FP4_MODE");
    if (e == nullptr) return 2;       // 2 = shared lut + split accumulators: -5.38 ms, text identical
    return atoi(e);                   // 0 scalar, 1 vectorised (both kept for bisection)
}();
// DOWN direction lane map, separate from the gate/up one on purpose.
//
// Why: at the production down shape (k = inter_local = 320) `nv2 = k >> 9 = 0`,
// so the whole vec==2 main loop is dead code and 100% of the dot used to run in
// the 2-value tail (1 LDG.U8 weight + 1 LDG.U8 scale + 1 LDS.64 LUT + 2 LDS.32
// activation per 2 values). Mode 3 is a 4-value/lane tail: 1 LDG.U16 weight +
// 1 LDS.128 activation + 2 LDS.64 LUT per 4 values (1.25 L1TEX op/value vs 2.5),
// and the scale covers the whole 4-value group so it is applied once per
// accumulator instead of once per element. Measured in isolation at the exact
// production shape (dim=7168, k=320, 256 threads, 896 blocks, sm_103a, nvcc
// 13.2, /tmp/dv320 evidence on the bench node), 5 interleaved rounds:
//   mode 2, 40 regs, 6 blocks/SM, 1.01 waves : 1.00 (baseline)
//   mode 3, 40 regs, 6 blocks/SM, 1.01 waves : 0.90   (+launch_bounds__(256,6))
//   mode 3, 56 regs, 4 blocks/SM, 1.51 waves : 0.87   <- fastest
// i.e. the 4-value form wins in 5/5 rounds and, at this shape, the extra
// registers cost LESS than the shorter tail - the "40 registers is a hard
// occupancy red line" reading does not reproduce (see the note in the vec==2
// branch below). The 8-value uint32 form (01291b2) is the one that loses:
//   mode 4, 40 regs, 6 blocks/SM : 1.03   mode 4, 62 regs, 4 blocks/SM : 0.97
//
// It is NOT fed to the gate/up launch: that path needs vec==2 for its fused
// swiglu body (`fuse` requires g_expert_fp4_mode == 2). Set DSV41_DOWN_VEC4=0
// to fall back to DSV41_EXPERT_FP4_MODE for the down launches (bisection).
static const int g_down_fp4_mode = [] {
    const char* e = getenv("DSV41_DOWN_VEC4");
    if (e != nullptr && atoi(e) == 0) return g_expert_fp4_mode;
    return g_expert_fp4_mode == 2 ? 3 : g_expert_fp4_mode;
}();

// ---------------------------------------------------------------------------
// PDL (programmatic dependent launch) for the DSV41 EXPERT chain.
//
// `dsv41_pdl_or_plain` (dsv41_kernels.cu, the attention projection chain) and
// `pdl_or_plain` (ferrite_kernels.cu) are file-static in OTHER translation
// units, so this one carries its own copy under the SAME `DSV41_PDL` gate
// (DEFAULT ON; an explicit "0" rolls back). The semantics are identical to
// those two, which are the verified-capture precedents:
//
//   * DSV41_PDL unset or != "0" -> the launch carries
//     cudaLaunchAttributeProgrammaticStreamSerialization, so the consumer grid
//     may be scheduled while the producer is still draining its tail. The
//     producer needs no cudaTriggerProgrammaticLaunchCompletion(): the implicit
//     trigger fires when its CTAs exit. The win is node-transition cost
//     (grid rasterisation, CTA scheduling, register allocation, plus whatever
//     prologue does NOT read the producer), NOT bandwidth.
//   * DSV41_PDL=0 -> the same cudaLaunchKernelEx path WITHOUT the attribute: a
//     plain launch, which records the identical node in a stream capture. This
//     is the A/B arm and the rollback.
//
// CONTRACT (must hold for every kernel routed through this helper): the kernel
// MUST call cudaGridDependencySynchronize() before reading ANY output written
// by the PREVIOUS kernel on the stream, unconditionally inside
// `#if __CUDA_ARCH__ >= 900`. The call is a documented no-op on a plain launch,
// so it stays in place when DSV41_PDL=0.
//
// COVERED HERE: the two consumers of the routed-expert fp4 chain
// (quant_fp4 -> gateup -> down_reduce):
//   * expert_gemv_fp4_batched_kernel<ILV>  -- BOTH the batched gate/up and the
//     batched down direction run through this one kernel (the staging source
//     is what differs: `a`/`a_scale` for gate/up, `act` for down);
//   * expert_gemv_fp4_down_reduce_kernel<STAGED> -- the fused down + reduce.
// The sequential per-slot (`*_indirect`) entries are deliberately NOT covered:
// they are the fallback arm nobody should silently start running under PDL.
//
// PRODUCER NOTE: the gate/up batched call's producer is quant_fp4_fused_kernel
// (it wrote `a`/`a_scale`); the down call's and down_reduce's producer is the
// gate/up launch (it wrote the swiglu'd activation `act`). `ids` and
// `row_weight` are NOT outputs of the immediately preceding kernel -- the
// router wrote them several kernels earlier, so they are already flushed by the
// time this PDL-secondary grid is released, and reading them before the sync is
// exactly the hoisted pointer work below.
//
// ARCH: the device sync is arch-gated, the host gate is not. This TU is built
// for sm_100a/sm_103a only (build.sh), so the guard is always taken. On an
// unsupported device the attribute makes cudaLaunchKernelEx fail loudly, so a
// mismatch cannot be silent.
static int dsv41_experts_pdl_enabled(void) {
    // Read once: these launchers run 40x/step and a per-call getenv on the hot
    // path is the slip every other gate in this file avoids.
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("DSV41_PDL");
        cached = (e != nullptr && e[0] == '0') ? 0 : 1;   // explicit "0" rolls back
    }
    return cached;
}

// Rows per CTA for the BATCHED gate/up launch (DSV41_GATEUP_ROWS, default 8 =
// today's shape). ONE warp owns ONE row here, so blockDim = rows*32 and
// grid.x = ceil(n_total/rows) with n_total = inter (fused) or 2*inter.
//
// ⚠️ THIS KNOB DOES NOT CHANGE THE WARP COUNT. rows x slots is the whole work
// split (320 x 6 = 1920 row-dots at DSV4.1 shapes, one warp each), so re-packing
// those warps into 240 / 480 / 960 SMALLER CTAs leaves the resident warps per SM
// untouched (1920/148 = 13 either way) AND leaves the per-warp MLP untouched.
// It is a CTA-GRANULARITY experiment, not an occupancy fix - see the
// "expert-floor-revisit" note in docs/agent/dsv41-kernel-inventory-v3.md.
// Its value is as a FALSIFICATION test of the "240 blocks = 1.6/SM is the
// bottleneck" reading: if the per-call time is flat across 8 / 4 / 2, the CTA
// count was never the lever and only a K-split (which multiplies the warp count)
// can move the latency-hiding number.
// Two secondary effects are real and both NEGATIVE:
//  * s_act (k floats) + the 256-entry LUT are staged PER CTA and shared by that
//    CTA's rows, so halving the rows per CTA DOUBLES the prologue per row;
//  * a smaller CTA has fewer warps with which to overlap that prologue against
//    the K loads.
// The window loop is closed over the whole grid, so any value works; 8/4/2/1 are
// the meaningful ones (a non-divisor only wastes tail CTAs).
constexpr int kGateUpRowsMin = 1;
constexpr int kGateUpRowsMax = 32;

static int dsv41_gateup_rows(void) {
    // Read once: this launcher runs 40x/step (same rule as the PDL gate above).
    static int cached = -1;
    if (cached < 0) {
        int v = 8;
        if (const char* e = getenv("DSV41_GATEUP_ROWS")) {
            v = atoi(e);
            if (v < kGateUpRowsMin) v = kGateUpRowsMin;
            if (v > kGateUpRowsMax) v = kGateUpRowsMax;
        }
        cached = v;
    }
    return cached;
}

// K-SPLIT for the FUSED gate/up body (DSV41_GATEUP_KSPLIT, default 1 = OFF).
//
// WHY: the fused branch was measured at ~6% issue with ~94% of cycles stalled
// on the K loads, i.e. the warp has too few in-flight load slots. Re-packing
// the SAME warps into more CTAs (DSV41_GATEUP_ROWS) cannot add a single warp -
// rows x slots (320 x 6 = 1920 row-dots) IS the whole work split, one warp per
// row. The only way to ADD warps is to hand ONE row to ksplit warps and give
// each a contiguous slice of the K groups.
//
// SHAPE: ksplit warps per row => blockDim = rows*ksplit*32 and the SAME
// grid.x = ceil(n_total/rows) as before (rows = DSV41_GATEUP_ROWS, the CTA's
// ROW count, unchanged). At the recommended rows=8 / ksplit=2 that is
// 16 warps = 512 threads and still 40 x 6 = 240 CTAs: the warp count doubles
// (1920 -> 3840, ~26/SM) WITHOUT doubling the per-CTA `s_act` (k floats) + LUT
// prologue (which is why rows=4/ksplit=2 - 480 CTAs - was rejected by the
// design review: it halves the rows sharing each CTA's prologue).
//
// PARITY: splitting the K walk changes the SUMMATION ORDER - originally one
// serial chain g0+g1+...+g9, now (g0..g4) + (g5..g9) with the halves summed by
// ONE deterministic __fadd_rn at the group boundary (half 0 owns [0,5), half 1
// owns [5,10), always merged in ascending half order). Mathematically identical,
// not bit-identical; the per-layer drift is ~1e-7. Hence DEFAULT OFF (1) - the
// whole point of this gate is to validate the text A/B before flipping it ON.
//
// The per-half group range is [half*nv2f/ksplit, (half+1)*nv2f/ksplit) with
// nv2f = k>>9 (10 at k=5120) - a cut on 512-value group boundaries, so no
// 32-value scale block is ever straddled. nv2f=10 is NOT divisible by 4 (only
// 3/3/2/2), so the meaningful values are 2 (or 5); 4 is allowed but unbalanced.
constexpr int kGateUpKsplitMax = 8;

static int dsv41_gateup_ksplit(void) {
    static int cached = -1;
    if (cached < 0) {
        int v = 1;
        if (const char* e = getenv("DSV41_GATEUP_KSPLIT")) {
            v = atoi(e);
            if (v < 1) v = 1;
            if (v > kGateUpKsplitMax) v = kGateUpKsplitMax;
        }
        cached = v;
    }
    return cached;
}

// NOTE: the `<<<>>>` launch syntax takes no launch attribute, so both arms go
// through cudaLaunchKernelEx (its variadic template applies the kernel's
// declared parameter types to the arguments, exactly like `<<<>>>` would).
template <typename K, typename... Args>
static inline cudaError_t dsv41_experts_pdl_or_plain(K kern, dim3 grid, dim3 block, size_t smem,
                                                     cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    if (dsv41_experts_pdl_enabled()) {
        cfg.attrs = attrs; cfg.numAttrs = 1;
    }
    return cudaLaunchKernelEx(&cfg, kern, args...);
}


// ILV = the w1 (gate) / w3 (up) fp4 weights were stored INTERLEAVED at load
// time (8-byte granule alternation, built by dsv41_interleave_gateup_fp4): the
// gate chunk of a row and the matching up chunk sit 8 bytes apart, so the fused
// gate/up branch fetches both with ONE LDG.128 instead of two LDG.64s. Same 16
// bytes, same decode, same fma chains - only the load count changes.
//
// A TEMPLATE parameter, not a runtime flag: the branch below folds away in the
// non-interleaved instantiation, so the default (plain-layout) path keeps the
// exact codegen it had. The launcher picks the instantiation from its explicit
// `ilv` argument; nothing here guesses, and the Rust side refuses to combine an
// interleaved pool with anything but the fused batched gate/up call.
template <bool ILV>
__global__ void expert_gemv_fp4_batched_kernel(const float* __restrict__ a_f32, long act_stride,
                                               const uint8_t* __restrict__ a,
                                               const float* __restrict__ a_scale,
                                               float* __restrict__ out, long out_slot_stride,
                                               int n_total, int k, int b_split, int epi_mode,
                                               float limit, const float* __restrict__ row_weight,
                                               long rw_stride, const uint8_t* __restrict__ b_base,
                                               long b_stride, const uint8_t* __restrict__ bs_base,
                                               long bs_stride, const uint8_t* __restrict__ bh_base,
                                               long bh_stride, const uint8_t* __restrict__ bhs_base,
                                               long bhs_stride, const int* __restrict__ ids,
                                               int vec, int fuse_swiglu, int ksplit) {
    const int slot = (int)blockIdx.y;
    const float* act = (a_f32 != nullptr) ? (a_f32 + (size_t)slot * (size_t)act_stride) : nullptr;
    const float* rw = (row_weight != nullptr) ? (row_weight + (size_t)slot * (size_t)rw_stride)
                                              : nullptr;
    out += (size_t)slot * (size_t)out_slot_stride;
    const size_t e = (size_t)ids[slot];
    const uint8_t* b_use = b_base + e * (size_t)b_stride;
    const uint8_t* bsc_use = bs_base + e * (size_t)bs_stride;
    const uint8_t* bhi_use = bh_base + e * (size_t)bh_stride;
    const uint8_t* bhs_use = bhs_base + e * (size_t)bhs_stride;
    // Same activation staging as the sequential kernel: the ONE shared quantised
    // row for gate/up, the slot's own f32 swiglu slice for down.
    // NOTE: the padded layout (one hole per 16 floats) was tried and REVERTED -
    // it removed a 16-way s_act bank conflict, but the conflicts were worth only
    // ~1 percent of the kernel (the same measurement the gemv LUT work made:
    // distinct-banks already at the uniform-random ceiling) while the extra
    // address ALU (x + (x>>4)) sat on the consume chain's critical path and cost
    // +0.24ms/step in serve. Plain linear staging is the correct form.
    extern __shared__ float s_act[];   // k floats
    // 256-entry byte->float2 table: one LDS.64 yields both nibbles' e2m1 values
    // (the old 16-entry scalar table needed two LDS.32 plus a shift per value).
    float2* s_lut2 = reinterpret_cast<float2*>(s_act + k);
    // K-split partials (DSV41_GATEUP_KSPLIT>1, fused gate/up only): ONE (gate,up)
    // float2 per warp, indexed by the CTA-local warp id. The ksplit halves of a
    // row are consecutive warps (warp = row_local*ksplit + half), so half 0 reads
    // s_ks[warp+1 .. warp+ksplit-1] to fold its partners. Laid out right after the
    // 256-entry LUT and allocated by the launcher ONLY when ksplit>1, so the
    // ksplit==1 launch keeps the original dynamic-smem size (and occupancy).
    float2* s_ks = reinterpret_cast<float2*>(s_lut2 + 256);
    const int kbytes = k >> 1;   // packed bytes per row
    const int ksc = k >> 5;      // e8m0 scales per row
    // PDL (DSV41_PDL, see dsv41_experts_pdl_or_plain above): the launcher may
    // have launched this grid with programmatic stream serialization, so the
    // grid is already resident here and this call is what makes the PRODUCER's
    // writes visible. The producer output this kernel consumes is the staged
    // activation below -- quant_fp4's packed row (`a`/`a_scale`) for the
    // gate/up direction, the gate/up launch's swiglu'd f32 slice (`act`) for
    // the down direction -- so the sync MUST stay before the first staging load.
    //
    // Everything ABOVE is producer-INDEPENDENT and is deliberately spent in
    // the producer's ramp-down instead of after it: the slot/pointer setup is
    // pure argument arithmetic, `ids[slot]` and the four per-expert weight
    // bases read buffers the ROUTER wrote (several kernels before the
    // producer, hence already flushed when this grid is released), and the LUT
    // below is built from device constants. The LUT build is the one piece of
    // real prologue work here, which is why it is hoisted above the sync. It
    // only touches this CTA's own smem, so ordering it before the sync is safe
    // and the existing __syncthreads() still publishes it.
    //
    // No-op on a plain launch (DSV41_PDL=0).
    // NOT a bare `if (threadIdx.x < 256)`: the block is only 256 threads at the
    // default rows=8 (DSV41_GATEUP_ROWS); at a smaller CTA the guard would leave
    // entries blockDim.x..255 of the table untouched (stale/garbage LUT). The
    // stride loop is BIT-IDENTICAL at >= 256 threads (one iteration per thread,
    // same index, same value).
    for (int t = threadIdx.x; t < 256; t += blockDim.x)
        s_lut2[t] = make_float2(dsv41_e2m1_to_f((uint8_t)(t & 0xF)),
                                dsv41_e2m1_to_f((uint8_t)(t >> 4)));
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
    // Vectorized staging (audit #2): one uint4 = 16 bytes = 32 fp4 values =
    // exactly one scale block, so each thread-iteration is 1 LDG.128 + 1 scale
    // load instead of 20 serial LDG.8s. BIT-EXACT: the nibble order (low ->
    // even j, high -> odd j), the e2m1 decode and the per-32 scale multiply are
    // identical to the byte loop below; only the number of load instructions
    // changes. The f32 path gets the same treatment via float4.
    if ((k & 31) == 0 && a != nullptr && act == nullptr &&
        (((uintptr_t)a & 15) == 0)) {
        const int nb32 = k >> 5;
        for (int b32 = threadIdx.x; b32 < nb32; b32 += blockDim.x) {
            const uint4 packed =
                *reinterpret_cast<const uint4*>(a + (size_t)b32 * 16);
            const float asc = a_scale[b32];
            float* dst = s_act + (size_t)b32 * 32;
            const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed);
#pragma unroll
            for (int q = 0; q < 16; ++q) {
                dst[2 * q] = dsv41_e2m1_to_f((uint8_t)(pb[q] & 0xFu)) * asc;
                dst[2 * q + 1] = dsv41_e2m1_to_f((uint8_t)(pb[q] >> 4)) * asc;
            }
        }
    } else if ((k & 3) == 0 && act != nullptr && (((uintptr_t)act & 15) == 0)) {
        const int nf4 = k >> 2;
        for (int f4 = threadIdx.x; f4 < nf4; f4 += blockDim.x) {
            const float4 v = *reinterpret_cast<const float4*>(act + (size_t)f4 * 4);
            float* dst = s_act + (size_t)f4 * 4;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    } else {
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        if (act != nullptr) {
            s_act[j] = act[j];
        } else {
            const uint8_t ab = a[j >> 1];
            const float asc = a_scale[j >> 5];
            s_act[j] = dsv41_e2m1_to_f((j & 1) ? (uint8_t)(ab >> 4) : (uint8_t)(ab & 0xFu)) * asc;
        }
    }
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = (blockDim.x + 31) >> 5;

    // K-SPLIT row mapping (DSV41_GATEUP_KSPLIT, default 1 = original).
    // With ksplit warps per row, nwarps = rows*ksplit, so the CTA's ROW count is
    // nwarps/ksplit and warp `w` owns row_local `w/ksplit`, half `w%ksplit`.
    // ksplit==1 => rows_per_cta == nwarps, row_local == warp: the exact original
    // mapping. grid.x is still ceil(n_total/rows) (the launcher sizes it by the
    // ROW count), i.e. ONE pass; ksplit>1 therefore stops the window loop after
    // its single iteration and lets the warps whose row is past n_total reach the
    // fused branch's cross-half __syncthreads() as well (guard by `active`, not
    // by the loop bound, or that barrier would deadlock the CTA).
    const int rows_per_cta = (ksplit > 0) ? (nwarps / ksplit) : nwarps;
    const int row_local = (ksplit > 0) ? (warp / ksplit) : warp;
    const int half = (ksplit > 0) ? (warp % ksplit) : 0;
    const int row_base = blockIdx.x * rows_per_cta + row_local;
    const int row_stop = (ksplit > 1) ? (row_base + 1) : n_total;   // ksplit>1: one trip
    for (int row = row_base; row < row_stop; row += gridDim.x * rows_per_cta) {
        // gate_up + swiglu fusion (fuse_swiglu && b_split > 0, gate/up direction).
        // Here n_total == inter and this warp owns ONE inter row `row`: it walks
        // BOTH halves of that row - the gate row `row` of the `b` pair and the up
        // row `row` of the `b_hi` pair - and writes the swiglu'd result straight
        // into out[row]. The caller then never materialises the 2*inter gate/up
        // buffer nor runs the separate swiglu pass.
        // NUMERIC CONTRACT (corrected 2026-09-11): this fused body is NOT the
        // vec==2 shape of the unfused body - the old comment claiming "same single
        // scale multiply per accumulator / same floats as the unfused rows" was
        // left over from 7100ebfe and invalidated by 667c6f66. The unfused vec==2
        // body keeps FOUR accumulators a0..a3 persistent across the groups (four
        // scale-FMAs per group, epilogue (a0+a1)+(a2+a3)); this fused body folds
        // each group into ONE 4-element tree and does ONE scale multiply per
        // group. The two are already NOT bit-identical at ksplit==1. What IS still
        // guaranteed here: the per-group fma chains run in ascending g2 order with
        // one `g`/`u` update each (unroll DEPTH is not part of the contract), so
        // ksplit==1 reproduces the pre-K-split fused bit pattern exactly.
        // ksplit>1 changes the summation order to (g0..g4)+(g5..g9) - see
        // dsv41_gateup_ksplit().
        // k = dim and the launcher only sets
        // fuse_swiglu when (dim % 512) == 0, so the two-chunk-per-scale tail loop
        // of the unfused body has no work here.
        if (fuse_swiglu && b_split > 0) {
            // K-split: warps whose `row` is past n_total still enter this branch
            // (they must reach the cross-half __syncthreads() below, or the CTA
            // would deadlock). `active` gates the final write; the loads of an
            // inactive warp are redirected to row 0 (a valid, harmless row) so no
            // out-of-range pointer is ever dereferenced. `row` is warp-uniform, so
            // the shfl tree below stays warp-uniform.
            const bool active = (row < n_total);
            const int row_c = active ? row : 0;
            // ILV: gate and up live in ONE region with an 8-byte granule
            // alternation, so the row pitch doubles and the up pointer is derived
            // from the gate pointer (the `b_hi` base is not read at all).
            const uint8_t* g_row = b_use + (size_t)row_c * (ILV ? (kbytes << 1) : kbytes);
            const uint8_t* u_row = ILV ? g_row : (bhi_use + (size_t)row_c * kbytes);
            const uint8_t* g_srow = bsc_use + (size_t)row_c * ksc;
            const uint8_t* u_srow = bhs_use + (size_t)row_c * ksc;
            const int nv2f = k >> 9;   // 16 values per lane per group; no tail (k % 512 == 0)
            // K-split group slice: this half walks the CONTIGUOUS groups
            // [half*nv2f/ksplit, (half+1)*nv2f/ksplit). ksplit==1 => [0, nv2f), the
            // original bounds. j and q below are unchanged because the cut always
            // lands on a 512-value group boundary (never inside a 32-value scale
            // block). half 0 owns the LOW half, half 1 the HIGH half.
            const int g_begin = (half * nv2f) / ksplit;
            const int g_end = ((half + 1) * nv2f) / ksplit;
            float g = 0.f, u = 0.f;
            // unroll 4 (was 2): the audit measured this branch at ~6% issue with
            // ~94% of cycles stalled on the K loads, i.e. too few in-flight load
            // slots per warp. Four groups in flight keep more LDG.64s outstanding
            // per warp without touching the accumulation ORDER (see the numeric
            // contract above): each group's fma chain is independent except for the
            // single `g`/`u` update per group, which still happens in g2 order.
#pragma unroll 4
            for (int g2 = g_begin; g2 < g_end; ++g2) {
                const int j = (g2 << 9) + (lane << 4);
                // Hoist the lane's 16 activation floats into registers ONCE per
                // group: the gate chain and the up chain read the SAME 16 slots of
                // s_act (identical addresses), but the SASS showed nvcc emitting
                // every LDS.32 twice (no cross-chain CSE) - 32 shared loads per lane
                // per group. Loading once and feeding both chains halves that to 16.
                // BIT-EXACT: same addresses, same values, only fewer loads; s_act is
                // read-only after the __syncthreads() above.
                float sa[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) sa[i] = s_act[j + i];
                // ---- gate chain: row `row` of the `b`/`bsc` pair ----
                const float gsc = __uint_as_float(((uint32_t)g_srow[j >> 5]) << 23);
                // q = this lane's 8 gate bytes inside the logical row; q + (g2<<8)
                // + (lane<<3) is a multiple of 8. The pool base is a 256-byte
                // aligned device allocation, every per-expert stride is a multiple
                // of 8 (all six per-expert tensor sizes here are multiples of 8
                // because dim % 512 == 0 gives dim/2 = 256k and dim/32 = 16k bytes
                // per row), and lane<<3 / g2<<8 are multiples of 8.
                //
                // PLAIN layout: ONE LDG.64 for the gate (here) and one for the up
                // (below); gw.x/gw.y are the bytes at gp and gp+4, BIT-EXACT vs
                // two LDG.32s.
                // INTERLEAVED layout (ILV): the gate chunk sits at 2*q and the
                // matching up chunk at 2*q + 8, so ONE LDG.128 fetches both -
                // v.x/v.y are the gate pair and v.z/v.w the up pair, the very same
                // 16 bytes the four LDG.32s returned. 2*q is 16-byte aligned for
                // the same reason q is 8-byte aligned. Same decode, same fma
                // chains, half the load instructions.
                const int q = (g2 << 8) + (lane << 3);
                uint32_t gw0, gw1, uw0 = 0u, uw1 = 0u;
                if (ILV) {
                    const uint4 v4 = *reinterpret_cast<const uint4*>(g_row + (size_t)2 * q);
                    gw0 = v4.x; gw1 = v4.y; uw0 = v4.z; uw1 = v4.w;
                } else {
                    const uint2 gw = *reinterpret_cast<const uint2*>(g_row + q);
                    gw0 = gw.x; gw1 = gw.y;
                }
                float gp0 = 0.f, gp1 = 0.f, gp2 = 0.f, gp3 = 0.f;
                const float2 gt0 = s_lut2[gw0 & 0xFFu];
                const float2 gt1 = s_lut2[(gw0 >> 8) & 0xFFu];
                const float2 gt2 = s_lut2[(gw0 >> 16) & 0xFFu];
                const float2 gt3 = s_lut2[(gw0 >> 24) & 0xFFu];
                gp0 = fmaf(sa[0], gt0.x, gp0);
                gp1 = fmaf(sa[1], gt0.y, gp1);
                gp2 = fmaf(sa[2], gt1.x, gp2);
                gp3 = fmaf(sa[3], gt1.y, gp3);
                gp0 = fmaf(sa[4], gt2.x, gp0);
                gp1 = fmaf(sa[5], gt2.y, gp1);
                gp2 = fmaf(sa[6], gt3.x, gp2);
                gp3 = fmaf(sa[7], gt3.y, gp3);
                const float2 gu0 = s_lut2[gw1 & 0xFFu];
                const float2 gu1 = s_lut2[(gw1 >> 8) & 0xFFu];
                const float2 gu2 = s_lut2[(gw1 >> 16) & 0xFFu];
                const float2 gu3 = s_lut2[(gw1 >> 24) & 0xFFu];
                gp0 = fmaf(sa[8], gu0.x, gp0);
                gp1 = fmaf(sa[9], gu0.y, gp1);
                gp2 = fmaf(sa[10], gu1.x, gp2);
                gp3 = fmaf(sa[11], gu1.y, gp3);
                gp0 = fmaf(sa[12], gu2.x, gp0);
                gp1 = fmaf(sa[13], gu2.y, gp1);
                gp2 = fmaf(sa[14], gu3.x, gp2);
                gp3 = fmaf(sa[15], gu3.y, gp3);
                g = fmaf(gsc, (gp0 + gp1) + (gp2 + gp3), g);
                // ---- up chain: row `row` of the `b_hi`/`bhs` pair ----
                const float usc = __uint_as_float(((uint32_t)u_srow[j >> 5]) << 23);
                // INTERLEAVED: the up pair already arrived in the LDG.128 above
                // (uw0/uw1). PLAIN: the second LDG.64 of the row, identical
                // 8-byte alignment argument; uw.x/uw.y == the old uw0/uw1.
                if (!ILV) {
                    const uint2 uw = *reinterpret_cast<const uint2*>(u_row + q);
                    uw0 = uw.x; uw1 = uw.y;
                }
                float up0 = 0.f, up1 = 0.f, up2 = 0.f, up3 = 0.f;
                const float2 ut0 = s_lut2[uw0 & 0xFFu];
                const float2 ut1 = s_lut2[(uw0 >> 8) & 0xFFu];
                const float2 ut2 = s_lut2[(uw0 >> 16) & 0xFFu];
                const float2 ut3 = s_lut2[(uw0 >> 24) & 0xFFu];
                up0 = fmaf(sa[0], ut0.x, up0);
                up1 = fmaf(sa[1], ut0.y, up1);
                up2 = fmaf(sa[2], ut1.x, up2);
                up3 = fmaf(sa[3], ut1.y, up3);
                up0 = fmaf(sa[4], ut2.x, up0);
                up1 = fmaf(sa[5], ut2.y, up1);
                up2 = fmaf(sa[6], ut3.x, up2);
                up3 = fmaf(sa[7], ut3.y, up3);
                const float2 uu0 = s_lut2[uw1 & 0xFFu];
                const float2 uu1 = s_lut2[(uw1 >> 8) & 0xFFu];
                const float2 uu2 = s_lut2[(uw1 >> 16) & 0xFFu];
                const float2 uu3 = s_lut2[(uw1 >> 24) & 0xFFu];
                up0 = fmaf(sa[8], uu0.x, up0);
                up1 = fmaf(sa[9], uu0.y, up1);
                up2 = fmaf(sa[10], uu1.x, up2);
                up3 = fmaf(sa[11], uu1.y, up3);
                up0 = fmaf(sa[12], uu2.x, up0);
                up1 = fmaf(sa[13], uu2.y, up1);
                up2 = fmaf(sa[14], uu3.x, up2);
                up3 = fmaf(sa[15], uu3.y, up3);
                u = fmaf(usc, (up0 + up1) + (up2 + up3), u);
            }
            for (int off = 16; off > 0; off >>= 1) {
                g += __shfl_xor_sync(0xFFFFFFFFu, g, off);
                u += __shfl_xor_sync(0xFFFFFFFFu, u, off);
            }
            // ---- K-split cross-half merge (ksplit > 1 only) ----
            // Each half' lane 0 now holds its own half's complete (gate, up) sum.
            // Park it in smem, one barrier, then half 0 folds the partners in
            // ASCENDING half order. The merge is deterministic and uses __fadd_rn
            // so --use_fast_math cannot reassociate it. It happens BEFORE the
            // clamp/silu (the clamp is on the summed gate/up, not on a partial).
            // This is where the summation order changes vs ksplit==1:
            //   (g0+..+g4) + (g5+..+g9)   instead of   g0+..+g9 serially.
            // Mathematically equivalent, not bit-identical (~1e-7/layer) - that
            // is the parity cost of K-split, hence DSV41_GATEUP_KSPLIT defaults
            // to 1 (OFF) until the text A/B validates it.
            if (ksplit > 1) {
                if (lane == 0) s_ks[warp] = make_float2(g, u);
                __syncthreads();
                if (half == 0 && lane == 0) {
                    float2 acc = make_float2(g, u);
                    for (int h = 1; h < ksplit; ++h) {
                        const float2 o = s_ks[warp + h];
                        acc.x = __fadd_rn(acc.x, o.x);
                        acc.y = __fadd_rn(acc.y, o.y);
                    }
                    g = acc.x; u = acc.y;
                }
            }
            // Only half 0 of a live row writes it (half>0 have their partials
            // already folded into half 0). Inactive warps wrote a valid row 0
            // number above but must not clobber the real out[row].
            if (active && half == 0 && lane == 0) {
                if (limit > 0.f) {
                    g = fminf(g, limit);                     // gate clamp
                    u = fminf(fmaxf(u, -limit), limit);      // up clamp
                }
                out[(size_t)row] = (g / (1.f + expf(-g))) * u;   // silu(gate) * up
            }
            continue;
        }
        // gate/up split: rows < b_split read the `b` pair, the rest the `b_hi` pair
        const bool hi = (b_split > 0) && (row >= b_split);
        const int r = hi ? (row - b_split) : row;
        const uint8_t* bb = hi ? bhi_use : b_use;
        const uint8_t* bb_s = hi ? bhs_use : bsc_use;
        const uint8_t* brow = bb + (size_t)r * kbytes;
        const uint8_t* srow = bb_s + (size_t)r * ksc;

        float acc = 0.f;
        if (vec == 2) {
            // Same 256-values-per-group shape as the vectorised branch, but the
            // unpack is a shared lookup and the accumulation is split four ways so
            // the dependency chain is forty fmas deep instead of a hundred and sixty.
            // Sixteen values per lane per group: 32 lanes * 16 = 512 values, and 512
            // packed fp4 values are 256 bytes, so the group stride is k >> 9 and the
            // byte base advances by g << 8. Sixteen is also exactly half a 32-value
            // scale block, so one scale lookup covers the whole lane iteration.
            const int nv2 = k >> 9;
            const int off2 = lane << 3;                  // 16 values = 8 bytes per lane
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            // Compiler-directed unroll (same treatment that worked on the fp8
            // gemv: let nvcc choose the register strategy, keep single-chain
            // source semantics).
#pragma unroll 2
            for (int g = 0; g < nv2; ++g) {
                const int j = (g << 9) + (lane << 4);
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                const uint8_t* bp = brow + (g << 8) + off2;
                const uint32_t w0 = *reinterpret_cast<const uint32_t*>(bp);
                const uint32_t w1 = *reinterpret_cast<const uint32_t*>(bp + 4);
                // One scale multiply per accumulator instead of one per element:
                // sc is a power of two (the ue8m0 exponent becomes the float
                // exponent here), so the sixteen terms of this group can be summed
                // first and scaled once. 20 FMA per 16 elements instead of 16 FMA
                // + 16 MUL, which matters because this kernel's issue slots are
                // ~80 percent stalled on the FMA port with only 3.2 blocks/SM.
                // The float2 table halves the lookups: one LDS.64 gives both
                // nibbles of a byte (16 LDS.32+16 shifts -> 8 LDS.64).
                float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
                const float2 t0 = s_lut2[w0 & 0xFFu];
                const float2 t1 = s_lut2[(w0 >> 8) & 0xFFu];
                const float2 t2 = s_lut2[(w0 >> 16) & 0xFFu];
                const float2 t3 = s_lut2[(w0 >> 24) & 0xFFu];
                p0 = fmaf(s_act[j + 0], t0.x, p0);
                p1 = fmaf(s_act[j + 1], t0.y, p1);
                p2 = fmaf(s_act[j + 2], t1.x, p2);
                p3 = fmaf(s_act[j + 3], t1.y, p3);
                p0 = fmaf(s_act[j + 4], t2.x, p0);
                p1 = fmaf(s_act[j + 5], t2.y, p1);
                p2 = fmaf(s_act[j + 6], t3.x, p2);
                p3 = fmaf(s_act[j + 7], t3.y, p3);
                const float2 u0 = s_lut2[w1 & 0xFFu];
                const float2 u1 = s_lut2[(w1 >> 8) & 0xFFu];
                const float2 u2 = s_lut2[(w1 >> 16) & 0xFFu];
                const float2 u3 = s_lut2[(w1 >> 24) & 0xFFu];
                p0 = fmaf(s_act[j + 8], u0.x, p0);
                p1 = fmaf(s_act[j + 9], u0.y, p1);
                p2 = fmaf(s_act[j + 10], u1.x, p2);
                p3 = fmaf(s_act[j + 11], u1.y, p3);
                p0 = fmaf(s_act[j + 12], u2.x, p0);
                p1 = fmaf(s_act[j + 13], u2.y, p1);
                p2 = fmaf(s_act[j + 14], u3.x, p2);
                p3 = fmaf(s_act[j + 15], u3.y, p3);
                a0 = fmaf(sc, p0, a0);
                a1 = fmaf(sc, p1, a1);
                a2 = fmaf(sc, p2, a2);
                a3 = fmaf(sc, p3, a3);
            }
            acc = (a0 + a1) + (a2 + a3);
            for (int j = (nv2 << 9) + lane * 2; j < k; j += 64) {
                const uint8_t byte = brow[j >> 1];
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                const float2 t = s_lut2[byte];
                acc += s_act[j] * (t.x * sc);
                acc += s_act[j + 1] * (t.y * sc);
            }
        } else if (vec) {
            // 32 lanes * 8 values = 256 values per iteration, four scale blocks.
            const int nv = k >> 8;              // full 256-value iterations
            const int off = lane << 2;          // byte offset of this lane's uint32
            for (int g = 0; g < nv; ++g) {
                const int j = (g << 8) + (lane << 3);
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                // 256 packed values are 128 bytes, so group g starts at g*128, not g*64.
            const uint32_t word = *reinterpret_cast<const uint32_t*>(brow + (g << 7) + off);
                acc += s_act[j + 0] * (dsv41_e2m1_to_f((uint8_t)(word & 0xFu)) * sc);
                acc += s_act[j + 1] * (dsv41_e2m1_to_f((uint8_t)((word >> 4) & 0xFu)) * sc);
                acc += s_act[j + 2] * (dsv41_e2m1_to_f((uint8_t)((word >> 8) & 0xFu)) * sc);
                acc += s_act[j + 3] * (dsv41_e2m1_to_f((uint8_t)((word >> 12) & 0xFu)) * sc);
                acc += s_act[j + 4] * (dsv41_e2m1_to_f((uint8_t)((word >> 16) & 0xFu)) * sc);
                acc += s_act[j + 5] * (dsv41_e2m1_to_f((uint8_t)((word >> 20) & 0xFu)) * sc);
                acc += s_act[j + 6] * (dsv41_e2m1_to_f((uint8_t)((word >> 24) & 0xFu)) * sc);
                acc += s_act[j + 7] * (dsv41_e2m1_to_f((uint8_t)((word >> 28) & 0xFu)) * sc);
            }
            for (int j = (nv << 8) + lane * 2; j < k; j += 64) {
                const uint8_t byte = brow[j >> 1];
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                acc += s_act[j] * (dsv41_e2m1_to_f(byte & 0xFu) * sc);
                acc += s_act[j + 1] * (dsv41_e2m1_to_f((uint8_t)(byte >> 4)) * sc);
            }
        } else {
            for (int j = lane * 2; j < k; j += 64) {
                const uint8_t byte = brow[j >> 1];
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                const float w0 = dsv41_e2m1_to_f(byte & 0xFu) * sc;
                const float w1 = dsv41_e2m1_to_f((uint8_t)(byte >> 4)) * sc;
                acc += s_act[j] * w0;
                acc += s_act[j + 1] * w1;
            }
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) {
            float x = acc;
            if (epi_mode == 1) {
                if (limit > 0.f) {
                    if (row < b_split) x = fminf(x, limit);
                    else x = fminf(fmaxf(x, -limit), limit);
                }
            } else if (epi_mode == 2 || epi_mode == 3) {
                if (rw != nullptr) x *= rw[0];
            }
            if (epi_mode == 3) out[(size_t)row] += x;
            else out[(size_t)row] = x;
        }
    }
}

// Fixed-order reduction of the batched down scratch:
//   out[i] = ((0 + part[0][i]) + part[1][i]) + ... + part[slots-1][i]
// i.e. the SAME ascending-slot order the sequential loop's `out[i] += x` used,
// starting from 0.0f. fp addition is not associative, so this order is part of
// the numerical contract - do NOT parallelise the slot loop.
__global__ void moe_down_reduce_kernel(const float* __restrict__ part, float* __restrict__ out,
                                       int n, int slots) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < slots; ++s) acc += part[(size_t)s * n + i];
        out[i] = acc;
    }
}

// ============================================================================
// down + reduce FUSED (DSV41_DOWN_FUSE on the Rust side, DEFAULT ON: chain_dev.rs `down_fuse()` is
// `.unwrap_or(true)` and the nsys v3 profile sees the fused kernel 40x/step.
// f3b1be1's OFF default was later flipped back; this comment was stale.)
// ============================================================================
// The batched down direction used to cost TWO launches per layer: the per-slot
// down GEMV (epi_mode 2, writing the [slots][dim] scratch) and
// moe_down_reduce_kernel (summing that scratch in ascending slot order). This
// kernel does both in ONE launch: each warp owns its output row, walks the slots
// SERIALLY and keeps the running total in a register, so the scratch never
// exists and the second launch disappears.
//
// NUMERIC CONTRACT - bit-identical to the pair it replaces:
//   * the per-slot K dot product and the butterfly shuffle below are the
//     VERBATIM source of expert_gemv_fp4_batched_kernel (same lane order, same
//     group order, same scale-multiply shape, same `#pragma unroll 2`), so each
//     slot's c_s is bit-identical;
//   * the slot loop is serial and ASCENDING inside the warp - there is no
//     blockIdx.y and no cross-slot parallelism anywhere - which reproduces
//       out[row] = ((0 + c_0*rw_0) + c_1*rw_1) + ...
//     the fixed order of moe_down_reduce_kernel from the zeroed `o`. fp addition
//     is not associative, so this serialisation IS the contract: never split the
//     slot loop across lanes or CTAs.
//   * the per-slot product goes through __fmul_rn and the accumulation through
//     __fadd_rn (see the epilogue): a bare `tot += acc * rwv` contracts into an
//     FMA under --use_fast_math, which rounds AFTER the add and differs in the
//     last bit from the scratch path's fl(c_s * rw_s) followed by an add.
//
// grid = (ceil(n_total / nwarps),) - ONE dimension only. One warp per output
// row and `nwarps` rows per block, exactly the row assignment of the two kernels
// it replaces. STAGED selects the activation staging (see the launcher): true
// puts every slot's [0,k) slice in smem once per block, false reads each slot's
// slice from global.
template <bool STAGED>
__global__ void expert_gemv_fp4_down_reduce_kernel(
    const float* __restrict__ act_base, long act_stride, float* __restrict__ out, int n_total,
    int k, int slots, const float* __restrict__ row_weight, long rw_stride,
    const uint8_t* __restrict__ w2_base, long w2_stride, const uint8_t* __restrict__ w2s_base,
    long w2s_stride, const int* __restrict__ ids, int vec) {
    // STAGED: s_smem is [slots][k] slot-major (slot s starts at s_smem + s*k) and
    // the 256-entry LUT follows it. Fallback: the LUT alone lives in smem and
    // each slot's activation is read straight from global - same numbers, more
    // L2 traffic (that path exists only for a slots*inter that outgrows the
    // device's opt-in smem ceiling).
    extern __shared__ float s_smem[];
    float2* s_lut2 = reinterpret_cast<float2*>(s_smem + (STAGED ? (size_t)slots * (size_t)k : 0));
    const int kbytes = k >> 1;   // packed bytes per row
    const int ksc = k >> 5;      // e8m0 scales per row
    const int nwarps = (blockDim.x + 31) >> 5;
    // PDL (DSV41_PDL, see dsv41_experts_pdl_or_plain above): the launcher may
    // have launched this grid with programmatic stream serialization, so the
    // grid is already resident here and this call is what makes the PRODUCER's
    // writes visible. The producer is the gate/up batched launch, which wrote
    // the swiglu'd f32 activation `act_base`. The sync MUST stay before the
    // first read of it -- the STAGED staging loop below, or the per-slot
    // `act_base` reads inside the row loop for the non-staged fallback (both
    // are after this point either way).
    //
    // Producer-INDEPENDENT work hoisted above the sync: the 256-entry e2m1 LUT
    // (built from device constants, written to this CTA's own smem, published by
    // the existing __syncthreads()) and the pitch/register setup. `ids` and
    // `row_weight` are the router's output (several kernels before the producer,
    // hence already flushed) and are not read until the row loop.
    //
    // No-op on a plain launch (DSV41_PDL=0).
    if (threadIdx.x < 256)
        s_lut2[threadIdx.x] = make_float2(dsv41_e2m1_to_f((uint8_t)(threadIdx.x & 0xF)),
                                          dsv41_e2m1_to_f((uint8_t)(threadIdx.x >> 4)));
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
    if (STAGED) {
        // ONE cooperative pass stages every slot's activation. `act_stride` is
        // the caller's slice pitch and only the first `k` floats of each slice
        // are read - today the pitch is 2*inter (the swiglu half of the gate/up
        // output); once gate_up+swiglu fusion shrinks that slice to inter, the
        // caller passes the smaller pitch and this loop is unchanged.
        for (int s = 0; s < slots; ++s) {
            const float* src = act_base + (size_t)s * (size_t)act_stride;
            float* dst = s_smem + (size_t)s * (size_t)k;
            for (int j = threadIdx.x; j < k; j += blockDim.x) dst[j] = src[j];
        }
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (int row = blockIdx.x * nwarps + warp; row < n_total; row += gridDim.x * nwarps) {
        float tot = 0.f;
        // Ascending slot loop, never parallel: see the contract above.
        for (int slot = 0; slot < slots; ++slot) {
            const float* s_act = STAGED ? (s_smem + (size_t)slot * (size_t)k)
                                        : (act_base + (size_t)slot * (size_t)act_stride);
            // Per-slot derivation identical to expert_gemv_fp4_batched_kernel,
            // except that the weight row is selected here: this kernel has no
            // blockIdx.y, every warp owns its whole row.
            const float rwv =
                (row_weight != nullptr) ? row_weight[(size_t)slot * (size_t)rw_stride] : 1.f;
            const size_t e = (size_t)ids[slot];
            const uint8_t* brow = w2_base + e * (size_t)w2_stride + (size_t)row * kbytes;
            const uint8_t* srow = w2s_base + e * (size_t)w2s_stride + (size_t)row * ksc;

            float acc = 0.f;
            if (vec == 3) {
                // 4 values per lane for k < 512, where the vec==2 main loop below
                // cannot run at all (nv2 = k >> 9 = 0 at the production
                // k = inter_local = 320). One LDG.U16 = 2 packed bytes = 4 nibbles,
                // one LDS.128 = the 4 activations, two LDS.64 = the LUT pairs:
                // 5 L1TEX ops per 4 values against 10 in the 2-value tail. The four
                // values of a group sit inside ONE 32-value scale block (j = lane*4,
                // block = j >> 5 = lane >> 3), so `sc` multiplies the accumulator once
                // per group instead of once per element: 6 FP ops per 4 values.
                const int nv4 = k >> 7;
                float a0 = 0.f, a1 = 0.f;
                for (int g = 0; g < nv4; ++g) {
                    const int j = (g << 7) + (lane << 2);
                    const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                    const uint16_t w =
                        *reinterpret_cast<const uint16_t*>(brow + (g << 6) + (lane << 1));
                    const float4 av = *reinterpret_cast<const float4*>(s_act + j);
                    const float2 t0 = s_lut2[w & 0xFFu];
                    const float2 t1 = s_lut2[(w >> 8) & 0xFFu];
                    float p0 = av.x * t0.x;
                    p0 = fmaf(av.y, t0.y, p0);
                    float p1 = av.z * t1.x;
                    p1 = fmaf(av.w, t1.y, p1);
                    a0 = fmaf(sc, p0, a0);
                    a1 = fmaf(sc, p1, a1);
                }
                acc = a0 + a1;
                for (int j = (nv4 << 7) + lane * 2; j < k; j += 64) {
                    const uint8_t byte = brow[j >> 1];
                    const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                    const float2 t = s_lut2[byte];
                    acc += s_act[j] * (t.x * sc);
                    acc += s_act[j + 1] * (t.y * sc);
                }
            } else if (vec == 2) {
                // Same 256-values-per-group shape as the vectorised branch, but the
                // unpack is a shared lookup and the accumulation is split four ways so
                // the dependency chain is forty fmas deep instead of a hundred and sixty.
                // Sixteen values per lane per group: 32 lanes * 16 = 512 values, and 512
                // packed fp4 values are 256 bytes, so the group stride is k >> 9 and the
                // byte base advances by g << 8. Sixteen is also exactly half a 32-value
                // scale block, so one scale lookup covers the whole lane iteration.
                const int nv2 = k >> 9;
                const int off2 = lane << 3;                  // 16 values = 8 bytes per lane
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                // Compiler-directed unroll (same treatment that worked on the fp8
                // gemv: let nvcc choose the register strategy, keep single-chain
                // source semantics).
#pragma unroll 2
                for (int g = 0; g < nv2; ++g) {
                    const int j = (g << 9) + (lane << 4);
                    const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                    const uint8_t* bp = brow + (g << 8) + off2;
                    const uint32_t w0 = *reinterpret_cast<const uint32_t*>(bp);
                    const uint32_t w1 = *reinterpret_cast<const uint32_t*>(bp + 4);
                    // One scale multiply per accumulator instead of one per element:
                    // sc is a power of two (the ue8m0 exponent becomes the float
                    // exponent here), so the sixteen terms of this group can be summed
                    // first and scaled once. 20 FMA per 16 elements instead of 16 FMA
                    // + 16 MUL, which matters because this kernel's issue slots are
                    // ~80 percent stalled on the FMA port with only 3.2 blocks/SM.
                    // The float2 table halves the lookups: one LDS.64 gives both
                    // nibbles of a byte (16 LDS.32+16 shifts -> 8 LDS.64).
                    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
                    const float2 t0 = s_lut2[w0 & 0xFFu];
                    const float2 t1 = s_lut2[(w0 >> 8) & 0xFFu];
                    const float2 t2 = s_lut2[(w0 >> 16) & 0xFFu];
                    const float2 t3 = s_lut2[(w0 >> 24) & 0xFFu];
                    p0 = fmaf(s_act[j + 0], t0.x, p0);
                    p1 = fmaf(s_act[j + 1], t0.y, p1);
                    p2 = fmaf(s_act[j + 2], t1.x, p2);
                    p3 = fmaf(s_act[j + 3], t1.y, p3);
                    p0 = fmaf(s_act[j + 4], t2.x, p0);
                    p1 = fmaf(s_act[j + 5], t2.y, p1);
                    p2 = fmaf(s_act[j + 6], t3.x, p2);
                    p3 = fmaf(s_act[j + 7], t3.y, p3);
                    const float2 u0 = s_lut2[w1 & 0xFFu];
                    const float2 u1 = s_lut2[(w1 >> 8) & 0xFFu];
                    const float2 u2 = s_lut2[(w1 >> 16) & 0xFFu];
                    const float2 u3 = s_lut2[(w1 >> 24) & 0xFFu];
                    p0 = fmaf(s_act[j + 8], u0.x, p0);
                    p1 = fmaf(s_act[j + 9], u0.y, p1);
                    p2 = fmaf(s_act[j + 10], u1.x, p2);
                    p3 = fmaf(s_act[j + 11], u1.y, p3);
                    p0 = fmaf(s_act[j + 12], u2.x, p0);
                    p1 = fmaf(s_act[j + 13], u2.y, p1);
                    p2 = fmaf(s_act[j + 14], u3.x, p2);
                    p3 = fmaf(s_act[j + 15], u3.y, p3);
                    a0 = fmaf(sc, p0, a0);
                    a1 = fmaf(sc, p1, a1);
                    a2 = fmaf(sc, p2, a2);
                    a3 = fmaf(sc, p3, a3);
                }
                // The two wider tails are NOT equivalent - re-measured on the bench
                // node (sm_103a, nvcc 13.2, k=320, dim=7168, slots=8, 256 threads,
                // 896 blocks, 5 interleaved rounds, 2-value tail = 1.00):
                //   uint16 / 4-value (mode 3): 0.90 @ 40 regs + launch_bounds(256,6)
                //                             0.87 @ 56 regs / 4 blocks/SM  <- fastest
                //   uint32 / 8-value (mode 4 = the 01291b2 form): 1.03 @ 40 regs,
                //                             0.97 @ 62 regs
                // => the loser is the 8-VALUE loop, not "wider loads". 4 values is the
                // last width whose working set (1 uint16 + 1 float4 + 2 float2) still
                // fits a 40-register schedule; 8 values buys fewer instructions than it
                // pays for in registers.
                // The earlier occupancy attribution ("40 regs is the red line, 54 regs
                // -> 4 blocks/SM -> +49%") does NOT reproduce: the 56-reg/4-block
                // mode 3 arm is the FASTEST of all. Do not attribute the 01291b2
                // regression to occupancy without re-measuring it.
                // Bit-parity note: mode 3's lane map differs from mode 2's, so the
                // unfused path needs the same branch - expert_gemv_fp4_batched_kernel
                // carries a verbatim mirror and both read g_down_fp4_mode.
                acc = (a0 + a1) + (a2 + a3);
                for (int j = (nv2 << 9) + lane * 2; j < k; j += 64) {
                    const uint8_t byte = brow[j >> 1];
                    const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                    const float2 t = s_lut2[byte];
                    acc += s_act[j] * (t.x * sc);
                    acc += s_act[j + 1] * (t.y * sc);
                }
            } else if (vec) {
                // 32 lanes * 8 values = 256 values per iteration, four scale blocks.
                const int nv = k >> 8;              // full 256-value iterations
                const int off = lane << 2;          // byte offset of this lane's uint32
                for (int g = 0; g < nv; ++g) {
                    const int j = (g << 8) + (lane << 3);
                    const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                    // 256 packed values are 128 bytes, so group g starts at g*128, not g*64.
                const uint32_t word = *reinterpret_cast<const uint32_t*>(brow + (g << 7) + off);
                    acc += s_act[j + 0] * (dsv41_e2m1_to_f((uint8_t)(word & 0xFu)) * sc);
                    acc += s_act[j + 1] * (dsv41_e2m1_to_f((uint8_t)((word >> 4) & 0xFu)) * sc);
                    acc += s_act[j + 2] * (dsv41_e2m1_to_f((uint8_t)((word >> 8) & 0xFu)) * sc);
                    acc += s_act[j + 3] * (dsv41_e2m1_to_f((uint8_t)((word >> 12) & 0xFu)) * sc);
                    acc += s_act[j + 4] * (dsv41_e2m1_to_f((uint8_t)((word >> 16) & 0xFu)) * sc);
                    acc += s_act[j + 5] * (dsv41_e2m1_to_f((uint8_t)((word >> 20) & 0xFu)) * sc);
                    acc += s_act[j + 6] * (dsv41_e2m1_to_f((uint8_t)((word >> 24) & 0xFu)) * sc);
                    acc += s_act[j + 7] * (dsv41_e2m1_to_f((uint8_t)((word >> 28) & 0xFu)) * sc);
                }
                for (int j = (nv << 8) + lane * 2; j < k; j += 64) {
                    const uint8_t byte = brow[j >> 1];
                    const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                    acc += s_act[j] * (dsv41_e2m1_to_f(byte & 0xFu) * sc);
                    acc += s_act[j + 1] * (dsv41_e2m1_to_f((uint8_t)(byte >> 4)) * sc);
                }
            } else {
                for (int j = lane * 2; j < k; j += 64) {
                    const uint8_t byte = brow[j >> 1];
                    const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                    const float w0 = dsv41_e2m1_to_f(byte & 0xFu) * sc;
                    const float w1 = dsv41_e2m1_to_f((uint8_t)(byte >> 4)) * sc;
                    acc += s_act[j] * w0;
                    acc += s_act[j + 1] * w1;
                }
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
            // Same contract as the scratch path: the product is a ROUNDED mul
            // (a bare `tot += acc * rwv` folds into an FMA under --use_fast_math
            // and rounds after the add), then the ascending add.
            tot = __fadd_rn(tot, __fmul_rn(acc, rwv));
        }
        if (lane == 0) out[(size_t)row] = tot;
    }
}

// --------------------------------------------------------------- launchers
inline cudaError_t launch_mxf4(const uint8_t* a, const float* a_scale, const float* a_f32,
                               const uint8_t* b, const uint8_t* b_scale, const uint8_t* b_hi,
                               const uint8_t* b_hi_scale, float* out, int rows, int n_total, int k,
                               int b_split, int epi_mode, float limit, const float* row_weight,
                               bool aq, cudaStream_t s) {
    if (rows <= 0 || n_total <= 0 || k <= 0) return cudaSuccess;
    if (k % kAtomK != 0) return cudaErrorInvalidValue;  // K must be a multiple of 64
    // M=1 (decode) takes the GEMV: the tcgen05 tile is M=128 by hardware, so the
    // tensor-core path is 128x redundant here and its grid collapses to a handful
    // of blocks. The GEMV is bandwidth-bound with one warp per output row.
    // rows == 1 AND !aq: the gate/up path only. Extending this to the down path
    // (AQ=true, epi_mode 3, b_split=-1) was tried and CORRUPTED the model — one
    // prompt returned all zeros and others hit an illegal memory access — so it is
    // reverted until the down call's exact arguments are worked out (its a_f32 is
    // the expert's act buffer and its n_total/k are dim/inter, not the gate/up
    // shapes). NOTE: a boot-time env read here would also break CUDA graph
    // capture; cache it in a static if a knob is needed.
    if (rows == 1 && getenv("DSV41_NO_GEMV_FP4") == nullptr) {
        const int warps = 8;
        const int cta = warps * 32;
        const int blocks = (n_total + warps - 1) / warps;
        expert_gemv_fp4_kernel<<<blocks, cta, (size_t)k * sizeof(float), s>>>(
            a_f32, a, a_scale, b, b_scale, b_hi, b_hi_scale, out, n_total, k, b_split, epi_mode,
            limit, row_weight, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0);
        return cudaGetLastError();
    }
    const dim3 grid((unsigned)((n_total + kNTile - 1) / kNTile),
                    (unsigned)((rows + kMTile - 1) / kMTile));
    if (aq)
        mxf4_gemm_kernel<true><<<grid, kThreads, 0, s>>>(a, a_scale, a_f32, b, b_scale, b_hi,
                                                         b_hi_scale, out, rows, n_total, k, b_split,
                                                         epi_mode, limit, row_weight, nullptr, 0,
                                                         nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0);
    else
        mxf4_gemm_kernel<false><<<grid, kThreads, 0, s>>>(a, a_scale, a_f32, b, b_scale, b_hi,
                                                          b_hi_scale, out, rows, n_total, k,
                                                          b_split, epi_mode, limit, row_weight,
                                                          nullptr, 0, nullptr, 0, nullptr, 0,
                                                          nullptr, 0, nullptr, 0);
    return cudaGetLastError();
}

}  // namespace

// ============================================================================
// LOAD-TIME gate/up interleave (DSV41_EXPERT_ILV, default ON on the Rust side)
// ============================================================================
// Rewrites an expert's w1 (gate) and w3 (up) fp4 row blocks into ONE region with
// an 8-byte granule alternation:
//     dst[16*i .. 16*i+8) = gate[8*i .. 8*i+8)
//     dst[16*i+8 .. 16*i+16) = up[8*i .. 8*i+8)
// so the fused gate/up GEMV fetches a gate chunk and its matching up chunk with
// ONE LDG.128 instead of two LDG.64s (see expert_gemv_fp4_batched_kernel<ILV>).
// PURE PERMUTATION: every byte keeps its value and its position inside its own
// 8-byte granule, so the kernel decodes exactly the same numbers and the mxf4
// accumulations are untouched — bit-identical, only the load count halves.
//
// `bytes` is the size of EACH side (gate == up == the local [inter, dim/2] byte
// count); it must be a multiple of 8. `dst` must not overlap `g`/`u`.
namespace {
__global__ void interleave_gateup_fp4_kernel(const uint2* __restrict__ g,
                                             const uint2* __restrict__ u,
                                             uint2* __restrict__ dst, long n8) {
    const long stride = (long)blockDim.x * (long)gridDim.x;
    // read-only sources, one write pair per iteration: the loop lets any grid
    // size cover the tensor, and every thread writes DISJOINT 16-byte slots.
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
        dst[2 * i] = g[i];
        dst[2 * i + 1] = u[i];
    }
}
}  // namespace

extern "C" int dsv41_interleave_gateup_fp4(const uint8_t* g, const uint8_t* u, uint8_t* dst,
                                           long bytes, cudaStream_t stream) {
    if (bytes <= 0) return (int)cudaSuccess;
    if ((bytes & 7) != 0) return (int)cudaErrorInvalidValue;   // 8-byte granule only
    if (g == nullptr || u == nullptr || dst == nullptr) return (int)cudaErrorInvalidValue;
    const long n8 = bytes >> 3;
    const int threads = 256;
    long nb = (n8 + threads - 1) / threads;
    if (nb > 4096) nb = 4096;                                  // grid-stride covers the rest
    interleave_gateup_fp4_kernel<<<(unsigned)nb, threads, 0, stream>>>(
        reinterpret_cast<const uint2*>(g), reinterpret_cast<const uint2*>(u),
        reinterpret_cast<uint2*>(dst), n8);
    return (int)cudaGetLastError();
}

// ============================================================================
// Test hook (used by kernels/cuda/tests_tcgen05_mxf4.cu; not part of the ABI).
// ============================================================================
extern "C" int dsv41_mxf4_test_gemm(const uint8_t* a, const float* a_scale, const uint8_t* b,
                                    const uint8_t* b_scale, float* out, int m, int n, int k,
                                    cudaStream_t s) {
    return (int)launch_mxf4(a, a_scale, nullptr, b, b_scale, b, b_scale, out, m, n, k, -1, 0,
                            0.f, nullptr, false, s);
}

// ============================================================================
// ABI: dsv41_expert_gate_up_fp4 — [rows, dim] fp4 x {W1, W3}[inter, dim] fp4
//      -> [rows, 2*inter] (gate first, then up), clamps applied with `limit`.
// ============================================================================
extern "C" int dsv41_expert_gate_up_fp4(const uint8_t* a, const float* a_scale,
                                        const uint8_t* w1, const uint8_t* w1_scale,
                                        const uint8_t* w3, const uint8_t* w3_scale, float* out,
                                        int rows, int dim, int inter, float limit,
                                        cudaStream_t stream) {
    if (rows <= 0 || dim <= 0 || inter <= 0) return (int)cudaErrorInvalidValue;
    const int n_total = 2 * inter;
    return (int)launch_mxf4(a, a_scale, nullptr, w1, w1_scale, w3, w3_scale, out, rows, n_total,
                            dim, inter, 1, limit, nullptr, false, stream);
}

// ============================================================================
// ABI: dsv41_expert_down_fp4 — [rows, inter] f32 act x W2[dim, inter] fp4
//      -> [rows, dim], scaled by the per-row routing weight.
// ============================================================================
// Indirect (graph-friendly) launcher: B pointers come from the pools + the
// device-side expert id instead of host-computed pointers.
inline cudaError_t launch_mxf4_indirect(const uint8_t* a, const float* a_scale, const float* a_f32,
                                        float* out, int rows, int n_total, int k, int b_split,
                                        int epi_mode, float limit, const float* row_weight,
                                        bool aq, const uint8_t* b_base, long b_stride,
                                        const uint8_t* bs_base, long bs_stride,
                                        const uint8_t* bh_base, long bh_stride,
                                        const uint8_t* bhs_base, long bhs_stride,
                                        const int* ids, int slot, cudaStream_t s) {
    if (rows <= 0 || n_total <= 0 || k <= 0) return cudaSuccess;
    if (k % kAtomK != 0) return cudaErrorInvalidValue;
    // M=1 (decode): the tcgen05 tile is M=128 by hardware, so the tensor-core path
    // is 128x redundant and its grid collapses to a handful of blocks. The GEMV is
    // bandwidth-bound with one warp per output row. (Same dispatch as launch_mxf4.)
    // rows == 1 covers BOTH the expert gate/up (fp4 activation, AQ=false) and the
    // expert down (f32 activation, AQ=true, epi_mode 3 accumulating into the MoE
    // buffer). The kernel unpacks either activation form and implements both
    // epilogues, so no separate path is needed for down.
    if (rows == 1 && getenv("DSV41_NO_GEMV_FP4") == nullptr) {
        const int warps = 8;
        const int blocks = (n_total + warps - 1) / warps;
        expert_gemv_fp4_kernel<<<blocks, warps * 32, (size_t)k * sizeof(float), s>>>(
            a_f32, a, a_scale, nullptr, nullptr, nullptr, nullptr, out, n_total, k, b_split,
            epi_mode, limit, row_weight, b_base, b_stride, bs_base, bs_stride, bh_base, bh_stride,
            bhs_base, bhs_stride, ids, slot);
        return cudaGetLastError();
    }
    const dim3 grid((unsigned)((n_total + kNTile - 1) / kNTile),
                    (unsigned)((rows + kMTile - 1) / kMTile));
    if (aq)
        mxf4_gemm_kernel<true><<<grid, kThreads, 0, s>>>(
            a, a_scale, a_f32, nullptr, nullptr, nullptr, nullptr, out, rows, n_total, k, b_split,
            epi_mode, limit, row_weight, b_base, b_stride, bs_base, bs_stride, bh_base, bh_stride,
            bhs_base, bhs_stride, ids, slot);
    else
        mxf4_gemm_kernel<false><<<grid, kThreads, 0, s>>>(
            a, a_scale, a_f32, nullptr, nullptr, nullptr, nullptr, out, rows, n_total, k, b_split,
            epi_mode, limit, row_weight, b_base, b_stride, bs_base, bs_stride, bh_base, bh_stride,
            bhs_base, bhs_stride, ids, slot);
    return cudaGetLastError();
}

// gate_up, indirect: derives w1/w3 (+scales) from the pools and ids[slot].
extern "C" int dsv41_expert_gate_up_fp4_indirect(
    const uint8_t* a, const float* a_scale, float* out, int rows, int dim, int inter, float limit,
    const uint8_t* w1_base, long w1_stride, const uint8_t* w1s_base, long w1s_stride,
    const uint8_t* w3_base, long w3_stride, const uint8_t* w3s_base, long w3s_stride,
    const int* ids, int slot, cudaStream_t stream) {
    return (int)launch_mxf4_indirect(a, a_scale, nullptr, out, rows, 2 * inter, dim, inter, 1, limit,
                                     nullptr, false, w1_base, w1_stride, w1s_base, w1s_stride,
                                     w3_base, w3_stride, w3s_base, w3s_stride, ids, slot, stream);
}

// down, indirect, accumulating into `out` (epi_mode 3).
extern "C" int dsv41_expert_down_fp4_indirect(
    const float* act, float* out, int rows, int dim, int inter, const float* row_weight,
    const uint8_t* w2_base, long w2_stride, const uint8_t* w2s_base, long w2s_stride,
    const int* ids, int slot, cudaStream_t stream) {
    return (int)launch_mxf4_indirect(nullptr, nullptr, act, out, rows, dim, inter, -1, 3, 0.f,
                                     row_weight, true, w2_base, w2_stride, w2s_base, w2s_stride,
                                     w2_base, w2_stride, w2s_base, w2s_stride, ids, slot, stream);
}

extern "C" int dsv41_expert_down_fp4(const float* act, const uint8_t* w2,
                                     const uint8_t* w2_scale, const float* weight, float* out,
                                     int rows, int dim, int inter, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0 || inter <= 0) return (int)cudaErrorInvalidValue;
    // epi_mode 3 = accumulate into `out`. This entry point is used ONLY by the
    // routed experts (the shared expert goes through gemm_fp8_mx), and the MoE
    // needs a sum over the selected experts — so accumulate here rather than
    // making the host issue one add_inplace launch per expert.
    return (int)launch_mxf4(nullptr, nullptr, act, w2, w2_scale, w2, w2_scale, out, rows, dim,
                            inter, -1, 3, 0.f, weight, true, stream);
}

// ============================================================================
// BATCHED expert entry points (DSV41_MOE_BATCH, default OFF). The Rust chain
// picks these to collapse one launch per (layer, top-k slot) into one launch
// per (layer, direction). See expert_gemv_fp4_batched_kernel for the numeric
// contract: per-slot outputs are DISJOINT, and a batched result is bit-identical
// to the sequential per-slot loop.
// ============================================================================

// gate/up, batched over the top-k slots: grid = (row_blocks, slots) and
// blockIdx.y = slot. `out` holds `slots` consecutive [2*inter] blocks, one per
// slot, `out_slot_stride` floats apart; nothing accumulates across slots. The
// activation is the ONE shared quantised row `a`/`a_scale`.
// `ilv` (trailing, new in ABI 2): the w1/w3 pools are INTERLEAVED (DSV41_EXPERT_ILV
// on the Rust side). It selects the kernel's ILV instantiation and REQUIRES the
// fused read: gate and up of a row then come from one region, so the unfused
// (b_split-split) body, which walks the `b` and `b_hi` pools separately, would
// read the wrong bytes. Rejected loudly instead of silently.
extern "C" int dsv41_expert_gate_up_fp4_batched(
    const uint8_t* a, const float* a_scale, float* out, long out_slot_stride, int rows, int dim,
    int inter, float limit, int slots, const uint8_t* w1_base, long w1_stride,
    const uint8_t* w1s_base, long w1s_stride, const uint8_t* w3_base, long w3_stride,
    const uint8_t* w3s_base, long w3s_stride, const int* ids, int ilv, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0 || inter <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
    // rows per CTA == warps per CTA (one warp owns one row). 8 = today's shape;
    // DSV41_GATEUP_ROWS=4/2/1 re-packs the SAME warps into more, smaller CTAs -
    // see the long note above dsv41_gateup_rows() for what that does and does
    // not buy. Everything below (grid.x, blockDim, the kernel's nwarps) derives
    // from this one value, so the window loop stays closed at any setting.
    const int warps = dsv41_gateup_rows();
    // gate_up+swiglu fusion: each warp produces the PAIR (gate_i, up_i) and the
    // epilogue writes the swiglu'd inter-width result directly - one inter-width
    // write instead of the old 2*inter write + a separate swiglu kernel pass.
    // Gated to mode 2 (the vec==2 LUT path, k=dim=5120 exactly divisible by 512)
    // where the fused K-loop is the verbatim copy of the single-row one.
    static const int g_fuse = [] {
        const char* e = getenv("DSV41_GATEUP_FUSE");
        // DEFAULT ON on BOTH sides: chain_dev.rs `gateup_fuse()` is
        // `DSV41_GATEUP_FUSE` .map(|v| v != "0").unwrap_or(true), i.e. unset or
        // any value other than "0" fuses. (This comment said "DEFAULT OFF /
        // .unwrap_or(false)" until 2026-09-11, which did not match the Rust
        // code.) Both sides MUST agree: the kernel's `fuse` decision changes the
        // act slot layout (inter vs 2*inter) and whether the host runs a
        // separate swiglu pass. A mismatch silently corrupts the activations
        // (round-18 bug), and an interleaved pool (DSV41_EXPERT_ILV) makes the
        // mismatch a hard cudaErrorInvalidValue instead - see the `ilv && !fuse`
        // guard below.
        if (e == nullptr) return 1;
        return atoi(e) != 0 ? 1 : 0;
    }();
    const int fuse = (g_fuse && g_expert_fp4_mode == 2 && (dim % 512) == 0) ? 1 : 0;
    // Interleaved weights are only addressable by the FUSED body (see the header
    // comment): refuse the combination rather than read the wrong bytes.
    if (ilv && !fuse) return (int)cudaErrorInvalidValue;
    // K-split (DSV41_GATEUP_KSPLIT, default 1 = original). Only the FUSED body
    // implements the cross-half merge, so the unfused arm always runs ksplit=1
    // (otherwise both halves of a row would compute the full row and race on the
    // same out[row]). blockDim must stay <= 1024 threads: warps*ksplit <= 32.
    int ksplit = fuse ? dsv41_gateup_ksplit() : 1;
    while (ksplit > 1 && warps * ksplit > 32) --ksplit;
    const int n_total = fuse ? inter : 2 * inter;
    // grid.x is the ROW count (n_total/rows), NOT the warp count: with ksplit the
    // CTA carries rows*ksplit warps but still owns `rows` rows. At rows=8 /
    // ksplit=2 this stays (40, 6) = 240 CTAs with 16 warps each (3840 warps in
    // flight vs 1920) - the recommended shape, which does NOT double the per-CTA
    // s_act+LUT prologue. rows=4 / ksplit=2 additionally gives the 480-CTA shape.
    const int ctas_x = (n_total + warps - 1) / warps;
    const size_t smem = (size_t)dim * sizeof(float) + 256 * sizeof(float2) +
                        ((ksplit > 1) ? (size_t)(warps * ksplit) * sizeof(float2) : (size_t)0);
    dim3 grid((unsigned)ctas_x, (unsigned)slots);
    const unsigned block_threads = (unsigned)(warps * ksplit * 32);
    // PDL (see dsv41_experts_pdl_or_plain): the consumer's grid may start during
    // quant_fp4's tail; the kernel's entry cudaGridDependencySynchronize() gates
    // the activation staging. NOTE the full argument list: the cudaLaunchKernelEx
    // path does not apply the kernel's default arguments, so every trailing slot
    // is spelled out.
    cudaError_t le;
    if (ilv)
        le = dsv41_experts_pdl_or_plain(
            expert_gemv_fp4_batched_kernel<true>, grid, dim3(block_threads), smem, stream, nullptr,
            0, a, a_scale, out, out_slot_stride, n_total, dim, inter, 1, limit, nullptr, 0,
            w1_base, w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride,
            ids, g_expert_fp4_mode, fuse, ksplit);
    else
        le = dsv41_experts_pdl_or_plain(
            expert_gemv_fp4_batched_kernel<false>, grid, dim3(block_threads), smem, stream, nullptr,
            0, a, a_scale, out, out_slot_stride, n_total, dim, inter, 1, limit, nullptr, 0,
            w1_base, w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride,
            ids, g_expert_fp4_mode, fuse, ksplit);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// down, batched: writes a [slots][dim] scratch (epi_mode 2 = write, scaled by
// the PER-SLOT routing weight; NOT accumulating). act_base holds `slots`
// consecutive [2*inter] slices `act_stride` floats apart - the swiglu half is
// the first `inter` floats of each slice, exactly the buffer the sequential call
// passed per slot. The fixed-order reduction is a separate kernel
// (dsv41_moe_down_reduce) so the host keeps control of the summation order.
extern "C" int dsv41_expert_down_fp4_batched(
    const float* act_base, long act_stride, float* out, long out_slot_stride, int rows, int dim,
    int inter, const float* row_weight, long rw_stride, int slots, const uint8_t* w2_base,
    long w2_stride, const uint8_t* w2s_base, long w2s_stride, const int* ids, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0 || inter <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
    const int warps = 8;
    dim3 grid((unsigned)((dim + warps - 1) / warps), (unsigned)slots);
    // PDL (see dsv41_experts_pdl_or_plain): same consumer contract as the
    // gate/up call above -- the producer is the gate/up launch that wrote the
    // swiglu'd `act_base`, and the kernel's entry sync gates the staging.
    cudaError_t le = dsv41_experts_pdl_or_plain(
        expert_gemv_fp4_batched_kernel<false>, grid, dim3(warps * 32),
        (size_t)inter * sizeof(float) + 256 * sizeof(float2), stream, act_base, act_stride, nullptr,
        nullptr, out, out_slot_stride, dim, inter, -1, 2, 0.f, row_weight, rw_stride, w2_base,
        w2_stride, w2s_base, w2s_stride, w2_base, w2_stride, w2s_base, w2s_stride, ids,
        g_down_fp4_mode, /*fuse_swiglu=*/0, /*ksplit=*/1);
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// Fixed-order sum of the batched down scratch: out[i] = sum of part[s][i] over
// s = 0,1,... in that order. See moe_down_reduce_kernel.
extern "C" int dsv41_moe_down_reduce(const float* part, float* out, int n, int slots,
                                     cudaStream_t stream) {
    if (n <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
    const unsigned blocks = (unsigned)((n + 255) / 256);
    moe_down_reduce_kernel<<<blocks, 256, 0, stream>>>(part, out, n, slots);
    return (int)cudaGetLastError();
}

// ============================================================================
// down + reduce FUSED entry point (DSV41_DOWN_FUSE on the Rust side, DEFAULT ON: chain_dev.rs `down_fuse()` is
// `.unwrap_or(true)` and the nsys v3 profile sees the fused kernel 40x/step.
// f3b1be1's OFF default was later flipped back; this comment was stale.)
// ============================================================================
// ONE launch covers the whole [slots] down GEMV and the ascending-slot sum,
// writing straight into `out` (OVERWRITE, exactly like moe_down_reduce_kernel:
// the caller needs no zero-fill, and no zero-fill may be applied on top of a
// live residual). `act_base` holds `slots` [2*inter] slices `act_stride` floats
// apart and only the first `inter` floats of each slice (the swiglu half) are
// read. The old `dsv41_expert_down_fp4_batched` + `dsv41_moe_down_reduce` pair is
// untouched and remains the DSV41_DOWN_FUSE=0 fallback.
extern "C" int dsv41_expert_down_reduce_fp4_batched(
    const float* act_base, long act_stride, float* out, int rows, int dim, int inter,
    const float* row_weight, long rw_stride, int slots, const uint8_t* w2_base, long w2_stride,
    const uint8_t* w2s_base, long w2s_stride, const int* ids, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0 || inter <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
    const int warps = 8;
    const size_t lut_bytes = 256 * sizeof(float2);
    // cudaFuncSetAttribute is PER-CONTEXT (per device) - the carve-out bug
    // documented in ferrite_kernels.cu: a one-shot set on whichever device
    // happened to be current left 7 of 8 ranks at the 48KB default. Probe every
    // device's opt-in ceiling the first time it becomes current and raise the
    // STAGED kernel to it. The ceiling is independent of `slots`, so a later call
    // with a different slot count can never find itself under-provisioned.
    static int dev_optin[64];
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = -1;
    if (dev >= 0 && dev < 64 && dev_optin[dev] == 0) {
        int optin = 0;
        if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) ==
                cudaSuccess &&
            optin > 0 &&
            cudaFuncSetAttribute(expert_gemv_fp4_down_reduce_kernel<true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 optin) == cudaSuccess) {
            dev_optin[dev] = optin;
        }
    }
    const size_t staged_bytes = (size_t)slots * (size_t)inter * sizeof(float) + lut_bytes;
    const size_t cap = (dev >= 0 && dev < 64) ? (size_t)dev_optin[dev] : (size_t)0;
    const dim3 grid((unsigned)((dim + warps - 1) / warps));
    // PDL (see dsv41_experts_pdl_or_plain): consumer of the gate/up launch; the
    // kernel's entry sync gates the activation staging (STAGED) and the per-slot
    // reads of the non-staged fallback.
    cudaError_t le;
    if (staged_bytes <= cap) {
        le = dsv41_experts_pdl_or_plain(
            expert_gemv_fp4_down_reduce_kernel<true>, grid, dim3(warps * 32), staged_bytes, stream,
            act_base, act_stride, out, dim, inter, slots, row_weight, rw_stride, w2_base,
            w2_stride, w2s_base, w2s_stride, ids, g_down_fp4_mode);
    } else {
        le = dsv41_experts_pdl_or_plain(
            expert_gemv_fp4_down_reduce_kernel<false>, grid, dim3(warps * 32), lut_bytes, stream,
            act_base, act_stride, out, dim, inter, slots, row_weight, rw_stride, w2_base,
            w2_stride, w2s_base, w2s_stride, ids, g_down_fp4_mode);
    }
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}
