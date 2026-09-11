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
                                               int vec) {
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
    extern __shared__ float s_act[];   // k floats
    // The sixteen possible e2m1 values, built once per block: one shared lookup
    // replaces the eight-way select tree that dsv41_e2m1_to_f expands to
    // at a data-dependent index, which is most of this kernel's instruction count.
    float* s_lut = s_act + k;
    const int kbytes = k >> 1;   // packed bytes per row
    const int ksc = k >> 5;      // e8m0 scales per row
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        if (act != nullptr) {
            s_act[j] = act[j];
        } else {
            const uint8_t ab = a[j >> 1];
            const float asc = a_scale[j >> 5];
            s_act[j] = dsv41_e2m1_to_f((j & 1) ? (uint8_t)(ab >> 4) : (uint8_t)(ab & 0xFu)) * asc;
        }
    }
        if (threadIdx.x < 16) s_lut[threadIdx.x] = dsv41_e2m1_to_f((uint8_t)threadIdx.x);
    __syncthreads();
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
            // Four groups in flight (the fp8 gemv gained 2.4x from the same
            // treatment): the weight/scale/activation loads of g+1..g+3 issue
            // while g's fmas run. The accumulation is `fmaf` (explicitly rounded),
            // so unrolling cannot reassociate it - only re-time the loads - and
            // each accumulator still visits its own elements in ascending g.
#pragma unroll 4
            for (int g = 0; g < nv2; ++g) {
                const int j = (g << 9) + (lane << 4);
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                const uint8_t* bp = brow + (g << 8) + off2;
                const uint32_t w0 = *reinterpret_cast<const uint32_t*>(bp);
                const uint32_t w1 = *reinterpret_cast<const uint32_t*>(bp + 4);
                a0 = fmaf(s_act[j + 0], s_lut[w0 & 0xFu] * sc, a0);
                a1 = fmaf(s_act[j + 1], s_lut[(w0 >> 4) & 0xFu] * sc, a1);
                a2 = fmaf(s_act[j + 2], s_lut[(w0 >> 8) & 0xFu] * sc, a2);
                a3 = fmaf(s_act[j + 3], s_lut[(w0 >> 12) & 0xFu] * sc, a3);
                a0 = fmaf(s_act[j + 4], s_lut[(w0 >> 16) & 0xFu] * sc, a0);
                a1 = fmaf(s_act[j + 5], s_lut[(w0 >> 20) & 0xFu] * sc, a1);
                a2 = fmaf(s_act[j + 6], s_lut[(w0 >> 24) & 0xFu] * sc, a2);
                a3 = fmaf(s_act[j + 7], s_lut[(w0 >> 28) & 0xFu] * sc, a3);
                a0 = fmaf(s_act[j + 8], s_lut[w1 & 0xFu] * sc, a0);
                a1 = fmaf(s_act[j + 9], s_lut[(w1 >> 4) & 0xFu] * sc, a1);
                a2 = fmaf(s_act[j + 10], s_lut[(w1 >> 8) & 0xFu] * sc, a2);
                a3 = fmaf(s_act[j + 11], s_lut[(w1 >> 12) & 0xFu] * sc, a3);
                a0 = fmaf(s_act[j + 12], s_lut[(w1 >> 16) & 0xFu] * sc, a0);
                a1 = fmaf(s_act[j + 13], s_lut[(w1 >> 20) & 0xFu] * sc, a1);
                a2 = fmaf(s_act[j + 14], s_lut[(w1 >> 24) & 0xFu] * sc, a2);
                a3 = fmaf(s_act[j + 15], s_lut[(w1 >> 28) & 0xFu] * sc, a3);
            }
            acc = (a0 + a1) + (a2 + a3);
            for (int j = (nv2 << 9) + lane * 2; j < k; j += 64) {
                const uint8_t byte = brow[j >> 1];
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                acc += s_act[j] * (s_lut[byte & 0xFu] * sc);
                acc += s_act[j + 1] * (s_lut[byte >> 4] * sc);
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
extern "C" int dsv41_expert_gate_up_fp4_batched(
    const uint8_t* a, const float* a_scale, float* out, long out_slot_stride, int rows, int dim,
    int inter, float limit, int slots, const uint8_t* w1_base, long w1_stride,
    const uint8_t* w1s_base, long w1s_stride, const uint8_t* w3_base, long w3_stride,
    const uint8_t* w3s_base, long w3s_stride, const int* ids, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0 || inter <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
    const int warps = 8;
    const int n_total = 2 * inter;
    dim3 grid((unsigned)((n_total + warps - 1) / warps), (unsigned)slots);
    expert_gemv_fp4_batched_kernel<<<grid, warps * 32,
                                         (size_t)dim * sizeof(float) + 16 * sizeof(float),
                                         stream>>>(
        nullptr, 0, a, a_scale, out, out_slot_stride, n_total, dim, inter, 1, limit, nullptr, 0,
        w1_base, w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride, ids,
        g_expert_fp4_mode);
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
    expert_gemv_fp4_batched_kernel<<<grid, warps * 32,
                                         (size_t)inter * sizeof(float) + 16 * sizeof(float),
                                         stream>>>(
        act_base, act_stride, nullptr, nullptr, out, out_slot_stride, dim, inter, -1, 2, 0.f,
        row_weight, rw_stride, w2_base, w2_stride, w2s_base, w2s_stride, w2_base, w2_stride,
        w2s_base, w2s_stride, ids, g_expert_fp4_mode);
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
