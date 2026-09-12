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
// fprintf/stderr for the host-side gate warnings (the PDEPTH pin, below).
#include <cstdio>
// P4 (DSV41_GATEUP_CPASYNC): __pipeline_memcpy_async / __pipeline_commit /
// __pipeline_wait_prior for the weight-first prologue of the batched expert
// gate/up kernel. 16-byte copies lower to cp.async.cg (the same instruction
// dsv41_kernels.cu's dsv41_cp_async16 emits by hand for the gemv P3 window).
#include <cuda_pipeline.h>

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

// 16-byte operand chunk out of GLOBAL memory, alignment-SAFE.
//
// Every tensor-core arm in this file addresses an operand row in uint4s, and a
// `uint4` load REQUIRES a 16-byte-aligned address: anything else is err 716
// ("misaligned address"), which does not even surface at the offending kernel -
// it poisons the context and the NEXT synchronisation reports it, detached from
// the cause (rank-local, e.g. "rank 6: sync: misaligned address"). The bases are
// 16-byte aligned by contract (cudaMalloc, see DevRuntime::alloc, returns
// 256-byte-aligned pointers, and every weight-plane stride is a multiple of 16),
// but a gathered/scattered activation buffer or a future offset view can break
// that silently, and a silent break here is a crash, not a wrong number.
//
// So every global uint4 operand load goes through this helper: the fast path is
// the plain uint4, and a misaligned base degrades to a byte-wise copy. The bytes
// are IDENTICAL either way, so the numeric domain of every arm using it is
// untouched - it only removes the fault.
__device__ __forceinline__ uint4 ld_uint4_a16(const uint8_t* __restrict__ p) {
    if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0)
        return *reinterpret_cast<const uint4*>(p);
    uint4 v;
    __builtin_memcpy(&v, p, 16);
    return v;
}

// 8-byte sibling of ld_uint4_a16: same contract, same reason, HALF the granule.
//
// The gate/up pair body's direct (non-prefetch) weight reads are uint2 loads,
// and a `uint2` REQUIRES an 8-byte-aligned address - 4 bytes off is the same
// err 716 ("misaligned address") as the uint4 case, reported just as detached
// from the cause. The prefetch PROLOGUE has its own `al_ok` gate, but that gate
// guards ONLY the cp.async issue: the direct-read path is what nearly every
// group of every warp walks (only the first group of the row-base warp comes
// from the ring), and it is reached with `pf_ok == false` on every launch whose
// `pf` is 0 and on every warp whose row is past n_total - i.e. the prologue
// guard does not cover it.
//
// The pool base is 256-byte aligned and every per-expert stride here is a
// multiple of 8 today, but the B side is addressed through a TP-sharded
// `DevBuf::view` of the checkpoint (b_base + e*b_stride), which does not
// inherit the 16-byte alignment of the mapped allocation - a shard boundary can
// leave one rank's w3 view 4 bytes off while every other rank is fine.
//
// Same cure as the uint4 helper: plain load on the aligned fast path, byte-wise
// copy otherwise. The 8 bytes are IDENTICAL either way (`v.x` is p[0..3]
// little-endian, `v.y` p[4..7]), so no numeric domain changes - it only removes
// the fault.
__device__ __forceinline__ uint2 ld_uint2_a8(const uint8_t* __restrict__ p) {
    if ((reinterpret_cast<uintptr_t>(p) & 7u) == 0)
        return *reinterpret_cast<const uint2*>(p);
    uint2 v;
    __builtin_memcpy(&v, p, 8);
    return v;
}

// 4-byte and 2-byte siblings for the SPLIT body — the body the grouped /
// GATEUP_FUSE=0 arm actually runs. Same contract, same reason as the two
// helpers above, and they close the SAME err 716 for the granules those two
// cannot express.
//
// WHY THEY ARE NEEDED. The helpers above serve the gate/up PAIR body (the arm
// selected by `pair_body = ((fuse_swiglu != 0) || ILV) && (b_split > 0)`), and
// that is NOT the body this file's grouped / DSV41_EXPERT_GROUPED smoke arm
// runs: `DSV41_GATEUP_FUSE=0` + `DSV41_EXPERT_ILV=0` gives `pair_body == false`,
// so the SPLIT body below walks `b_use` and `bhi_use` separately and reads its
// weight row DIRECTLY:
//     brow = (hi ? bhi_use : b_use) + (size_t)r * kbytes;
//     *(const uint32_t*)(brow + (g << 8) + (lane << 3))   // vec==2
//     *(const uint16_t*)(brow + (g << 6) + (lane << 1))   // vec==3
//     *(const uint32_t*)(brow + (g << 7) + (lane << 2))   // vec==1 fast path
// None of those reads is touched by ld_uint4_a16 / ld_uint2_a8, so "the guard
// is in place" never said anything about this body — which is exactly the
// asymmetry the tcgen05 smoke failures ran into.
//
// The alignment premise is `b_use`/`bhi_use` = `base + e * stride`, i.e. the
// SAME TP-sharded `DevBuf::view` of the checkpoint that ld_uint2_a8 documents
// (a shard boundary can leave one rank's W1/W3 view a few bytes off while every
// other rank is fine), plus `r * kbytes` with `kbytes = k >> 1`. A `uint32_t`
// load still REQUIRES a 4-byte-aligned address (2 bytes for uint16), and a
// 1..3-byte slip raises the identical err 716 ("misaligned address") that the
// 16-byte case does, reported — as the note above explains — on the NEXT
// synchronisation rather than at the offending kernel. That is the signature in
// `rank 7: config error: sync: misaligned address`.
//
// Same cure as the two helpers above: plain load on the aligned fast path,
// byte-wise `__builtin_memcpy` otherwise. The bytes are IDENTICAL either way
// (`v` is p[0..3] little-endian, `v` is p[0..1] for the uint16), so no numeric
// domain changes — it only removes the fault.
__device__ __forceinline__ uint32_t ld_uint32_a4(const uint8_t* __restrict__ p) {
    if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0)
        return *reinterpret_cast<const uint32_t*>(p);
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

// 2-byte sibling, same contract (the split body's vec==3 `uint16` read).
__device__ __forceinline__ uint16_t ld_uint16_a2(const uint8_t* __restrict__ p) {
    if ((reinterpret_cast<uintptr_t>(p) & 1u) == 0)
        return *reinterpret_cast<const uint16_t*>(p);
    uint16_t v;
    __builtin_memcpy(&v, p, 2);
    return v;
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
                    val = ld_uint4_a16(a + (size_t)row * (k >> 1) + (k0 >> 1) +
                                       atom * kAtomBytes + kb * 16);
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
                    val = ld_uint4_a16(src_base + (size_t)row * (k >> 1) + (k0 >> 1) +
                                       atom * kAtomBytes + kb * 16);
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

// e4m3 byte -> f32: sign(1) / exponent(4, bias 7) / mantissa(3); e == 0 is the
// subnormal arm m * 2^-9. EXACT (every e4m3 value is representable in f32),
// which is what lets the e4m3 activation path feed the SAME float dot the e2m1
// one does - the official `fp4_gemm` is "FP8 act x FP4 weight" with the FP4
// weight cast up to FP8, and an fp4/fp8 -> f32 decode composes with that cast.
// The expert activation quantiser (`quant.rs::e4m3_encode` via `dsv41_quant_fp8`)
// saturates into +-448 and never emits 0x7F/0xFF, so no Inf/NaN arm is needed.
__device__ __forceinline__ float dsv41_e4m3_to_f(uint8_t b) {
    const uint32_t e = (b >> 3) & 0x0Fu;
    const uint32_t m = b & 0x07u;
    // e == 0: m * 2^-9        e > 0: (8 + m) * 2^(e - 10)
    const float v = (e == 0) ? ((float)m * (1.0f / 512.0f))
                             : (float)(8u + m) * __uint_as_float((e + 117u) << 23);
    return (b & 0x80u) ? -v : v;
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
                                       const int* __restrict__ ids, int slot, int act_e4m3) {
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
    const int ksc = k >> 5;      // scales per row (f32 for the activation)
    // DSV41_EXPERT_ACT_E4M3 (direct e4m3, official semantics): `a` holds ONE
    // e4m3 byte per value (no packing), so the row is `k` bytes; `a_scale` is
    // the quantiser's own f32 per 32 (`dsv41_quant_fp8(block=32)`). Same float
    // staging as the e2m1 arm - only the decode and the row pitch differ - so
    // the dot, the shuffle tree and the epilogue are untouched.
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        if (a_f32 != nullptr) {
            s_act[j] = a_f32[j];
        } else if (act_e4m3) {
            s_act[j] = dsv41_e4m3_to_f(a[j]) * a_scale[j >> 5];
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
//
// `DSV41_DOWN_4VAL` (the spec-side name, same polarity: an explicit "0" turns
// the 4-value drain off, anything else keeps it) is an ALIAS of the same switch,
// so an A/B can be driven by either name - the canonical, historical one stays
// DSV41_DOWN_VEC4. Same precedent as the DSV41_VERIFY_ROPE_MROWS alias for
// DSV41_ROW_FOLD_ROPE. An explicit "0" from EITHER name wins (conservative: the
// rollback must never lose to the historical default).
static const int g_down_fp4_mode = [] {
    const char* names[2] = {"DSV41_DOWN_VEC4", "DSV41_DOWN_4VAL"};
    for (int i = 0; i < 2; ++i) {
        const char* e = getenv(names[i]);
        if (e != nullptr && atoi(e) == 0) return g_expert_fp4_mode;
    }
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
        // Default OFF, matching dsv41_pdl_enabled() in dsv41_kernels.cu: PDL was
        // introduced default-ON but never production-verified, and the two TUs
        // must not disagree on the unset case (a split default is a silent
        // behaviour change depending on which launcher a call site happens to
        // bind to). The other TU is the source of truth; keep them identical:
        // unset = OFF, only "1" enables.
        cached = (e != nullptr && e[0] == '1') ? 1 : 0;   // explicit "1" enables
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

// K-SPLIT for the FUSED gate/up body (DSV41_GATEUP_KSPLIT, default 2 = ON).
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
// not bit-identical; the per-layer drift is ~1e-7. That text A/B has since been
// run (6.90 -> 6.57ms, -0.33ms), so the gate is flipped ON and the DEFAULT is 2;
// `=1` is still accepted for the original OFF/parity arm.
//
// The per-half group range is [half*nv2f/ksplit, (half+1)*nv2f/ksplit) with
// nv2f = k>>9 (10 at k=5120) - a cut on 512-value group boundaries, so no
// 32-value scale block is ever straddled. nv2f=10 is NOT divisible by 4 (only
// 3/3/2/2), so the meaningful values are 2 (or 5); 4 is allowed but unbalanced.
constexpr int kGateUpKsplitMax = 8;

static int dsv41_gateup_ksplit(void) {
    static int cached = -1;
    if (cached < 0) {
        int v = 2;  // K-split default ON (A/B verified: 6.90->6.57ms, -0.33ms)
        if (const char* e = getenv("DSV41_GATEUP_KSPLIT")) {
            v = atoi(e);
            if (v < 1) v = 1;
            if (v > kGateUpKsplitMax) v = kGateUpKsplitMax;
        }
        cached = v;
    }
    return cached;
}

// P4 (DSV41_GATEUP_CPASYNC, default ON, `=0` restores the old issue order):
// cp.async WEIGHT-FIRST prologue for the FUSED batched gate/up body. Same idea
// as the gemv P3 window in dsv41_kernels.cu (see g_gemv_cpasync there): the
// row's weight bytes are model constants, so their transfer can START under the
// block prologue (activation staging + LUT) instead of after its barrier, where
// it used to sit in front of the first group's dot on the critical path.
//
// WHAT IS PREFETCHED. The gateup row is NOT staged wholesale: one warp owns ONE
// inter row and walks it in `nv2f = k>>9` groups (512 values = 512 B of
// interleaved w1/w3 bytes). The prefetch covers exactly the FIRST group the warp
// consumes (`g_begin = half*nv2f/ksplit`), i.e. 512 B per warp: for ILV one
// contiguous 512-byte chunk, for the plain layout the 256-byte gate head and the
// 256-byte up head. That is the only part of the transfer the prologue barrier
// can cover; staging the WHOLE row (5 KB/warp -> 80 KB/CTA at rows=8/ksplit=2)
// would trade the overlap for residency, the trap the gemm-prologue-overlap
// analysis flagged for the double-buffered gemv variant.
//
// SMEM COST: 512 B x nwarps (8 KB at rows=8/ksplit=2) added to the existing
// dim*4 + 256*float2 [+ nwarps*float2 when ksplit>1] pool. The CTA is
// thread-limited (512 threads, 4 CTAs/SM) at the default shape, so 8 KB costs
// nothing there; the buffer sits between s_lut2 and s_ks, so the launcher's
// formula and the kernel's pointer must move together (the s_ks coupling hazard,
// documentation §0.55).
//
// BIT-EXACTNESS: same bytes, same slot, same consume order; only the moment of
// the copy moves. The smem slot is read by the same lane offsets the gmem load
// used, and cp.async completion is published by the existing pre-loop
// __syncthreads() (the plain-layout chunks are wider than one lane's read, so
// the barrier - not just per-lane ordering - is required).
//
// SCOPE: only the fused gate/up branch implements the substitution. The
// unfused vec/vec2 bodies and the downdirection (fuse_swiglu==0) keep their
// direct gmem reads and are launched with pf=0 (no buffer reserved).
static int dsv41_gateup_cpasync(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("DSV41_GATEUP_CPASYNC");
        cached = (e != nullptr && e[0] == '0') ? 0 : 1;   // default ON
    }
    return cached;
}

// Number of bytes ONE k-group of gate+up occupies per warp (fixed 512 = one
// 512-value group). Shared by the kernel's pointer arithmetic and the
// launcher's dynamic-smem formula - keep them on this constant.
constexpr int kGateUpPfBytes = 512;

// P4.2 (DSV41_GATEUP_PIPELINE) historical maximum pipeline depth. 5 was the
// largest depth that is ever USEFUL: at the production shape a warp owns half of
// one row (ksplit=2 of nv2f = k>>9 = 10 groups), i.e. 5 groups, so depth 5 = the
// whole slice in flight. Depth 2/5 both measured +0.04ms in serve, so the kernel
// is instantiated at depth 1 ONLY (see dsv41_gateup_pipeline): this constant now
// survives as the documented ceiling for a future re-enable, and as the number
// quoted in the "env value ignored" warning.
constexpr int kGateUpPfDepthMax = 5;

// P4.2 (DSV41_GATEUP_PIPELINE): DEPTH of the fused gate/up weight pipeline, in
// k-groups in flight per warp. CURRENTLY PINNED TO 1 (the P4 behaviour).
//
// WHY: the P4 prologue (above) only covers group 0. Every later group's LDG fell
// between the barrier and its own dot, so a warp's weight stream was a chain of
// ~600-cycle HBM loads with only ~50 cycles of FMA between them - the measured
// symptom is an expert gateup at 22.2us/call, 443 GB/s and IPC 0.8/4, i.e. the
// issue slots are 80 percent stalled on operand supply, not on arithmetic. A
// register-resident unroll (the `#pragma unroll 4` below) cannot fix it: the LDG
// destination registers sit on the consumer's scoreboard, so the warp stalls
// anyway, and 64 regs/thread (the __launch_bounds__(1024) cap) buy only a few
// groups of slack. cp.async takes the load OFF the register scoreboard: the warp
// issues the copies for group g+D, and by the time it needs group g the data is
// already in smem - the wait is on a cp.async group counter, not on a register.
//
// HOW: per-warp ring of PDEPTH kGateUpPfBytes slots. The prologue issues
// PDEPTH groups (group g_begin .. g_begin+PDEPTH-1); iteration i of the group
// loop waits with wait_prior(PDEPTH-1) - which, because exactly one commit
// group is added per iteration (empty commits at the tail), always covers
// group i - reads slot i%PDEPTH, then immediately issues group i+PDEPTH into
// that same (now consumed) slot. PDEPTH == 1 is the P4 behaviour, kept as the
// A/B arm (`DSV41_GATEUP_PIPELINE=1`).
//
// The value is a COMPILE-TIME kernel template parameter because
// cp.async.wait_group takes an immediate operand - the launcher picks the
// instantiation matching the depth (see the dispatch in
// dsv41_expert_gate_up_fp4_batched).
//
// PINNED TO 1 (2026-09-12, gate-hygiene): serve A/B showed depth 2 and 5 are
// each +0.04ms (occupancy loss > latency hiding; the isolated IPC gain does not
// translate), so no depth above 1 is worth an instantiation. The launcher now
// references ONLY <ILV, 1>, which makes every depth >1 body dead code - keeping
// the other 8 instantiations in the .so is pure artifact bloat (the same
// "code presence" cost sparse-merge paid for: 3570524, +0.34ms). A depth
// request from the environment is therefore accepted but CLAMPED to 1, and the
// caller is told once instead of being silently ignored.
// kGateUpPfDepthMax (5) is kept as the historical ceiling: the kernel body still
// implements the ring for any PDEPTH, so re-enabling a depth means restoring the
// two other instantiations in that dispatch, nothing else.
static int dsv41_gateup_pipeline(void) {
    static int cached = -1;
    if (cached < 0) {
        int v = 1;
        if (const char* e = getenv("DSV41_GATEUP_PIPELINE")) v = atoi(e);
        if (v > 1) {
            fprintf(stderr,
                    "dsv41: DSV41_GATEUP_PIPELINE=%d ignored - only depth 1 is instantiated "
                    "(max useful was %d; 2/5 both measured +0.04ms)\n",
                    v, kGateUpPfDepthMax);
            v = 1;
        }
        if (v < 1) v = 1;
        cached = v;   // always 1
    }
    return cached;
}

// cp.async global -> shared copies and the commit/wait triple.
//
// Written as raw PTX (instead of __pipeline_memcpy_async / __pipeline_commit /
// __pipeline_wait_prior) for ONE reason: the CUDA header's forms carry no
// "memory" clobber, so the compiler is free to move the surrounding shared
// loads across the copy issue. In the ring the copy for group i+PDEPTH lands in
// the very slot group i is being read from, so the read MUST be issued before
// the copy - the clobber is what guarantees it (the timing argument alone would
// hold today, but a compiler that hoists the copy above the LDS would make the
// ring silently read the wrong group, with no crash and no failed check).
//
// 16 bytes lowers to cp.async.cg (L2 only), 8 bytes can only be cp.async.ca -
// cp.async.cg exists for 16 bytes alone. The pre-sm_80 arms keep the TU
// compilable for any arch with identical data movement (this TU is only built
// for sm_100a/sm_103a, so they are never taken in production).
__device__ __forceinline__ void dsv41_gateup_pf16(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(smem)),
                 "l"(gmem)
                 : "memory");
#else
    *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
#endif
}
// 8-byte variant, used by the PLAIN (non-interleaved) layout: there a lane
// consumes 8 gate bytes AND 8 up bytes per group, and 8-byte copies keep the
// copy assignment LANE-LOCAL (see dsv41_gateup_pf_group), so no __syncwarp /
// __syncthreads is needed to publish a staged group to its consumer.
__device__ __forceinline__ void dsv41_gateup_pf8(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(smem)),
                 "l"(gmem)
                 : "memory");
#else
    *reinterpret_cast<uint2*>(smem) = *reinterpret_cast<const uint2*>(gmem);
#endif
}
__device__ __forceinline__ void dsv41_gateup_pf_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
// Waits for ALL BUT THE NEWEST N commit groups of THIS thread. N is an
// immediate in PTX, hence the template. Called with no outstanding group when
// the prefetch did not arm, which is a documented no-op.
template <int N>
__device__ __forceinline__ void dsv41_gateup_pf_wait_prior() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}

// Stage ONE k-group (512 B of gate+up) of row (`g_row`, `u_row`) into the
// per-warp ring slot `dst`, LANE-LOCAL: every lane copies exactly the bytes it
// will consume, so a staged group is visible to its consumer as soon as that
// lane's own wait_group retires - no __syncwarp / __syncthreads is required
// (this is what lets the pipeline run without a barrier per group).
//   ILV   : lane L consumes the 16 B at g_row + gi*512 + L*16 (gate pair first,
//           up pair second - the interleaver stores dst[2i]=gate[i],
//           dst[2i+1]=up[i] at 8-byte granularity) -> one 16-byte copy.
//   plain : lane L consumes 8 B at g_row + gi*256 + L*8 and 8 B at
//           u_row + gi*256 + L*8, staged at dst + L*8 and dst + 256 + L*8 -
//           exactly the offsets the consumer reads (the P4 prologue produced
//           the same buffer contents with a cross-lane 16-byte mapping plus the
//           pre-loop barrier; the 8-byte form removes that dependency).
// Byte offsets: the row is walked in 512-VALUE groups, so a group's gate (plain)
// chunk is 256 B and its interleaved chunk is 512 B.
template <bool ILV>
__device__ __forceinline__ void dsv41_gateup_pf_group(uint8_t* dst, const uint8_t* g_row,
                                                      const uint8_t* u_row, int gi, int lane) {
    if (ILV) {
        dsv41_gateup_pf16(dst + (lane << 4), g_row + (size_t)gi * 512 + (lane << 4));
    } else {
        dsv41_gateup_pf8(dst + (lane << 3), g_row + (size_t)gi * 256 + (lane << 3));
        dsv41_gateup_pf8(dst + 256 + (lane << 3), u_row + (size_t)gi * 256 + (lane << 3));
    }
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
// `ilv` argument; nothing here guesses.
//
// ILV and THE EPILOGUE ARE INDEPENDENT (decoupled 2026-09-12). The interleaved
// pool is addressable by exactly one body - the GATE/UP PAIR body, where one
// warp owns one row of the inter-width output space and produces BOTH halves of
// it (see the `pair_body` predicate) - and THAT body has two epilogues,
// selected by the runtime `fuse_swiglu`:
//   * fuse_swiglu != 0: the swiglu'd [inter] slice (the classic fused write);
//   * fuse_swiglu == 0: the RAW gate|up pair, out[row] = gate and
//     out[b_split + row] = up, i.e. the unfused [2*inter] layout the e2m1x2
//     (DSV41_EXPERT_ACT_E4M3) two-pass arm accumulates in before ONE swiglu.
// The read path and the write path share no smem or register state (the read
// picks bytes, the epilogue picks a destination), so any combination is legal.
// What the interleaved pool DOES require is the pair body's K contract, dim %
// 512 == 0 - the launcher refuses the rest (see its own guard).
//
// __launch_bounds__(1024): pins the register ceiling that the launch geometry
// depends on. The block size is NOT fixed here - the launcher derives it as
// warps*ksplit*32 (`dsv41_expert_gateup_fp4_batched`), where DSV41_GATEUP_ROWS
// (1..32) x DSV41_GATEUP_KSPLIT (1..8, clamped to warps*ksplit <= 32) can take
// it all the way to 1024. 1024 is therefore the widest block this kernel is
// EVER launched with, and declaring it makes ptxas cap registers at
// 65536/1024 = 64/thread: the production 256-thread shape then keeps its 4
// blocks/SM (the same ceiling the microbench's fastest 56-reg arm needs on the
// mirrored down kernel) and the widest shape stays launchable.
// A literal __launch_bounds__(256, N) would be WRONG - it is a hard launch
// check, so every DSV41_GATEUP_KSPLIT=2 (512 threads) / DSV41_GATEUP_ROWS=16+
// launch would fail with "too many resources requested". Measured usage was
// 48 regs / 0 spill, so the 64 cap is headroom, not a spill risk.
// A second template parameter, PDEPTH (P4.2): the fused gate/up weight
// pipeline depth in k-groups. It must be a compile-time constant because the
// per-group wait is `cp.async.wait_group N` with N an immediate - that is the
// ONLY reason it is a template parameter, and the reason a depth costs a whole
// kernel instantiation. Only PDEPTH == 1 is INSTANTIATED (the P4 behaviour:
// prologue stages group 0 only, every later group read straight from global);
// the body still implements the ring for any PDEPTH, so re-enabling a depth is a
// matter of restoring the launcher's cascade + the smem opt-in, not of rewriting
// this kernel (see dsv41_gateup_pipeline for why depth >1 is not worth it).
template <bool ILV, int PDEPTH>
__launch_bounds__(1024)
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
                                               int vec, int fuse_swiglu, int ksplit, int pf,
                                               int act_e4m3) {
    const int slot = (int)blockIdx.y;
    // ---- ACTIVATION ROW (grid.z): the MULTI-ROW dimension ------------------
    // `rows` is now the launcher's THIRD grid dimension (it used to be
    // validation-only, so every call computed ONE activation row). rows == 1
    // leaves blockIdx.z == 0, every offset below is 0 and this kernel is the
    // pre-existing single-row one BIT FOR BIT.
    //
    // LAYOUT CONTRACT (rows > 1) - the caller's buffers are [rows][slot][...],
    // each row laid out EXACTLY as the rows == 1 call laid out its single row:
    //   ids         : [rows][slots]                  row pitch = slots
    //   row_weight  : [rows][slots][rw_stride]       row pitch = slots*rw_stride
    //   out         : [rows][slots][out_slot_stride] row pitch = slots*out_slot_stride
    //   act (f32)   : [rows][slots][act_stride]      row pitch = slots*act_stride
    //   a / a_scale : [rows][k/2] bytes / [rows][k/32] floats - the quantiser's
    //                 own row-major packed layout (`dsv41_quant_fp4`), so the fp4
    //                 activation needs NO extra stride argument.
    // Every pitch is derived from an argument the caller ALREADY passes plus
    // gridDim.y (the slot count), which is what keeps this ABI-2 entry point
    // unchanged (the Rust side needs no new parameter). A caller that wants a
    // different row pitch needs a NEW entry point - do not reinterpret these.
    //
    // ROW INDEPENDENCE (why the acceptance test can compare row-by-row): rows
    // share no output, no accumulator and no shared-memory staging - the only
    // thing that changes with `arow` is a base pointer. Row r's K loop is the
    // rows==1 K loop (same group order, same fma chains, same shuffle tree, same
    // epilogue), so row r of a rows=m call is BIT-IDENTICAL to a rows==1 call on
    // the same row. See kernels/cuda/tests_dsv41_experts_mrows.cu.
    const int arow = (int)blockIdx.z;
    const size_t slot0 = (size_t)arow * (size_t)gridDim.y;   // this row's slot-0 index
    const float* act = (a_f32 != nullptr)
                           ? (a_f32 + (slot0 + (size_t)slot) * (size_t)act_stride)
                           : nullptr;
    const float* rw = (row_weight != nullptr)
                          ? (row_weight + (slot0 + (size_t)slot) * (size_t)rw_stride)
                          : nullptr;
    out += (slot0 + (size_t)slot) * (size_t)out_slot_stride;
    const size_t e = (size_t)ids[slot0 + (size_t)slot];
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
    // Warp/lane ids are needed HERE (before the prologue) because the P4
    // weight-first staging below is per-warp; they were previously derived after
    // the barrier. Pure arithmetic on threadIdx - no dependency on any staging.
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = (blockDim.x + 31) >> 5;
    // P4 (DSV41_GATEUP_CPASYNC) weight-first staging: kGateUpPfBytes per warp =
    // the warp's FIRST k-group of gate+up weight bytes (see
    // dsv41_gateup_cpasync). Placed right after the 256-entry LUT: the dynamic
    // smem base is 16-byte aligned and k*4 + 256*8 is a multiple of 16, so every
    // per-warp slot starts 16-byte aligned for cp.async.cg. `pf` is non-zero only
    // for the FUSED gate/up launch, and that launcher reserves exactly
    // `nwarps*kGateUpPfBytes` when it sets it - moving this pointer without the
    // matching change in dsv41_expert_gate_up_fp4_batched's smem formula would
    // shift s_ks (and read/write past the allocation).
    // P4.2 (DSV41_GATEUP_PIPELINE): the per-warp staging area is a RING of
    // PDEPTH slots of kGateUpPfBytes (PDEPTH == 1 reproduces the P4 layout byte
    // for byte, because warp*(512*1) IS warp*512). Slot (g2-g_begin)%PDEPTH
    // holds group g2 while it is in flight. `pf` is non-zero only for the FUSED
    // gate/up launch, and that launcher reserves exactly
    // `nwarps*kGateUpPfBytes*PDEPTH` when it sets it (PDEPTH is pinned to 1 today
    // - see dsv41_gateup_pipeline - so what is allocated is the PDEPTH == 1
    // size; the expression still tracks PDEPTH in case a depth is re-enabled) -
    // moving this pointer
    // without the matching change in dsv41_expert_gate_up_fp4_batched's smem
    // formula would shift s_ks (and read/write past the allocation).
    uint8_t* s_pf = reinterpret_cast<uint8_t*>(s_lut2 + 256);
    // K-split partials (DSV41_GATEUP_KSPLIT>1, fused gate/up only): ONE (gate,up)
    // float2 per warp, indexed by the CTA-local warp id. The ksplit halves of a
    // row are consecutive warps (warp = row_local*ksplit + half), so half 0 reads
    // s_ks[warp+1 .. warp+ksplit-1] to fold its partners. Laid out after the LUT
    // (and, when pf != 0, after the per-warp prefetch buffer) and allocated by
    // the launcher ONLY when ksplit>1, so the ksplit==1 launch keeps the original
    // dynamic-smem size (and occupancy).
    float2* s_ks = reinterpret_cast<float2*>(s_pf + (size_t)nwarps * kGateUpPfBytes * PDEPTH);
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
    // ---- P4 (DSV41_GATEUP_CPASYNC): cp.async WEIGHT-FIRST prologue ---------
    // See dsv41_gateup_cpasync above for the reasoning. Short version: this
    // warp's row is `row_base` and the FUSED body walks it in `k>>9` groups of
    // 512 B, starting at group `g_begin`. The first group's weight bytes depend
    // on NOTHING the prologue computes, so they are issued into shared memory
    // HERE - before the activation staging - and the staging + the pre-loop
    // barrier become their cover. Without this the first group's LDG sat after
    // the barrier, in front of its own dot, with nothing to hide it.
    //
    // It sits AFTER cudaGridDependencySynchronize() on purpose: `w` can be
    // written by the previous node on the stream, and under PDL that producer
    // may still be running - an early issue would be a race (same rule as the
    // gemv P3 window).
    //
    // The row mapping is computed here (it is pure `blockIdx`/`threadIdx`
    // arithmetic) because the prefetch needs it; the row loop below reuses these
    // exact values, so there is ONE mapping definition.
    const int rows_per_cta = (ksplit > 0) ? (nwarps / ksplit) : nwarps;
    const int row_local = (ksplit > 0) ? (warp / ksplit) : warp;
    const int half = (ksplit > 0) ? (warp % ksplit) : 0;
    const int row_base = blockIdx.x * rows_per_cta + row_local;
    const int nv2f = k >> 9;                    // 512-value groups per row
    const int g_begin = (half * nv2f) / ksplit; // this warp's first group
    // Hoisted out of the row loop together with g_begin: the prologue has to
    // know how many groups this warp's slice has before it issues PDEPTH of
    // them (P4 only ever needed to know where the slice STARTS). Same
    // expression, one definition.
    const int g_end = ((half + 1) * nv2f) / ksplit;
    // The GATE/UP PAIR BODY: one warp owns ONE row of the inter-width output
    // space and produces BOTH halves of it - the gate row and the up row - so
    // the two halves of one output row are one work unit. TWO INDEPENDENT
    // properties select it:
    //   * INTERLEAVED weights (ILV, compile-time): the gate chunk and the up
    //     chunk of a row sit 8 bytes apart, so this body fetches both with ONE
    //     LDG.128. It is the ONLY body that can address that pool - the
    //     plain-layout split arm below walks the `b` base and the `b_hi` base
    //     separately, which an interleaved pool does not have.
    //   * the fused EPILOGUE (fuse_swiglu, runtime): swiglu'd [inter] write
    //     instead of the raw gate|up pair (out[row] / out[b_split + row]).
    // `b_split > 0` keeps it out of the down direction (b_split == -1) and out
    // of that split arm. The launcher derives the SAME predicate for its
    // n_total / ksplit / pf decisions - the two must agree (see
    // dsv41_expert_gate_up_fp4_batched).
    const bool pair_body = ((fuse_swiglu != 0) || ILV) && (b_split > 0);
    // Armed only for the pair body (the one body that implements the
    // substitution) and only for a warp that owns a real row. `pf` is set by the
    // launcher together with the matching smem reservation.
    bool pf_ok = (pf != 0) && pair_body && ((k & 511) == 0) && (row_base < n_total);
    // P4.2: this warp's ring of PDEPTH slots (512 B each). PDEPTH == 1 gives
    // back the P4 single slot: warp*(512*1) == warp*512.
    uint8_t* pf_ring = s_pf + (size_t)warp * (kGateUpPfBytes * PDEPTH);
    if (pf_ok) {
        // Family/pitch select copied from the row loop verbatim: drifting from
        // it here would silently stage the WRONG row (no crash, wrong dot).
        const uint8_t* g_row = b_use + (size_t)row_base * (ILV ? (kbytes << 1) : kbytes);
        const uint8_t* u_row = ILV ? g_row : (bhi_use + (size_t)row_base * kbytes);
        // Alignment is a cp.async REQUIREMENT on both sides (16 B for the .cg
        // copy, 8 B for the .ca one). The pool base is 256-byte aligned and
        // every per-expert stride / row pitch here is a multiple of 16
        // (dim % 512 == 0), so this always holds today; the guard keeps a future
        // odd stride from faulting (err 716) instead of falling back silently -
        // same lesson as the gemv scale row. Checking the ROW base is enough:
        // every group / lane offset added below is a multiple of 16 (ILV) or of
        // 8 (plain), and 16-aligned implies 8-aligned.
        const bool al_ok = (((uintptr_t)g_row & 15u) == 0) &&
                           (ILV || (((uintptr_t)u_row & 15u) == 0));
        if (al_ok) {
            // Issue the first PDEPTH groups (or as many as the slice holds).
            // ONE commit per iteration, including the out-of-range tail: the
            // loop's wait_prior(PDEPTH-1) is only correct if the number of
            // commit groups issued before consuming group i is exactly
            // PDEPTH + i, which the empty commits at the tail preserve.
#pragma unroll
            for (int r = 0; r < PDEPTH; ++r) {
                const int gi = g_begin + r;
                if (gi < g_end)
                    dsv41_gateup_pf_group<ILV>(pf_ring + (size_t)r * kGateUpPfBytes, g_row,
                                               u_row, gi, lane);
                dsv41_gateup_pf_commit();
            }
        } else {
            pf_ok = false;
        }
    }
    // Vectorized staging (audit #2): one uint4 = 16 bytes = 32 fp4 values =
    // exactly one scale block, so each thread-iteration is 1 LDG.128 + 1 scale
    // load instead of 20 serial LDG.8s. BIT-EXACT: the nibble order (low ->
    // even j, high -> odd j), the e2m1 decode and the per-32 scale multiply are
    // identical to the byte loop below; only the number of load instructions
    // changes. The f32 path gets the same treatment via float4.
    //
    // MULTI-ROW: the fp4 activation's row pitch is the quantiser's own row-major
    // packing (kbytes bytes / ksc scales per row) - see the layout contract at
    // the top of this kernel. `arow == 0` folds every offset below to exactly the
    // pre-multi-row address arithmetic.
    //
    // DSV41_EXPERT_ACT_E4M3 (direct e4m3, official semantics): `a` holds ONE
    // e4m3 byte per value instead of two packed nibbles, so the row PITCH
    // doubles (`abytes = k`) while the weights stay fp4-packed (`kbytes`). This
    // is the official `fp4_gemm` activation - `act_quant(e4m3, block=32)` -
    // consumed in one pass instead of the retired e2m1x2 two-pass simulation.
    const int abytes = act_e4m3 ? k : kbytes;
    const uint8_t* a_row = (a != nullptr) ? (a + (size_t)arow * (size_t)abytes) : a;
    const float* asc_row = (a != nullptr) ? (a_scale + (size_t)arow * (size_t)ksc) : a_scale;
    if (act_e4m3 && a_row != nullptr && act == nullptr) {
        // 16 bytes = 16 values, all inside ONE 32-value scale block (a 16-value
        // group starts at a multiple of 16 and `g >> 1` is its block), so one
        // LDG.128 + one LDS.32 scale covers the group.
        if ((k & 15) == 0 && (((uintptr_t)a_row & 15) == 0)) {
            const int n16 = k >> 4;
            for (int g = threadIdx.x; g < n16; g += blockDim.x) {
                const uint4 packed =
                    *reinterpret_cast<const uint4*>(a_row + (size_t)g * 16);
                const float asc = asc_row[g >> 1];
                const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed);
                float* dst = s_act + (size_t)g * 16;
#pragma unroll
                for (int q = 0; q < 16; ++q) dst[q] = dsv41_e4m3_to_f(pb[q]) * asc;
            }
        } else {
            for (int j = threadIdx.x; j < k; j += blockDim.x)
                s_act[j] = dsv41_e4m3_to_f(a_row[j]) * asc_row[j >> 5];
        }
    } else if ((k & 31) == 0 && a_row != nullptr && act == nullptr &&
        (((uintptr_t)a_row & 15) == 0)) {
        const int nb32 = k >> 5;
        for (int b32 = threadIdx.x; b32 < nb32; b32 += blockDim.x) {
            const uint4 packed =
                *reinterpret_cast<const uint4*>(a_row + (size_t)b32 * 16);
            const float asc = asc_row[b32];
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
            const uint8_t ab = a_row[j >> 1];
            const float asc = asc_row[j >> 5];
            s_act[j] = dsv41_e2m1_to_f((j & 1) ? (uint8_t)(ab >> 4) : (uint8_t)(ab & 0xFu)) * asc;
        }
    }
    }
    // P4: drain the weight-first cp.async group here, BEFORE the barrier, so the
    // barrier publishes the staged group block-wide to every reader lane (the
    // plain-layout chunks are 16 B wide while a lane consumes 8 B, so the
    // consumer lane is not always the copying lane). No-op when nothing was
    // committed. This is the "commit + wait before the dot" point of the scheme:
    // the actual dot runs right after the row loop's first iteration starts.
    //
    // P4.2 (PDEPTH > 1): NOT drained here. With a ring nothing is published to
    // other lanes - every staged byte is copied by the very lane that reads it
    // (dsv41_gateup_pf_group) - so the only owner of the wait is the consumer
    // iteration itself (wait_prior(PDEPTH-1) in the group loop). Draining here
    // would serialize the whole depth away. The __syncthreads() below stays:
    // it is what publishes the ACTIVATION staging (s_act), which is still
    // block-cooperative.
    if constexpr (PDEPTH == 1) dsv41_gateup_pf_wait_prior<0>();
    __syncthreads();

    // K-SPLIT row mapping (DSV41_GATEUP_KSPLIT, default 1 = original):
    // rows_per_cta / row_local / half / row_base are computed in the P4 prologue
    // above (the prefetch needs them). With ksplit warps per row, nwarps =
    // rows*ksplit, so the CTA's ROW count is nwarps/ksplit and warp `w` owns
    // row_local `w/ksplit`, half `w%ksplit`; ksplit==1 => rows_per_cta == nwarps,
    // row_local == warp: the exact original mapping. grid.x is still
    // ceil(n_total/rows) (the launcher sizes it by the ROW count), i.e. ONE pass;
    // ksplit>1 therefore stops the window loop after its single iteration and
    // lets the warps whose row is past n_total reach the fused branch's
    // cross-half __syncthreads() as well (guard by `active`, not by the loop
    // bound, or that barrier would deadlock the CTA).
    const int row_stop = (ksplit > 1) ? (row_base + 1) : n_total;   // ksplit>1: one trip
    for (int row = row_base; row < row_stop; row += gridDim.x * rows_per_cta) {
        // THE GATE/UP PAIR BODY (pair_body, gate/up direction). Here n_total ==
        // inter and this warp owns ONE inter row `row`: it walks BOTH halves of
        // that row - the gate row `row` of the `b` pair and the up row `row` of
        // the `b_hi` pair (or, ILV, both halves of the interleaved row) - and
        // writes ONE of two results:
        //   * fuse_swiglu: the swiglu'd value straight into out[row]. The caller
        //     then never materialises the 2*inter gate/up buffer nor runs the
        //     separate swiglu pass.
        //   * !fuse_swiglu (raw epilogue, ILV only): the RAW pair, gate into
        //     out[row] and up into out[b_split + row] - the unfused [2*inter]
        //     layout of act_slot, which is what the e2m1x2 two-pass arm needs
        //     (it sums the two passes and runs ONE swiglu on the sum, because
        //     swiglu(x+y) != swiglu(x)+swiglu(y)). The K loop, the read path and
        //     the reduction are IDENTICAL on both epilogues - only the final
        //     write differs.
        // NUMERIC CONTRACT (corrected 2026-09-11): this pair body is NOT the
        // vec==2 shape of the unfused body - the old comment claiming "same single
        // scale multiply per accumulator / same floats as the unfused rows" was
        // left over from 7100ebfe and invalidated by 667c6f66. The unfused vec==2
        // body keeps FOUR accumulators a0..a3 persistent across the groups (four
        // scale-FMAs per group, epilogue (a0+a1)+(a2+a3)); this pair body folds
        // each group into ONE 4-element tree and does ONE scale multiply per
        // group. The two are already NOT bit-identical at ksplit==1. What IS still
        // guaranteed here: the per-group fma chains run in ascending g2 order with
        // one `g`/`u` update each (unroll DEPTH is not part of the contract), so
        // ksplit==1 reproduces the pre-K-split bit pattern exactly.
        // ksplit>1 changes the summation order to (g0..g4)+(g5..g9) - see
        // dsv41_gateup_ksplit().
        // k = dim and the launcher only admits the pair body when
        // (dim % 512) == 0, so the two-chunk-per-scale tail loop of the unfused
        // body has no work here (the launcher's ILV guard is the same term).
        if (pair_body) {
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
            // nv2f (k>>9 groups per row, no tail because k % 512 == 0) and
            // g_begin are the PROLOGUE's copies - the P4 prefetch needs them
            // there, so they are defined once, above, and reused here.
            // K-split group slice: this half walks the CONTIGUOUS groups
            // [half*nv2f/ksplit, (half+1)*nv2f/ksplit). ksplit==1 => [0, nv2f), the
            // original bounds. j and q below are unchanged because the cut always
            // lands on a 512-value group boundary (never inside a 32-value scale
            // block). half 0 owns the LOW half, half 1 the HIGH half.
            // g_end is the PROLOGUE's copy (the P4.2 pipeline issues the first
            // PDEPTH groups before the row loop, so it needs the slice bounds
            // too) - defined once, above, and reused here. ksplit==1 => [0, nv2f),
            // the original bounds.
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
                // Three ways to get this group's 16 weight bytes (gw0/gw1 gate,
                // uw0/uw1 up), all BIT-EXACT: same bytes, same lane offsets,
                // same consume order - only which memory they come from and WHEN
                // the copy is issued change.
                //
                //   PDEPTH == 1  (P4): the warp's FIRST group was staged into the
                //   single s_pf slot before the prologue barrier; every later
                //   group is the direct global read, on the dot's critical path.
                //   `from_pf` is constant false when the launch has pf == 0, so
                //   the non-prefetching shapes keep their codegen.
                //
                //   PDEPTH > 1  (P4.2): EVERY group of this warp's slice comes
                //   from its ring slot (g2-g_begin)%PDEPTH, and the slot is
                //   refilled with group g2+PDEPTH right after it is read. The
                //   wait_prior(PDEPTH-1) covers group g2 because exactly one
                //   commit group was added per consumed group (see the prologue).
                //
                //   fallback: the prefetch did not arm (inactive warp, or the
                //   alignment guard rejected the row) -> the same direct global
                //   reads as PDEPTH == 1.
                uint32_t gw0, gw1, uw0 = 0u, uw1 = 0u;
                if constexpr (PDEPTH == 1) {
                    const bool from_pf = pf_ok && (row == row_base) && (g2 == g_begin);
                    if (ILV) {
                        const uint4 v4 = from_pf
                            ? *reinterpret_cast<const uint4*>(pf_ring + (size_t)(lane << 4))
                            : ld_uint4_a16(g_row + (size_t)2 * q);
                        gw0 = v4.x; gw1 = v4.y; uw0 = v4.z; uw1 = v4.w;
                    } else {
                        const uint2 gw = from_pf
                            ? *reinterpret_cast<const uint2*>(pf_ring + (size_t)(lane << 3))
                            : ld_uint2_a8(g_row + q);
                        gw0 = gw.x; gw1 = gw.y;
                    }
                } else if (pf_ok && (row == row_base)) {
                    const int pi = g2 - g_begin;
                    uint8_t* slot = pf_ring + (size_t)(pi % PDEPTH) * kGateUpPfBytes;
                    dsv41_gateup_pf_wait_prior<PDEPTH - 1>();
                    if (ILV) {
                        const uint4 v4 = *reinterpret_cast<const uint4*>(slot + (size_t)(lane << 4));
                        gw0 = v4.x; gw1 = v4.y; uw0 = v4.z; uw1 = v4.w;
                    } else {
                        const uint2 gw = *reinterpret_cast<const uint2*>(slot + (size_t)(lane << 3));
                        gw0 = gw.x; gw1 = gw.y;
                        const uint2 uw =
                            *reinterpret_cast<const uint2*>(slot + 256 + (size_t)(lane << 3));
                        uw0 = uw.x; uw1 = uw.y;
                    }
                    // The slot has just been consumed - start group g2+PDEPTH in
                    // it. Out-of-range groups still COMMIT (with no copy): the
                    // commit count must stay `PDEPTH + groups consumed` for the
                    // wait above to keep covering group g2 in the tail.
                    const int gi = g2 + PDEPTH;
                    if (gi < g_end)
                        dsv41_gateup_pf_group<ILV>(slot, g_row, u_row, gi, lane);
                    dsv41_gateup_pf_commit();
                } else if (ILV) {
                    const uint4 v4 = ld_uint4_a16(g_row + (size_t)2 * q);
                    gw0 = v4.x; gw1 = v4.y; uw0 = v4.z; uw1 = v4.w;
                } else {
                    const uint2 gw = ld_uint2_a8(g_row + q);
                    gw0 = gw.x; gw1 = gw.y;
                    const uint2 uw = ld_uint2_a8(u_row + q);
                    uw0 = uw.x; uw1 = uw.y;
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
                // PDEPTH > 1 loads both rows in the read block above (from the
                // ring slot or, on the fallback, from gmem), so this deferred
                // load only exists on the P4 arm - keeping its exact emission
                // order for the DSV41_GATEUP_PIPELINE=1 A/B.
                if constexpr (PDEPTH == 1) {
                    // Recomputed here (not carried from the read block above):
                    // on the P4 arm it is the same constant-folded predicate, and
                    // it is `false` for every shape the launcher gives pf == 0.
                    const bool from_pf = pf_ok && (row == row_base) && (g2 == g_begin);
                    if (!ILV) {
                        const uint2 uw = from_pf
                            ? *reinterpret_cast<const uint2*>(pf_ring + 256 + (size_t)(lane << 3))
                            : ld_uint2_a8(u_row + q);
                        uw0 = uw.x; uw1 = uw.y;
                    }
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
            // P4.2: retire the pipeline's tail commits. All REAL groups were
            // waited for in-loop (iteration i waits group i), so what is left
            // here are the empty commits issued past the end of the slice -
            // wait(0) is therefore immediate. Kept so no thread can leave a
            // dangling commit group pending when the CTA unwinds.
            if constexpr (PDEPTH > 1) dsv41_gateup_pf_wait_prior<0>();
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
                if (fuse_swiglu) {
                    out[(size_t)row] = (g / (1.f + expf(-g))) * u;   // silu(gate) * up
                } else {
                    // RAW pair epilogue (the unfused [2*inter] layout): gate in
                    // the low half, up in the high half - the SAME convention the
                    // split arm above writes (rows < b_split are gate, rows >=
                    // b_split are up) and the SAME clamp each half got there
                    // (epi_mode 1: upper-only for the gate, both-sided for the
                    // up). `b_split` is the caller's `inter`, so this is
                    // out[row] / out[inter + row] of the caller's slot block.
                    // The e2m1x2 two-pass arm sums the two passes here and runs
                    // ONE swiglu on the sum (swiglu(x+y) != swiglu(x)+swiglu(y)).
                    out[(size_t)row] = g;
                    out[(size_t)b_split + (size_t)row] = u;
                }
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
                const uint32_t w0 = ld_uint32_a4(bp);
                const uint32_t w1 = ld_uint32_a4(bp + 4);
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
        } else if (vec == 3) {
            // 4 values per lane for k < 512, where the vec==2 main loop above
            // cannot run at all (nv2 = k >> 9 = 0 at the production k = 320).
            // This is the VERBATIM mirror of expert_gemv_fp4_down_reduce_kernel's
            // vec==3 body (same lane map, same group order, same
            // one-scale-multiply-per-group shape): the down direction has TWO
            // entries (the fused `_down_reduce` kernel and this batched one, which
            // is its DSV41_DOWN_FUSE=0 fallback) and BOTH read `g_down_fp4_mode`,
            // so the two branches are one unit - changing either one without the
            // other silently drops the bit-parity contract between the fused and
            // unfused arms (mode 3's lane map differs from mode 2's).
            //
            // One LDG.U16 = 2 packed bytes = 4 nibbles, one LDS.128 = the 4
            // activations, two LDS.64 = the LUT pairs: 5 L1TEX ops per 4 values
            // against 10 in the 2-value tail. The four values of a group sit
            // inside ONE 32-value scale block (j = lane*4, block = j >> 5 =
            // lane >> 3), so `sc` multiplies the accumulator once per group
            // instead of once per element: 6 FP ops per 4 values.
            const int nv4 = k >> 7;
            float a0 = 0.f, a1 = 0.f;
            for (int g = 0; g < nv4; ++g) {
                const int j = (g << 7) + (lane << 2);
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                const uint16_t w =
                    ld_uint16_a2(brow + (g << 6) + (lane << 1));
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
        } else if (vec) {
            // 32 lanes * 8 values = 256 values per iteration, four scale blocks.
            const int nv = k >> 8;              // full 256-value iterations
            const int off = lane << 2;          // byte offset of this lane's uint32
            for (int g = 0; g < nv; ++g) {
                const int j = (g << 8) + (lane << 3);
                const float sc = __uint_as_float(((uint32_t)srow[j >> 5]) << 23);
                // 256 packed values are 128 bytes, so group g starts at g*128, not g*64.
            const uint32_t word = ld_uint32_a4(brow + (g << 7) + off);
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
//
// __launch_bounds__(256, 6) - 6 blocks/SM, NOT 4. The launcher hardcodes
// warps=8 => blockDim is ALWAYS 256 (`dsv41_expert_down_reduce_fp4_batched`),
// so unlike the gate/up kernel this can be the literal block size.
//
// WHY 6 AND NOT 4 (the 2026-09-12 correction; this pair was (256, 4) and that
// was the +38% regression): the grid is ceil(dim / 8) = 896 blocks at the
// production dim = 7168. The kernel was DESIGNED around 6 blocks/SM:
//   6 blocks/SM x 148 SM = 888 resident ~= 896 grid  -> 1.01 waves
//   4 blocks/SM x 148 SM = 592 resident -> 1.51 waves, i.e. a second wave with
//   304 of 148-SM slots busy and a 1.5x tail.
// mode 2 (2-value tail) needs 40 regs => 6 blocks/SM and stayed at 17.2 us.
// mode 3 (4-value) needs 56 regs free-running. Register allocation is
// PER-FUNCTION, so that 56 covers the WHOLE kernel - even the vec==2
// instantiation - and 65536/(256*56) = 4.57 => 4 blocks/SM => 1.51 waves.
// That is the entire regression: 17.2 -> 23.8 us on nsys v7/v8 is exactly the
// 1.01 -> 1.51 wave cliff at this shape, and `DSV41_DOWN_VEC4=0` does NOT
// restore occupancy (the 56-reg mode-3 branch is still in the function).
//
// WHY THE ISOLATION MICROBENCH MISSED IT (/tmp/dv320, and the arm table in the
// vec==2 comment below): that bench re-runs the SAME buffers back-to-back, so
// its entire working set (w2 = 8 slots x dim x k/2 = 9.17 MB + a 40 KB act
// buffer) is L2-RESIDENT. Production streams w2 from HBM: w2 is never touched
// earlier in the step (the gate/up pass reads w1/w3, not w2), 9.17 MB per layer
// x 40 layers = 367 MB/step >> L2, measured 9.17 MB / 23.8 us = 385 GB/s, far
// below the HBM peak -> the kernel is LATENCY-bound, where resident threads
// (1024 -> 1536 per SM) and the wave count decide, not the instruction count.
// Hence "0.87x in isolation" coexisted with "+38% in production": at 4 blocks/SM
// the extra ILP wins on an L2-hot bench and loses on an HBM-cold stream.
// The same trap invalidated the 01291b2 verdict in the other direction - the
// 40 -> 54 reg occupancy cliff it attributed the +45% to was real; the
// microbench that "disproved" it was simply not reproducing the cache state.
//
// 42 regs is the hard cap here (65536/(256*6) = 42.67). The bench node's
// (256,6) mode-3 arm compiled to 40 regs with no spill and measured 0.90,
// i.e. ptxas has a compact legal schedule for the 4-value loop - this is not a
// spill-heavy pin. If a future change does start spilling, shrink the vec==3
// body's live set (float4 activation + 2 float2 LUT pairs) rather than raising
// the pin: 4 blocks/SM is a 1.51-wave configuration at this grid and is not
// recoverable by any instruction-level tuning.
//
// A/B on the bench node (same .so, one line): flip 6 -> 4 here and rebuild
// (see the AGENTS.md `--lib` recipe). Predicted: 23.8 -> ~18-19 us, -0.2 ms/step.
template <bool STAGED>
__launch_bounds__(256, 6)
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
    // ---- ACTIVATION ROW (grid.z): the MULTI-ROW dimension ------------------
    // Same contract as expert_gemv_fp4_batched_kernel (see the layout block
    // there): the caller's buffers are [rows][slot][...] with each row laid out
    // exactly as the rows == 1 call laid out its single row -
    //   ids        : [rows][slots]              row pitch = slots
    //   row_weight : [rows][slots][rw_stride]   row pitch = slots*rw_stride
    //   act        : [rows][slots][act_stride]  row pitch = slots*act_stride
    //   out        : [rows][n_total]            row pitch = n_total (this fused
    //                kernel overwrites ONE [n_total] row per activation row, so
    //                its output pitch is n_total, not slots*something)
    // rows == 1 leaves blockIdx.z == 0 and every offset below is 0, i.e. the
    // pre-existing single-row kernel bit for bit.
    //
    // ROW INDEPENDENCE: each row's serial ascending slot loop, its K loop, its
    // shuffle tree and its rounded product/add pair are the rows==1 ones; only
    // the base pointers move. Row r of a rows=m call is therefore BIT-IDENTICAL
    // to a rows==1 call on the same row (tests_dsv41_experts_mrows.cu).
    const int arow = (int)blockIdx.z;
    const size_t slot0 = (size_t)arow * (size_t)slots;            // this row's slot-0 index
    const float* act_row = act_base + slot0 * (size_t)act_stride;  // this row's slot-0 slice
    float* out_row = out + (size_t)arow * (size_t)n_total;
    if (STAGED) {
        // ONE cooperative pass stages every slot's activation. `act_stride` is
        // the caller's slice pitch and only the first `k` floats of each slice
        // are read - today the pitch is 2*inter (the swiglu half of the gate/up
        // output); once gate_up+swiglu fusion shrinks that slice to inter, the
        // caller passes the smaller pitch and this loop is unchanged.
        // MULTI-ROW: `act_row` shifts the whole [slots] slice block by this
        // row's pitch; the smem budget is per ROW, so it does not grow with
        // `rows` (each grid.z CTA stages its own row).
        for (int s = 0; s < slots; ++s) {
            const float* src = act_row + (size_t)s * (size_t)act_stride;
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
                                        : (act_row + (size_t)slot * (size_t)act_stride);
            // Per-slot derivation identical to expert_gemv_fp4_batched_kernel,
            // except that the weight row is selected here: this kernel has no
            // blockIdx.y, every warp owns its whole row. MULTI-ROW: the router's
            // per-slot scalars move with the activation row (row pitch = slots).
            const float rwv =
                (row_weight != nullptr) ? row_weight[(slot0 + (size_t)slot) * (size_t)rw_stride]
                                        : 1.f;
            const size_t e = (size_t)ids[slot0 + (size_t)slot];
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
                        ld_uint16_a2(brow + (g << 6) + (lane << 1));
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
                    const uint32_t w0 = ld_uint32_a4(bp);
                    const uint32_t w1 = ld_uint32_a4(bp + 4);
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
                // ⚠️ THESE NUMBERS ARE L2-HOT AND MUST NOT BE USED TO PICK A REGISTER
                // PIN. The bench re-runs one buffer set, so w2 stays in L2; production
                // streams w2 from HBM (never touched earlier in the step) and is
                // latency-bound, where blocks/SM and the wave count decide. Acting on
                // the "0.87 @ 56 regs / 4 blocks/SM is fastest" line is what produced
                // the +38% production regression: 56 regs is a 4-blocks/SM = 1.51-wave
                // configuration at this 896-block grid, against the designed
                // 6 blocks/SM = 1.01 waves. See the __launch_bounds__ note on the
                // kernel declaration. The 01291b2 occupancy cliff (40 -> 54 regs,
                // 6 -> 4 blocks/SM) was real; this bench "disproved" it only because
                // it does not reproduce the production cache state.
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
                const uint32_t word = ld_uint32_a4(brow + (g << 7) + off);
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
        if (lane == 0) out_row[(size_t)row] = tot;
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
            limit, row_weight, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
            /*act_e4m3=*/0);
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
__global__ void interleave_gateup_fp4_kernel(const uint8_t* __restrict__ g,
                                             const uint8_t* __restrict__ u,
                                             uint8_t* __restrict__ dst, long n8) {
    const long stride = (long)blockDim.x * (long)gridDim.x;
    // read-only sources, one write pair per iteration: the loop lets any grid
    // size cover the tensor, and every thread writes DISJOINT 16-byte slots.
    //
    // The SOURCES are the checkpoint's W1/W3 planes addressed through a
    // TP-sharded `DevBuf::view` (`g` is the W3 one the grouped arm calls
    // `bh_base`): a shard boundary can leave one rank's view a few bytes off
    // while every other rank is fine, and a `uint2` read of such a base is err
    // 716. Both reads therefore go through the alignment-safe helper; the bytes
    // are IDENTICAL on both paths. `dst` is our own freshly allocated pool, so
    // its store keeps the plain (8-byte-aligned) form.
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) {
        *reinterpret_cast<uint2*>(dst + (size_t)(2 * i) * 8) = ld_uint2_a8(g + (size_t)i * 8);
        *reinterpret_cast<uint2*>(dst + (size_t)(2 * i + 1) * 8) = ld_uint2_a8(u + (size_t)i * 8);
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
    interleave_gateup_fp4_kernel<<<(unsigned)nb, threads, 0, stream>>>(g, u, dst, n8);
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
                                        const int* ids, int slot, int act_e4m3, cudaStream_t s) {
    if (rows <= 0 || n_total <= 0 || k <= 0) return cudaSuccess;
    if (k % kAtomK != 0) return cudaErrorInvalidValue;
    // The e4m3 activation form exists only in the GEMV (the tcgen05 `kind::mxf4`
    // MMA is e2m1 x e2m1). Every routed-expert caller here is rows == 1 at
    // decode, but a rows > 1 call would silently take the wrong arm, so it is
    // refused rather than mis-decoded.
    if (act_e4m3 != 0 && !(rows == 1 && getenv("DSV41_NO_GEMV_FP4") == nullptr))
        return cudaErrorInvalidValue;
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
            bhs_base, bhs_stride, ids, slot, act_e4m3);
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
// `act_e4m3` (trailing, new): the activation `a` is e4m3 bytes (1/value) with
// `a_scale` f32 per 32 instead of e2m1 packed nibbles (DSV41_EXPERT_ACT_E4M3).
extern "C" int dsv41_expert_gate_up_fp4_indirect(
    const uint8_t* a, const float* a_scale, float* out, int rows, int dim, int inter, float limit,
    const uint8_t* w1_base, long w1_stride, const uint8_t* w1s_base, long w1s_stride,
    const uint8_t* w3_base, long w3_stride, const uint8_t* w3s_base, long w3s_stride,
    const int* ids, int slot, int act_e4m3, cudaStream_t stream) {
    return (int)launch_mxf4_indirect(a, a_scale, nullptr, out, rows, 2 * inter, dim, inter, 1, limit,
                                     nullptr, false, w1_base, w1_stride, w1s_base, w1s_stride,
                                     w3_base, w3_stride, w3s_base, w3s_stride, ids, slot, act_e4m3,
                                     stream);
}

// down, indirect, accumulating into `out` (epi_mode 3).
extern "C" int dsv41_expert_down_fp4_indirect(
    const float* act, float* out, int rows, int dim, int inter, const float* row_weight,
    const uint8_t* w2_base, long w2_stride, const uint8_t* w2s_base, long w2s_stride,
    const int* ids, int slot, cudaStream_t stream) {
    return (int)launch_mxf4_indirect(nullptr, nullptr, act, out, rows, dim, inter, -1, 3, 0.f,
                                     row_weight, true, w2_base, w2_stride, w2s_base, w2s_stride,
                                     w2_base, w2_stride, w2s_base, w2s_stride, ids, slot,
                                     /*act_e4m3=*/0, stream);
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

// P4.2 (DSV41_GATEUP_PIPELINE): REMOVED with the deep instantiations
// (2026-09-12, gate-hygiene). This used to hold the per-(device, ilv) dynamic
// smem ceiling for the depth 4/5 pipeline instantiations - depth 4/5 need more
// than the 48 KB default, so they were opted in with cudaFuncSetAttribute.
//
// It is gone for two reasons, and the ORDER matters:
//   1. with the depth pinned to 1 (see dsv41_gateup_pipeline) there is nothing
//      left to clamp - the 1-slot ring is 30848 B at the production shape;
//   2. more importantly, `cudaFuncSetAttribute(expert_gemv_fp4_batched_kernel<
//      ILV, 4|5>, ...)` ODR-USES those instantiations: deleting the depth
//      cascade in the launcher alone would still have left all four deep
//      instantiations (2 ILV x {4,5}) compiled into the .so through this probe.
//      A compile-time-only cleanup has to take every reference out, and this was
//      the second one.
// If a depth is ever re-enabled, BOTH have to come back together with the
// launcher's cascade (and so does the per-device opt-in, for the reason the old
// comment gave: cudaFuncSetAttribute is PER-CONTEXT, so a one-shot set on
// whichever device happened to be current would leave the other 7 ranks at the
// default).


// ============================================================================
// BATCHED expert entry points (DSV41_MOE_BATCH, default OFF). The Rust chain
// picks these to collapse one launch per (layer, top-k slot) into one launch
// per (layer, direction). See expert_gemv_fp4_batched_kernel for the numeric
// contract: per-slot outputs are DISJOINT, and a batched result is bit-identical
// to the sequential per-slot loop.
// ============================================================================

// gate/up, batched over the top-k slots: grid = (row_blocks, slots, rows) with
// blockIdx.y = slot and blockIdx.z = the ACTIVATION ROW (`rows` used to be
// validation-only; the kernel's row mapping and the [rows][slot][...] layout are
// documented at the top of expert_gemv_fp4_batched_kernel). `out` holds `slots`
// consecutive [2*inter] blocks, one per slot, `out_slot_stride` floats apart;
// nothing accumulates across slots. The activation is the ONE shared quantised row
// `a`/`a_scale` (per activation row).
// `ilv` (trailing, new in ABI 2): the w1/w3 pools are INTERLEAVED (DSV41_EXPERT_ILV
// on the Rust side). It selects the kernel's ILV instantiation, whose gate/up PAIR
// body reads both halves of a row from one region with ONE LDG.128 (the split body,
// which walks the `b` and `b_hi` bases separately, cannot address that layout and is
// never selected for ILV - the choice is compile-time, there is no runtime flag).
// ILV and the EPILOGUE are INDEPENDENT (decoupled 2026-09-12): `fuse` still selects
// swiglu'd [inter] vs the raw gate|up pair, so an interleaved pool is now legal with
// either write - which is what the e2m1x2 two-pass arm (raw [2*inter] output) needs.
// The interleaved pool keeps ONE hard requirement, the pair body's K contract
// (dim % 512 == 0, it walks whole 512-value groups and has no tail): rejected loudly
// rather than silently skipping the tail of every row.
extern "C" int dsv41_expert_gate_up_fp4_batched(
    const uint8_t* a, const float* a_scale, float* out, long out_slot_stride, int rows, int dim,
    int inter, float limit, int slots, const uint8_t* w1_base, long w1_stride,
    const uint8_t* w1s_base, long w1s_stride, const uint8_t* w3_base, long w3_stride,
    const uint8_t* w3s_base, long w3s_stride, const int* ids, int ilv, int act_e4m3,
    cudaStream_t stream) {
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
        // (round-18 bug) - which is why the CALLER's pitch is what binds the
        // decision (see below), for the interleaved pool too: `2*inter` selects
        // the raw pair epilogue, `inter` the swiglu'd one. The `ilv` guard below
        // is about the pair body's K contract, NOT about the epilogue.
        if (e == nullptr) return 1;
        return atoi(e) != 0 ? 1 : 0;
    }();
    // ⚠️ SINGLE SOURCE OF TRUTH for the [inter] vs [2*inter] layout: the CALLER's
    // out_slot_stride (the twopass-degen-hunt verdict — round-18's root cause was
    // Rust and this launcher each deriving `fuse` independently; the caller's
    // `two` flag only existed on the Rust side, so the kernel kept fusing while
    // the caller expected unfused, writing swiglu'd [inter] with the high half
    // as uninitialized garbage). Binding fuse to the pitch makes the two sides
    // structurally unable to disagree: `out_slot_stride == inter` = caller wants
    // the fused epilogue; `2*inter` = caller wants raw gate|up.
    const int fuse = (g_fuse && g_expert_fp4_mode == 2 && (dim % 512) == 0 &&
                      out_slot_stride == (long)inter) ? 1 : 0;
    // The interleaved pool needs the gate/up PAIR body (the only body that reads
    // both halves of a row from one region) and that body walks whole 512-value
    // groups (`nv2f = k >> 9`) with NO tail - so an interleaved pool is legal for
    // BOTH epilogues, but only with dim % 512 == 0. Refuse the rest loudly instead
    // of silently skipping the tail of every row.
    // (The former `ilv && !fuse` refusal is gone: `fuse` selects the WRITE
    // epilogue, `ilv` the READ layout, and the two are independent - the kernel's
    // pair-body epilogue now implements both, see the kernel header.)
    if (ilv && ((dim % 512) != 0)) return (int)cudaErrorInvalidValue;
    // The SAME predicate the kernel derives for its pair-body branch (there
    // `b_split == inter > 0`, so the two agree): everything below - n_total,
    // ksplit, pf - is a property of the pair body, not of the fused write.
    const bool pair_body = (fuse != 0) || (ilv != 0);
    // K-split (DSV41_GATEUP_KSPLIT, default 2 = ON). Only the PAIR body implements
    // the cross-half merge, so the plain-layout split arm always runs ksplit=1
    // (otherwise both halves of a row would compute the full row and race on the
    // same out[row]). The interleaved arm IS the pair body (with the raw
    // epilogue), so it keeps the K-split - and therefore the same launch geometry
    // the fused arm uses (warps*ksplit per CTA), which is what makes an
    // E4M3+ILV vs fused+ILV A/B a layout-only comparison. blockDim must stay
    // <= 1024 threads: warps*ksplit <= 32.
    int ksplit = pair_body ? dsv41_gateup_ksplit() : 1;
    while (ksplit > 1 && warps * ksplit > 32) --ksplit;
    // n_total is the number of OUTPUT ROWS the warps own, which is a property of
    // the body and NOT of the per-slot output pitch: the pair body owns ONE row
    // per (gate, up) pair - inter rows, whether it writes the swiglu'd half or the
    // raw pair - so the ILV raw launch has n_total == inter while its slot pitch
    // (the caller's out_slot_stride) is 2*inter. Only the plain-layout split arm
    // walks all 2*inter rows one at a time. gating on `fuse` here would make the
    // ILV raw launch skip the up half entirely.
    const int n_total = pair_body ? inter : 2 * inter;
    // grid.x is the ROW count (n_total/rows), NOT the warp count: with ksplit the
    // CTA carries rows*ksplit warps but still owns `rows` rows. At rows=8 /
    // ksplit=2 this stays (40, 6) = 240 CTAs with 16 warps each (3840 warps in
    // flight vs 1920) - the recommended shape, which does NOT double the per-CTA
    // s_act+LUT prologue. rows=4 / ksplit=2 additionally gives the 480-CTA shape.
    const int ctas_x = (n_total + warps - 1) / warps;
    // P4 (DSV41_GATEUP_CPASYNC + DSV41_GATEUP_PIPELINE): the weight pipeline
    // stages PDEPTH k-groups of gate+up per warp, so the PAIR-BODY launch reserves
    // kGateUpPfBytes*PDEPTH per warp on top of the s_act + LUT [+ s_ks] pool.
    // The kernel's s_pf/s_ks pointers are derived from this same expression (in
    // the same order) - a mismatch reads or writes past the allocation. The
    // prefetch substitution lives in the READ path, which the two epilogues
    // share, so the raw-pair launch (ilv && !fuse) gets it too and the plain
    // split arm + the down direction launch with pf=0 and are sized exactly as
    // before.
    const int pf = (pair_body && dsv41_gateup_cpasync()) ? 1 : 0;
    const int nwarps = warps * ksplit;
    // PDEPTH is PINNED TO 1: only the <ILV, 1> instantiations exist (see
    // dsv41_gateup_pipeline for why - depth 2/5 both measured +0.04ms, so the
    // other 8 instantiations were pure artifact bloat), and the depth is a
    // COMPILE-TIME kernel parameter (cp.async.wait_group takes an immediate)
    // that the dispatch below therefore no longer selects.
    // The former depth clamps are gone with them: clamping against this warp's
    // SLICE (a warp owns ceil(nv2f/ksplit) = 5 groups) and against the
    // per-device opt-in smem ceiling (dsv41_gateup_pf_smem_cap) only existed to
    // keep a DEEP request launchable. A 1-slot ring is the P4 layout, 30848 B at
    // the production shape, always inside the 48 KB default - so the clamp has
    // nothing left to clamp and the cudaFuncSetAttribute opt-in (which is itself
    // what referenced <ILV,4|5>) is gone. The env is still read once for the
    // warning; its value cannot change the launch.
    dsv41_gateup_pipeline();
    const size_t smem_fixed =
        (size_t)dim * sizeof(float) + 256 * sizeof(float2) +
        ((ksplit > 1) ? (size_t)nwarps * sizeof(float2) : (size_t)0);
    const size_t smem_pf_stride = (size_t)nwarps * kGateUpPfBytes;
    const size_t smem = smem_fixed + (pf ? smem_pf_stride : (size_t)0);
    // grid = (output-row blocks, slots, rows): blockIdx.y = slot (unchanged),
    // blockIdx.z = the ACTIVATION ROW (new - `rows` used to be validation-only,
    // so every call computed one activation row). rows == 1 gives grid.z == 1 and
    // the pre-existing grid, so every rows == 1 caller is bit-for-bit unchanged.
    // The per-row buffer pitches are derived INSIDE the kernel from the existing
    // arguments + gridDim.y - see the layout contract on the kernel.
    dim3 grid((unsigned)ctas_x, (unsigned)slots, (unsigned)rows);
    const unsigned block_threads = (unsigned)(nwarps * 32);
    // PDL (see dsv41_experts_pdl_or_plain): the consumer's grid may start during
    // quant_fp4's tail; the kernel's entry cudaGridDependencySynchronize() gates
    // the activation staging. NOTE the full argument list: the cudaLaunchKernelEx
    // path does not apply the kernel's default arguments, so every trailing slot
    // is spelled out.
    auto gateup_launch = [&](auto kern) -> cudaError_t {
        return dsv41_experts_pdl_or_plain(kern, grid, dim3(block_threads), smem, stream, nullptr, 0,
                                          a, a_scale, out, out_slot_stride, n_total, dim, inter, 1,
                                          limit, nullptr, 0, w1_base, w1_stride, w1s_base, w1s_stride,
                                          w3_base, w3_stride, w3s_base, w3s_stride, ids,
                                          g_expert_fp4_mode, fuse, ksplit, pf, act_e4m3);
    };
    cudaError_t le;
    // ONE depth (1) x 2 ILV instantiations. The former 5-way depth cascade
    // existed only while dsv41_gateup_pipeline() could return 2..5; with the gate
    // pinned at 1 the other 8 instantiations were unreachable code that still
    // landed in the .so (each is a full copy of this kernel's body).
    if (ilv) {
        le = gateup_launch(expert_gemv_fp4_batched_kernel<true, 1>);
    } else {
        le = gateup_launch(expert_gemv_fp4_batched_kernel<false, 1>);
    }
    if (le != cudaSuccess) { (void)cudaGetLastError(); return (int)le; }
    return (int)cudaGetLastError();
}

// Capability marker for the Rust-side `DSV41_EXPERT_ACT_E4M3` gate. The gate
// changes what the CALLER writes into `a` (e4m3 bytes instead of packed e2m1),
// and `act_e4m3` is a trailing argument of the launchers above - a stale .so
// would ignore it and decode e4m3 bytes as fp4 nibbles, i.e. a silent wrong
// answer, not a failure. Exporting a symbol only the direct-e4m3 build has is
// the established probe pattern (`Device::supports_*`), so a stale .so leaves
// the gate OFF with a one-shot notice instead.
extern "C" int dsv41_expert_act_e4m3_cap(void) { return 1; }
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
    // blockIdx.z = the activation row (the multi-row dim, same contract as the
    // gate/up launcher: `act`/`row_weight`/`ids`/`out` are [rows][slot][...]).
    // rows == 1 reproduces the previous grid exactly.
    dim3 grid((unsigned)((dim + warps - 1) / warps), (unsigned)slots, (unsigned)rows);
    // PDL (see dsv41_experts_pdl_or_plain): same consumer contract as the
    // gate/up call above -- the producer is the gate/up launch that wrote the
    // swiglu'd `act_base`, and the kernel's entry sync gates the staging.
    cudaError_t le = dsv41_experts_pdl_or_plain(
        expert_gemv_fp4_batched_kernel<false, 1>, grid, dim3(warps * 32),
        (size_t)inter * sizeof(float) + 256 * sizeof(float2), stream, act_base, act_stride, nullptr,
        nullptr, out, out_slot_stride, dim, inter, -1, 2, 0.f, row_weight, rw_stride, w2_base,
        w2_stride, w2s_base, w2s_stride, w2_base, w2_stride, w2s_base, w2s_stride, ids,
        g_down_fp4_mode, /*fuse_swiglu=*/0, /*ksplit=*/1, /*pf=*/0, /*act_e4m3=*/0);
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
    // blockIdx.z = the activation row (the multi-row dim). The stage buffer is
    // per ROW (each grid.z CTA stages only its own row's [slots][k] slices), so
    // `staged_bytes` - and therefore the STAGED/fallback decision - does not
    // change with `rows`. rows == 1 reproduces the previous grid exactly.
    const dim3 grid((unsigned)((dim + warps - 1) / warps), 1u, (unsigned)rows);
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

// ============================================================================
// w2 L2 PREWARM (DSV41_W2_PREWARM, default OFF, `=1` enables)
// ============================================================================
// WHY. The down GEMV (`expert_gemv_fp4_down_reduce_kernel`) streams w2 straight
// from HBM: w2 is never touched earlier in the step (the gate/up pass reads
// w1/w3, not w2), and the down kernel measures 286-385 GB/s against a ~7 TB/s
// part, i.e. it is LATENCY-bound, not bandwidth-bound. Same argument as the
// "isolation microbench missed it" note on that kernel: a cache-warm run is a
// different machine. The fix is therefore not more ILP but a shorter LATENCY,
// and the only way to shorten it is to have the bytes in L2 before the down
// kernel asks for them.
//
// WHY HERE AND NOT INSIDE THE DOWN KERNEL. An in-kernel `prefetch.global.L2`
// one slot ahead cannot work: each warp's per-slot arithmetic is ~30 cycles
// (~20 ns) against a ~600 ns HBM latency, so the "lead" a software pipeline can
// buy is two orders of magnitude short of the latency it is trying to hide. The
// fetch has to start OUTSIDE the consumer, in a window where the memory system
// is otherwise idle: the gate/up tail.
//
// WHY NOT INSIDE THE GATEUP KERNEL (the "epilogue prefetch" variant). It would
// work (the gate/up grid is ONE wave of 320 CTAs at the default shape - 8 rows x
// 2 ksplit = 512 threads, 4 CTAs/SM x 148 SM = 592 slots - so all of its CTAs do
// reach their epilogue at roughly the same time) but it means adding five
// parameters to - and perturbing the schedule of - the most fragile kernel in
// the tree (it is register-pinned at 48/64 regs and its launch geometry depends
// on __launch_bounds__; the +38% wave-cliff regression in its history came from
// exactly this kind of change). A separate entry point costs one launch per
// layer and keeps the hot kernel's codegen bit-for-bit untouched.
//
// WHEN IT RUNS. The grid is launched on the SAME stream right after the gate/up
// launch, with the same PDL attribute as every other kernel in this file, so its
// CTAs begin launching as the gate/up CTAs retire - i.e. the prefetch burst is
// issued in the gate/up ramp-down, which is dead time for the memory system, and
// completes under the down kernel's first microseconds. The kernel deliberately
// does NOT call cudaGridDependencySynchronize(): it reads only `ids` (a router
// output several kernels upstream) and the w2 pools (weights), never anything
// the gate/up launch wrote, so it is legal for it to race the producer.
//
// HOW MUCH. One layer's down reads w2 for `slots` experts: `slots` x dim x
// (inter/2) bytes (8 x 7168 x 160 = 9.17 MB at the production shape) plus the
// e8m0 scale rows (8 x 7168 x 10 = 0.57 MB). Against a Blackwell-class L2
// (~120 MB) that is under 10%, so the burst survives until the down consumes
// it; against the fp4 model's per-STEP w2 traffic (40 layers x 9.17 MB = 367 MB)
// it obviously does not, which is why this is strictly a same-layer trick.
//
// BIT-EXACTNESS. The kernel issues L2 prefetch hints and writes nothing. It
// changes no pointer, no order, no arithmetic - only which level of the memory
// hierarchy answers the down kernel's subsequent loads. The prefetch region is
// EXACTLY the region the down kernel reads: the down GEMV's row `r` lives at
// `w2_base + e*w2_stride + r*kbytes`, rows are walked contiguously 0..dim-1, so
// the union over r is [e*w2_stride, e*w2_stride + dim*kbytes) with no holes.
//
// COST. One launch per layer (a 32-thread CTA per 16 KB chunk) issuing one
// `cp.async.bulk.prefetch.L2.global` each; no shared memory, no registers to
// speak of, no completion wait. It adds no HBM traffic at all - it moves bytes
// that the down kernel would fetch anyway.
constexpr int kW2PfChunk = 16384;   // bytes per cp.async.bulk.prefetch

// cp.async.bulk.prefetch.L2.global [srcMem], size;  (PTX, sm_90+)
// `size` must be a multiple of 16 and `srcMem` 16-byte aligned - both hold by
// construction for these pools (per-expert strides are 512-byte-aligned tensor
// allocations and every span used here is a multiple of 16), and the callers
// round defensively anyway. Pre-sm_90 (never built in production: this TU is
// sm_100a/sm_103a only) it compiles to nothing.
__device__ __forceinline__ void dsv41_w2_pf_bulk(const void* gmem, unsigned bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(gmem), "r"(bytes));
#else
    (void)gmem; (void)bytes;
#endif
}

// Line fallback for a chunk whose base is not 16-byte aligned (never taken for
// the fp4 expert pools; exists so a future misaligned pool degrades to hints
// instead of a fault - the same defensive rule the P4 cp.async prologue uses).
__device__ __forceinline__ void dsv41_w2_pf_line(const void* gmem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(gmem));
#else
    (void)gmem;
#endif
}

// grid = (slots, nchunks); ONE 32-thread CTA per (slot, 16 KB chunk). Only
// threadIdx.x == 0 issues - the prefetch is a memory-system operation on a
// region, not a per-thread load, and 32 identical issues would be 32x the TMA
// work for the same bytes.
__global__ void w2_l2_prewarm_kernel(const uint8_t* __restrict__ w2_base, long w2_stride,
                                     const uint8_t* __restrict__ w2s_base, long w2s_stride,
                                     const int* __restrict__ ids, long sel_bytes, long sc_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (threadIdx.x != 0) return;
    const int nsel = (int)((sel_bytes + kW2PfChunk - 1) / kW2PfChunk);
    const int nsc = (sc_bytes > 0) ? (int)((sc_bytes + kW2PfChunk - 1) / kW2PfChunk) : 0;
    const int j = (int)blockIdx.y;
    if (j >= nsel + nsc) return;
    // ids[slot] is the SAME value the down kernel resolves (`blockIdx.y` there is
    // the slot as well), so the warm set and the consumed set are identical.
    const size_t e = (size_t)ids[blockIdx.x];
    const uint8_t* p;
    long span;
    if (j < nsel) {
        p = w2_base + e * (size_t)w2_stride + (size_t)j * kW2PfChunk;
        span = sel_bytes - (long)j * kW2PfChunk;
    } else {
        p = w2s_base + e * (size_t)w2s_stride + (size_t)(j - nsel) * kW2PfChunk;
        span = sc_bytes - (long)(j - nsel) * kW2PfChunk;
    }
    unsigned n = (unsigned)(span < (long)kW2PfChunk ? span : (long)kW2PfChunk);
    if ((((uintptr_t)p) & 15u) == 0) {
        n &= ~15u;                 // documented size rule: a multiple of 16
        if (n != 0) dsv41_w2_pf_bulk(p, n);
    } else {
        for (unsigned o = 0; o < n; o += 128) dsv41_w2_pf_line(p + o);
    }
#endif
}

// Fire-and-forget. NEVER fails the step: a launch error is swallowed (the down
// kernel that follows is the correctness path, this is a hint).
//
// `sel_bytes` = dim * (inter / 2), `sc_bytes` = dim * (inter / 32) -- the exact
// spans the caller's down launch reads (rows = dim, k = inter). Both are plain
// byte counts so this entry point does not have to know the direction's
// (rows, k) convention.
extern "C" int dsv41_w2_l2_prewarm(const uint8_t* w2_base, long w2_stride,
                                   const uint8_t* w2s_base, long w2s_stride,
                                   const int* ids, int slots, long sel_bytes, long sc_bytes,
                                   cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("DSV41_W2_PREWARM");
        return (e != nullptr && e[0] == '1') ? 1 : 0;   // default OFF (serve A/B: +0.04ms regression,
                                                          // warmer's SM contention > L2 hit benefit)
    }();
    if (!enabled || w2_base == nullptr || w2s_base == nullptr || ids == nullptr) return 0;
    if (slots <= 0 || sel_bytes <= 0) return 0;
    const int nsel = (int)((sel_bytes + kW2PfChunk - 1) / kW2PfChunk);
    const int nsc = (sc_bytes > 0) ? (int)((sc_bytes + kW2PfChunk - 1) / kW2PfChunk) : 0;
    dim3 grid((unsigned)slots, (unsigned)(nsel + nsc));
    cudaError_t le = dsv41_experts_pdl_or_plain(w2_l2_prewarm_kernel, grid, dim3(32), 0, stream,
                                                w2_base, w2_stride, w2s_base, w2s_stride, ids,
                                                sel_bytes, sc_bytes);
    if (le != cudaSuccess) {
        // Best effort. A stale driver / an unsupported arch keeps the old timing,
        // and the sticky error is cleared so the next real launch is not poisoned.
        (void)cudaGetLastError();
        return 0;
    }
    (void)cudaGetLastError();
    return 0;
}

// =============================================================================
// PHASE 1 — tcgen05 swapAB gate/up kernel (SKELETON)
// =============================================================================
// STATUS: SKELETON, NOT IN THE BUILD. build.sh does not define
// DSV41_TCGEN05_GATEUP_SKELETON, so nothing below is compiled into the .so.
// Every body IS written out (no empty stubs) — the fill-in work is:
//   (1) GPU-verify the operand/descriptor arithmetic against Phase 0 with the
//       real K (the harness is tests_tcgen05_mxf8f6f4_1x.cu + the real
//       checkpoint bytes),
//   (2) tune kPackK / kRing (see the SMEM BUDGET note),
//   (3) Phase 2 wiring: the pool/ids launcher, the Rust FFI, the gate.
//
// Compile-verify (no GPU needed; this is the cheapest possible regression net):
//     nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//          -DDSV41_TCGEN05_GATEUP_SKELETON=1 -c kernels/cuda/dsv41_experts_mxf4.cu -o /tmp/t5.o
//     ptxas -v (or `-Xptxas -v`) must show 0 spills and the
//     `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X` spelling.
//     Golden reference for every layout constant below: Phase 0,
//     kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu (probe_kernel / ph0_parity_kernel).
//
// -----------------------------------------------------------------------------
// WHY THIS KERNEL (docs/agent/expert-tcgen05-plan.md §0/§2)
// -----------------------------------------------------------------------------
// The expert path is DRAM-bound: gate/up at (dim=5120, inter=1920) reads
// 2*inter*dim/2 = 9.83 MB of packed fp4 weights per direction, and the current
// M=1 path runs the hardware-fixed M=128 tcgen05 tile with the SINGLE activation
// row as its A operand — 128x redundant work per real output row. Measured
// 16.8 GB/s = 0.2% of the part (dsv41_experts_mxf4.cu:560).
//
// swapAB fixes the MAPPING (not the staging): the WEIGHTS become the M operand.
//   A = W1W3 [2*inter, dim] fp4 e2m1   -> M = 2*inter = 3840 = 30 FULL 128-row
//                                         tiles, ZERO redundancy
//   B = act   [dim, 8]      e4m3        -> N = 8 (the minimum legal N for
//                                         mxf8f6f4; only column 0 carries the
//                                         token, so the ACTIVATION side is 8x
//                                         redundant — 40 KB/step, negligible)
//   D = TMEM [128, 8], only column 0 kept.
//   K = dim = 5120 = 160 scale blocks of 32 = 160 MMAs, all accumulating into
//   the same TMEM D (enable_input_d = 0 only on the very first one).
//
// But mapping alone is NOT the fix — the staging is. mxf4_gemm_kernel:348-421 is
// LDG -> STS -> fence -> __syncthreads -> MMA, one stage at a time: every one of
// the 160 K stages pays its full DRAM latency in front of its own MMA. swapAB
// with the same staging would still be a 0.2% floor. This kernel therefore has a
// kRing-deep TMA ring (below) and recycles a slot only after `tcgen05.commit`
// proves the slot's MMA has retired.
//
// -----------------------------------------------------------------------------
// THE ONE LAYOUT FACT THAT COSTS SMEM (Phase 0 header, "easy to get wrong")
// -----------------------------------------------------------------------------
//   *** mxf8f6f4 CONSUMES FP4 UNPACKED: one e2m1 element per BYTE. ***
//   (CUTLASS cute/arch/mma_sm100_desc.hpp: float_e2m1_unpacksmem_t ->
//    MXF8F6F4Format::E2M1, while the packed float_e2m1_t -> MXF4Format::E2M1.
//    The packed nibble form belongs to kind::mxf4 only.)
//   The checkpoint stores 2 fp4 per byte, so every weight byte must be expanded
//   1:2 on the way to smem. No copy engine can do that, so each ring slot needs
//   TWO buffers: the raw packed staging area the TMA fills, and the unpacked
//   operand the MMA reads. That is the single biggest smem cost here, and the
//   reason a plain "TMA straight into the operand" design is impossible.
//
//   ALTERNATIVE B (flagged, NOT implemented — the lead's call): run kind::mxf4
//   here instead. Both operands packed -> no unpack pass, HALF the operand smem,
//   and the MMA is the one tests_tcgen05_mxf4.cu already validates numerically.
//   The price is the activation format (e2m1 instead of e4m3), i.e. exactly the
//   numeric safety margin the Phase 0 plan bought by choosing e4m3, and mxf4 is
//   2X (scale granularity 64) so consecutive checkpoint scale words must be
//   paired — which mxf4_gemm_kernel already does today. The switch is ~30 lines
//   (idesc formats, LBO/SBO, the ring sizing, drop the unpack). Phase 0 was built
//   on mxf8f6f4, so this skeleton follows the plan and keeps Alternative B as a
//   note; if the e4m3 margin turns out not to be needed, B is strictly cheaper.
//
// -----------------------------------------------------------------------------
// SCALE FACTORS: PACKED IS FORCED, NOT CHOSEN
// -----------------------------------------------------------------------------
// Phase 0 runs two SF hypotheses: PACKED (4 blocks per 32-bit TMEM word, byte
// selector = block & 3, word column advances 4 per 32-row group) and PERBLK
// (one block per word in byte 0, column advances 4 per block).
//   PERBLK COSTS 4 TMEM COLUMNS PER K-BLOCK. At the real K that is
//   4 * (5120/32) = 640 columns for SFA alone, against the 512 columns a CTA
//   has. PERBLK cannot be shipped — it only ever existed to disambiguate the
//   byte-selector model in Phase 0. PACKED needs 4*40 = 160 (SFA) + 40 (SFB),
//   which fits: D(8) + 160 + 40 = 208 -> tcgen05.alloc 256 columns.
// Consequence: this kernel's per-MMA SF addressing is
//   SFA column = sfa_col + 4*(b >> 2),  a_sf_id = b & 3   (b = global 32-block)
//   SFB column = sfb_col + (b >> 2),    b_sf_id = b & 3
// Exactly the PACKED branch of ph0_parity_kernel. Contract: dim % 128 == 0
// (so the 4-block word never straddles the row end).
//
// -----------------------------------------------------------------------------
// SMEM BUDGET (why it is STATIC smem)
// -----------------------------------------------------------------------------
// Everything fits in STATIC shared memory (< 48 KiB), deliberately: dynamic
// smem needs cudaFuncSetAttribute, and the plan's §5 capture rules put those
// outside any graph capture (err 900/901 in-capture). The static_assert below
// fails the BUILD if a tuning change overflows the window instead of failing at
// launch with an unhelpful error. Going deeper than kRing=3 needs dynamic smem
// + an INIT-TIME (never capture-time) cudaFuncSetAttribute — that is the
// production tuning axis, see the DEPTH note after the constants.
//
// -----------------------------------------------------------------------------
// OCCUPANCY / WHY THE MICROBENCH MUST RUN slots > 1  (read before timing!)
// -----------------------------------------------------------------------------
// grid = (2*inter/128, slots) = (30, slots) at the production shape, and each CTA
// allocates 256 of the 512 TMEM columns -> at most 2 CTAs/SM. At slots=1 the
// whole grid is 30 CTAs = 15 SMs of ~148 busy, i.e. the isolated number will be
// LATENCY-bound and meaningless for the serve decision. The DRAM argument:
// in-flight bytes = (kRing-1) * (kPackK/2 * 128 + kPackK) ~= 8 KiB per CTA; the
// 8 TB/s * ~700 ns latency product wants ~5.6 MB in flight, which 240 CTAs
// (slots=8) get to ~1.9 MB and 30 CTAs get nowhere near. Consequence:
//   * measure at slots=8 (the real MoE dispatch shape), and
//   * if slots=8 still lands far from the floor, the fix is K-SPLIT (more CTAs
//     over the same weights), which needs a deterministic ascending-order
//     reduce — see the K-SPLIT TODO near the launcher, NOT this kernel's ring.
//
// -----------------------------------------------------------------------------
// NAMING / REUSE
// -----------------------------------------------------------------------------
// Everything new lives in `namespace tc5` so that enabling this block cannot
// collide with the proven helpers already at file scope (tc_alloc, mbar_init,
// tc_commit, tc_st_x4, make_desc, ...). Reused verbatim from the file's anonymous
// namespace (identical semantics, no second copy to drift): smem_addr, tc_alloc,
// tc_relinquish, tc_dealloc, tc_commit, tc_wait_ld, tc_wait_st, tc_fence_*,
// mbar_init, mbar_wait. New here: the mxf8f6f4 MMA + its descriptors, the TMA
// bulk/expect_tx pair, the fp4 nibble expansion, the x1 TMEM store.
// =============================================================================

#ifdef DSV41_TCGEN05_GATEUP_SKELETON

#include <cstdlib>  // getenv (launcher gate)

namespace tc5 {

// ------------------------------------------------------------------ geometry
// Every constant is traceable to Phase 0: kMTile/kNTile are the instruction's
// own constraints (CUTLASS SM100_MMA_MXF8F6F4_SS static_asserts M == 128 and
// N a multiple of 8 in [8,256]); kKStep is the 1X scale granularity; the two
// descriptor strides are PROBE_LBO_BYTES / PROBE_SBO_BYTES.
constexpr int kMTile = 128;  // MMA M, pinned by the instruction
constexpr int kNTile = 8;    // minimum legal N; only column 0 is kept
constexpr int kKStep = 32;   // one 1X scale block == one MMA (k_size = 0, dense K32)
constexpr int kPackK = 64;   // K elements staged per ring slot (multiple of kKStep)
constexpr int kNStep = kPackK / kKStep;  // MMAs issued per slot
constexpr int kRing = 3;     // ring depth (slots in flight). See DEPTH NOTE below.
constexpr int kThreads = 128;  // 4 warps == the 4 TMEM lane partitions (warp*32)
constexpr int kLboBytes = 128;  // 8 x 16 B: K-chunk stride of the K=32 atom (Phase 0)
constexpr int kSboBytes = 256;  // 16 x 16 B: 8-row-group stride (Phase 0)
constexpr int kTmemCols = 256;  // power of two >= 8 + 160 + 40 = 208
constexpr int kMaxDim = 5120;   // gate/up dim; sizes the SF staging chunk
constexpr int kSfWords = kMaxDim / 32 / 4;    // PACKED SF words per row = 40
constexpr int kSfaCols = 4 * kSfWords;        // 160 (one column per (quad, row-group))
constexpr int kSfbCols = kSfWords;            // 40

// Per-K_STEP operand bytes. A = 128 rows x 32 B unpacked fp4 = 4096 B;
// B = 8 rows x 32 B e4m3 = 256 B. Same numbers as PH0_A_BYTES / PH0_B_BYTES.
constexpr int kABytes = kMTile * kKStep;   // 4096
constexpr int kBBytes = kNTile * kKStep;   // 256
// Raw (packed) staging: half of the unpacked A, full e4m3 B.
constexpr int kARawBytes = kMTile * (kPackK / 2);  // 128 x 32 = 4096
constexpr int kBRawBytes = kNTile * kPackK;        //   8 x 64 = 512

// SF staging chunk: ONE 32-row group of A scales (32 * dim/32 bytes, contiguous
// in the pool because the CTA's rows are consecutive) + the whole B scale row.
// Staged one group at a time so the buffer stays small (a whole 128-row block
// would be 20 KB and does not fit next to the ring).
constexpr int kSfChunkRows = 32;
constexpr int kSfAChunkBytes = kSfChunkRows * (kMaxDim / 32);  // 32 * 160 = 5120
constexpr int kSfBChunkBytes = kNTile * (kMaxDim / 32);        //  8 * 160 = 1280

// TMA transaction count for one ring stage: 128 A rows + 1 activation row.
// (Rows 1..7 of B are zero and are NOT transferred — see the prologue.)
constexpr unsigned kStageTxBytes = kMTile * (kPackK / 2) + kPackK;

struct Smem {
    // ---- ring: raw (what the TMA lands) then unpacked (what the MMA reads) ----
    // a_raw is [row][kPackK/2 packed bytes]; a_op is the canonical UMMA Major-K
    // SWIZZLE_NONE operand layout, one 16-byte-chunk-atom per K_STEP:
    //     unit16(m, kb) = (m % 8) + 8*kb + 16*(m / 8)      kb in {0,1}
    // i.e. LBO = 128 B between the two K-chunks, SBO = 256 B between 8-row
    // groups. Identical to the probe's s_a / ph0_parity_kernel's staging.
    alignas(1024) uint8_t a_raw[kRing][kARawBytes];
    alignas(1024) uint8_t a_op[kRing][kNStep][kABytes];
    alignas(16) uint8_t b_raw[kRing][kBRawBytes];
    alignas(16) uint8_t b_op[kRing][kNStep][kBBytes];
    alignas(1024) uint8_t sf_stage[kSfAChunkBytes > kSfBChunkBytes ? kSfAChunkBytes
                                                                  : kSfBChunkBytes];
    alignas(8) uint64_t tma_bar[kRing];  // TMA tx-count completion, one per slot
    alignas(8) uint64_t mma_bar[kRing];  // tcgen05.commit retirement, one per slot
    uint32_t tmem_base;
};

// The 48 KiB static window. If a tuning change trips this, do NOT switch to
// dynamic smem casually: see the SMEM BUDGET note in the header (capture rules).
static_assert(sizeof(Smem) <= 48 * 1024, "tc5 ring does not fit static smem");
// sizeof at the shipped constants: a_raw 12288 + a_op 24576 + b_raw 1536 +
// b_op 1536 + sf_stage 6400 + barriers 52 ~= 46.4 KiB -> 47 KiB with alignment.

// DEPTH NOTE. kPackK and kRing are the two knobs that trade smem for in-flight
// bytes, and they are the ONLY reason this kernel can beat the 0.2% floor:
//   in flight ~= (kRing - 1) * kPackK/2 * kMTile  bytes  ( 8 KiB at 3/64 )
// Raising kRing to 4 (10.7 KiB in flight) still fits the static window at
// kPackK=32; raising kPackK to 128 (16 KiB in flight) does NOT and needs the
// dynamic-smem path. Measure both before picking — the microbench must report
// achieved DRAM bytes/cycle, not just wall time, or the choice is unfalsifiable.

// ------------------------------------------------------------------- helpers
// --- SMEM operand descriptor, K=32 atom, UNPACKED fp4 ------------------------
// [CHANGED vs the file-scope make_desc(): that one encodes LBO=8/SBO=16 u128
// because a kind::mxf4 atom consumes 16 PACKED bytes per row. An mxf8f6f4 K=32
// atom consumes 32 UNPACKED bytes per row, so the atom is 2 x 16-byte chunks and
// the strides double: LBO = 8 u128 = 128 B, SBO = 16 u128 = 256 B.]
__device__ __forceinline__ uint64_t tc5_make_desc(uint32_t smem_base) {
    const uint64_t start = (uint64_t)((smem_base >> 4) & 0x3FFFu);
    const uint64_t lbo = (uint64_t)((kLboBytes >> 4) & 0x3FFFu);
    const uint64_t sbo = (uint64_t)((kSboBytes >> 4) & 0x3FFFu);
    return start | (lbo << 16) | (sbo << 32) | ((uint64_t)1 << 46);  // version = 1
}

// --- instruction descriptor, block-scaled form --------------------------------
// Formats come from MXF8F6F4Format (E4M3 = 0, E2M1 = 5) — NOT MXF4Format::E2M1
// = 1. Everything else is the file-scope make_idesc() layout. Expected value for
// this geometry is 0x08820280 (Phase 0 prints it; use it as a probe assertion).
__device__ __forceinline__ uint32_t tc5_make_idesc(uint32_t a_sf_id, uint32_t b_sf_id) {
    uint32_t d = 0;
    d |= (b_sf_id & 0x3u) << 4;
    d |= 5u << 7;   // a_format = E2M1 (the fp4 weight)
    d |= 0u << 10;  // b_format = E4M3 (the activation)
    d |= (uint32_t)(kNTile >> 3) << 17;
    d |= 1u << 23;  // scale_format = UE8M0
    d |= (uint32_t)(kMTile >> 4) << 24;
    d |= (a_sf_id & 0x3u) << 29;
    d |= 0u << 31;  // k_size = 0 -> dense K32 for mxf8f6f4
    return d;
}

// --- the MMA under test (byte-identical to Phase 0's tc_mma_mxf8f6f4_1x) -----
__device__ __forceinline__ void tc5_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                        uint32_t idesc, uint32_t sfa_tmem, uint32_t sfb_tmem,
                                        uint32_t enable_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
        : "memory");
}

// --- TMEM access --------------------------------------------------------------
// x1 store: ONE column, all 32 lanes of the calling warp's own partition. Used
// by the SF prologue, which processes one 32-row group at a time (so it cannot
// use the x4 form Phase 0's single-shot probe used, where all four groups were
// resident at once). PTX .x1 exists alongside .x2/.x4/.x8.
__device__ __forceinline__ void tc5_st_x1(uint32_t taddr, uint32_t w0) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(taddr), "r"(w0)
                 : "memory");
}
__device__ __forceinline__ void tc5_ld_x8(uint32_t taddr, uint32_t* v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7])
        : "r"(taddr)
        : "memory");
}

// --- TMA (1D bulk) ------------------------------------------------------------
// 1D, NOT the tensor form: `cp.async.bulk.shared::cluster.global` takes
// [dstMem],[srcMem],size,[mbar] directly, so there is NO tensormap, no
// cuTensorMapEncode* plumbing, no host-side descriptor, and no commit_group —
// completion is counted by the mbarrier's tx-count. Both the `.shared::cluster`
// spelling used here and `.shared::cta` assemble on sm_103a; the cluster form
// with a CTA-relative address and no ctaMask targets this CTA's own smem (the
// same call dsv41_kernels.cu:569 makes, verified there in production).
// Requirements: 16-byte alignment on both sides, size % 16 == 0.
//
// [TODO-3] The performance follow-up is the 2D TENSOR form
// (`cp.async.bulk.tensor.2d`): ONE instruction per ring stage instead of 129,
// which is the difference between a real DRAM stream and a per-row issue storm.
// It needs a CUtensorMap built on the HOST (128 rows x kPackK/2 bytes, pitch =
// dim/2, SWIZZLE_NONE) — note that the tensormap must be built OUTSIDE graph
// capture (§5), and that the per-expert base address then has to be patched on
// device (`tensormap.replace.tile.global_address`) or one tensormap per expert
// kept in the pool. Keep the 1D path as the bring-up path: it has no host
// plumbing, so a layout bug cannot hide inside a tensormap.
__device__ __forceinline__ void tc5_bulk_g2s(void* smem_dst, const void* gmem_src, unsigned bytes,
                                             uint64_t* bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];" ::"r"((unsigned)__cvta_generic_to_shared(smem_dst)),
        "l"(gmem_src), "r"(bytes), "r"((unsigned)__cvta_generic_to_shared(bar))
        : "memory");
#else
    (void)smem_dst; (void)gmem_src; (void)bytes; (void)bar;
#endif
}

// Arms the byte count the bulk copies of one stage will retire, and performs the
// single arrival the barrier was initialised with (mbar_init(bar, 1)). MUST be
// issued before the copies it covers: the phase completes when arrivals == 1 AND
// the tx count reaches zero, so an arm after the copies could observe a phase
// that flipped on the previous use.
__device__ __forceinline__ void tc5_mbar_expect_tx(uint64_t* bar, unsigned bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     (unsigned)__cvta_generic_to_shared(bar)),
                 "r"(bytes)
                 : "memory");
#else
    (void)bar; (void)bytes;
#endif
}

// --- packed fp4 -> unpacked e2m1 (1 element per byte) -------------------------
// The checkpoint packs elements 2i (low nibble) and 2i+1 (high nibble) into
// byte i (dsv41_experts_mxf4.cu's AQ path does `lo | (hi << 4)`; quant.rs is the
// same order). Expansion is therefore: mask the low nibbles of the four packed
// bytes and the high nibbles separately, then byte-interleave them back.
//
//   p = [b0 b1 b2 b3], bi = e(2i) | e(2i+1) << 4
//   ev = p & 0x0F0F0F0F  = [e0 e2 e4 e6]     (low nibbles)
//   od = (p >> 4) & mask = [e1 e3 e5 e7]     (high nibbles)
//   out bytes [e0 e1 e2 e3] = __byte_perm(ev, od, 0x5140)
//   out bytes [e4 e5 e6 e7] = __byte_perm(ev, od, 0x7362)
// The 16-byte (uint4) load therefore becomes 8 u32 halves -> 32 unpacked bytes =
// two uint4, which is exactly one K=32 atom chunk pair for one A row.
__device__ __forceinline__ void tc5_expand(uint32_t p, uint32_t& ev, uint32_t& od) {
    ev = p & 0x0F0F0F0Fu;
    od = (p >> 4) & 0x0F0F0F0Fu;
}
__device__ __forceinline__ uint32_t tc5_ilv_lo(uint32_t ev, uint32_t od) {
    return __byte_perm(ev, od, 0x5140u);  // [e0 e1 e2 e3]
}
__device__ __forceinline__ uint32_t tc5_ilv_hi(uint32_t ev, uint32_t od) {
    return __byte_perm(ev, od, 0x7362u);  // [e4 e5 e6 e7]
}

// -----------------------------------------------------------------------------
// [TODO-1] raw packed A -> the unpacked canonical operand.
// One (row, K_STEP-within-slot) pair per thread per iteration: kMTile*kNStep =
// 256 items over 128 threads = 2 each. Verify against ph0_parity_kernel:1249-1257
// (the same unit16 map, but there the data was already unpacked).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void tc5_unpack_a(Smem& s, int slot) {
    for (int i = threadIdx.x; i < kMTile * kNStep; i += kThreads) {
        const int m = i / kNStep;   // weight row within the CTA's 128-row tile
        const int st = i % kNStep;  // K_STEP index within the slot
        const uint4 p =
            *reinterpret_cast<const uint4*>(s.a_raw[slot] + (size_t)m * (kPackK / 2) + st * 16);
        uint32_t ev[4], od[4];
        tc5_expand(p.x, ev[0], od[0]);
        tc5_expand(p.y, ev[1], od[1]);
        tc5_expand(p.z, ev[2], od[2]);
        tc5_expand(p.w, ev[3], od[3]);
        uint4 o0, o1;  // elements 0..15 and 16..31 of this K_STEP
        o0.x = tc5_ilv_lo(ev[0], od[0]);
        o0.y = tc5_ilv_hi(ev[0], od[0]);
        o0.z = tc5_ilv_lo(ev[1], od[1]);
        o0.w = tc5_ilv_hi(ev[1], od[1]);
        o1.x = tc5_ilv_lo(ev[2], od[2]);
        o1.y = tc5_ilv_hi(ev[2], od[2]);
        o1.z = tc5_ilv_lo(ev[3], od[3]);
        o1.w = tc5_ilv_hi(ev[3], od[3]);
        // kb = 0 (elements [0,16)) and kb = 1 (elements [16,32)) of the atom.
        uint8_t* dst = s.a_op[slot][st];
        *reinterpret_cast<uint4*>(dst + ((size_t)((m & 7) + 16 * (m >> 3))) * 16) = o0;
        *reinterpret_cast<uint4*>(dst + ((size_t)((m & 7) + 8 + 16 * (m >> 3))) * 16) = o1;
    }
}

// -----------------------------------------------------------------------------
// [TODO-2] raw e4m3 B -> the canonical operand.
// B has N=8 rows = ONE 8-row group, so unit16(n, kb) = (n & 7) + 8*kb: the row's
// first 16 bytes go to unit n, the second 16 to unit 8+n (KB != LBO here only
// because there is a single row group; the descriptor still carries SBO, which
// the hardware ignores at N=8 — Phase 0 says so and the probe relies on it).
// 32 uint4 moves per slot; trivial, but it must happen for EVERY stage because
// the raw slot is overwritten by the next TMA.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void tc5_permute_b(Smem& s, int slot) {
    for (int i = threadIdx.x; i < kNTile * kNStep * 2; i += kThreads) {
        const int st = i / (kNTile * 2);
        const int r = i % (kNTile * 2);
        const int n = r >> 1, kb = r & 1;
        const uint4 v = *reinterpret_cast<const uint4*>(s.b_raw[slot] + (size_t)n * kPackK +
                                                        st * kKStep + kb * 16);
        uint8_t* dst = s.b_op[slot][st];
        *reinterpret_cast<uint4*>(dst + ((size_t)(n + 8 * kb)) * 16) = v;
    }
}

// -----------------------------------------------------------------------------
// The kernel.
//
// Contract (checked by the launcher, not here):
//   dim % kPackK == 0   (the ring never stages a partial K group)
//   dim % 128 == 0      (the PACKED SF word never straddles the row end)
//   dim <= kMaxDim      (sizes the SF staging chunk)
//   (2*inter) % kMTile == 0   -> 30 tiles at the production shape
//   w_scale rows are per-(row, k/32) e8m0; act_scale is per-(k/32) e8m0.
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) expert_tcgen05_gateup_kernel(
    const uint8_t* __restrict__ w,          // [2*inter, dim/2] fp4, 2 elem/byte
    const uint8_t* __restrict__ w_scale,    // [2*inter, dim/32] e8m0
    const uint8_t* __restrict__ act,        // [dim] e4m3, the ONE decode token
    const uint8_t* __restrict__ act_scale,  // [dim/32] e8m0
    float* __restrict__ out,                // [slots][2*inter] (out_slot_stride apart)
    long out_slot_stride,
    int dim, int epi_mode, float limit, int split,
    // Indirect (graph-friendly) pool addressing, same convention as
    // expert_gemv_fp4_batched_kernel: the launcher stays routing-independent and
    // the per-expert base is derived in-kernel from the router's ids[slot].
    // ids == nullptr keeps the direct pointers. [Phase 2 wires this]
    const uint8_t* __restrict__ w_base, long w_stride,
    const uint8_t* __restrict__ ws_base, long ws_stride,
    const int* __restrict__ ids) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot = (int)blockIdx.y;

    // ---- expert / row-block resolution (pure argument arithmetic) -------------
    const uint8_t* wp = w;
    const uint8_t* wsp = w_scale;
    if (ids != nullptr) {
        const size_t e = (size_t)ids[slot];
        wp = w_base + e * (size_t)w_stride;
        wsp = ws_base + e * (size_t)ws_stride;
    }
    const int m0 = (int)blockIdx.x * kMTile;  // first weight row of this CTA
    const int kbytes = dim >> 1;              // packed weight bytes per row
    const int nsf = dim >> 5;                 // e8m0 bytes per weight row
    const int quads = nsf >> 2;               // PACKED SF words per row
    const int nk = dim / kKStep;              // MMAs over the whole K = 160
    const int ngrp = nk / kNStep;             // ring iterations = 80
    float* out_s = out + (size_t)slot * (size_t)out_slot_stride;

    __shared__ Smem s;

    // ---- TMEM + barrier init --------------------------------------------------
    if (warp == 0) {
        tc_alloc(&s.tmem_base, kTmemCols);
        tc_relinquish();
    }
    if (tid < kRing) {
        mbar_init(&s.tma_bar[tid], 1u);  // 1 arrival (expect_tx) + the tx bytes
        mbar_init(&s.mma_bar[tid], 1u);  // 1 arrival (tcgen05.commit)
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();

    const uint32_t tb = s.tmem_base;
    const uint32_t d_col = tb;                        // kNTile columns
    const uint32_t sfa_col = tb + kNTile;             // kSfaCols columns
    const uint32_t sfb_col = sfa_col + kSfaCols;      // kSfbCols columns
    // 8 + 160 + 40 = 208 <= kTmemCols = 256.

    // =========================================================================
    // 1. STAGING PROLOGUE — arm + issue the first kRing-1 ring slots.
    // =========================================================================
    // FIRST, deliberately: the ring TMA is then in flight for the whole of the SF
    // prologue below, so the ~20 KB of scale traffic does not sit in front of the
    // first MMA with an idle memory system behind it.
    //
    // Issue one ring stage. All kThreads take part: 128 A rows (threads 0..127
    // each copy kPackK/2 = 32 B = exactly one 32-byte DRAM sector, so a per-row
    // copy wastes nothing) + 1 activation row (threads 0..7 carry one B row each;
    // row 0 is the token and is the only one transferred).
    //
    // NOTE the B row choice: rows 1..7 of b_raw are zeroed ONCE (below) and are
    // never re-written by a TMA, because the operand's B rows 1..7 are
    // mathematically dead (their D columns are 0 by construction). This removes
    // the need for an [8, dim] padded activation buffer upstream and saves 7/8 of
    // the activation traffic. If that ever stops holding (e.g. a second token),
    // rows 1..7 must be TMA'd like row 0.
    auto issue = [&](int g) {
        const int sslot = g % kRing;
        const size_t k0 = (size_t)g * kPackK;
        if (tid < kMTile)
            tc5_bulk_g2s(s.a_raw[sslot] + (size_t)tid * (kPackK / 2),
                         wp + (size_t)(m0 + tid) * kbytes + (k0 >> 1), kPackK / 2,
                         &s.tma_bar[sslot]);
        if (tid < kNTile)
            tc5_bulk_g2s(s.b_raw[sslot] + (size_t)tid * kPackK, act + k0, kPackK,
                         &s.tma_bar[sslot]);
    };
    // Zero B rows 1..7, once: they are never TMA'd again.
    for (int i = tid; i < (kNTile - 1) * kPackK / 16; i += kThreads) {
        const int n = 1 + i / (kPackK / 16), off = (i % (kPackK / 16)) * 16;
        for (int r = 0; r < kRing; ++r)
            *reinterpret_cast<uint4*>(s.b_raw[r] + (size_t)n * kPackK + off) =
                make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();  // the tx count must be armed, and the zeroed B rows in smem,
                      // before any copy is issued / any raw slot is permuted
    if (tid == 0) {
        for (int g = 0; g < kRing - 1; ++g)
            if (g < ngrp) tc5_mbar_expect_tx(&s.tma_bar[g], kStageTxBytes);
    }
    __syncthreads();  // arm-before-issue
    for (int g = 0; g < kRing - 1; ++g)
        if (g < ngrp) issue(g);

    // =========================================================================
    // 2. SF PROLOGUE — the WHOLE scale block into TMEM, once, before the MMAs.
    // =========================================================================
    // Every MMA of the K loop reads all four row-group columns of its quad, so
    // the full 160 SFA columns must exist before the first MMA: the scales cannot
    // be staged lazily per ring slot.
    //
    // WHY NOT per-stage staging (the obvious alternative): the A scale for one
    // K-quad of one row is 4 CONTIGUOUS bytes, but consecutive rows are nsf=160
    // bytes apart, so a per-quad gather is a 4-byte strided load = 1 sector per
    // 4 bytes = 32x DRAM overfetch on a quantity that is already 1/32 of the
    // weight bytes. Instead: the block for 128 CONSECUTIVE rows is
    // 128 * 160 = 20480 contiguous bytes, so it is copied one 32-row group at a
    // time (coalesced uint4s) into sf_stage, and the four-blocks-per-word
    // assembly happens out of smem.
    // =========================================================================
    for (int j = 0; j < 4; ++j) {
        const uint8_t* src = wsp + (size_t)(m0 + 32 * j) * nsf;
        // coalesced: 32*nsf bytes = 2*nsf uint4 (nsf % 4 == 0 by contract).
        // The SOURCE row base is a TP-sharded `DevBuf::view` of the checkpoint
        // (`wsp = w_scale + e*w_scale_stride`), and a shard boundary can leave one
        // rank's scale view a few bytes off while every other rank is fine — the
        // same premise ld_uint2_a8 documents. A `uint4` read needs 16-byte
        // alignment, so it goes through the alignment-safe helper; the bytes are
        // IDENTICAL on both paths (see ld_uint4_a16).
        for (int i = tid; i < 2 * nsf; i += kThreads)
            reinterpret_cast<uint4*>(s.sf_stage)[i] = ld_uint4_a16(src + (size_t)i * 16);
        __syncthreads();  // the chunk is read by every warp below
        // PACKED word for row (32j + lane), quad q, column (sfa_col + 4q + j).
        // Every warp writes its OWN 32-lane partition with identical content:
        // PTX requires the scale factors to be duplicated to all four partitions,
        // and it is what Phase 0 validated.
        for (int q = 0; q < quads; ++q) {
            const uint32_t word =
                *reinterpret_cast<const uint32_t*>(s.sf_stage + (size_t)lane * nsf + 4 * q);
            tc5_st_x1(((uint32_t)(warp * 32) << 16) | (sfa_col + 4 * q + j), word);
        }
        __syncthreads();  // sf_stage is reused by the next row group
    }
    // B scales: one word per quad, replication across the warps as above. Only
    // lane 0 (activation row 0 = the token) carries a value; lanes 1..7 keep 0,
    // which decodes to 2^-127 — FINITE, so the zeroed B rows it scales stay
    // exactly 0. Never write 0xFF (NaN): NaN * 0 is NaN and would poison the
    // whole D column.
    for (int q = 0; q < quads; ++q) {
        const uint32_t word =
            (lane == 0) ? *reinterpret_cast<const uint32_t*>(act_scale + 4 * q) : 0u;
        tc5_st_x1(((uint32_t)(warp * 32) << 16) | (sfb_col + q), word);
    }
    tc_wait_st();
    tc_fence_before_thread_sync();
    __syncthreads();
    tc_fence_after_thread_sync();

    // =========================================================================
    // 3. THE RING
    // =========================================================================
    // Per iteration: one slot's raw bytes land (TMA completion), get unpacked
    // into the operand layout, and are consumed by kNStep MMAs. The barrier armed
    // for the NEXT refill is armed early (step 2 below) and the copies are issued
    // last (step 6), so the arm/copy order is preserved by the __syncthreads in
    // step 4 — no extra barrier in the loop.
    //
    // Phase bookkeeping — slot s is used by iterations g ≡ s (mod kRing); the
    // k-th use has parity k & 1.
    // =========================================================================
    for (int g = 0; g < ngrp; ++g) {
        const int sslot = g % kRing;
        const uint32_t ph = (uint32_t)((g / kRing) & 1);

        // 1. this slot's raw bytes have landed. No __syncthreads: the mbarrier
        //    wait has acquire semantics, so whoever observes the phase flip also
        //    observes the TMA writes.
        mbar_wait(&s.tma_bar[sslot], ph);

        // 2. arm the refill barrier of the slot we will fill at the end of this
        //    iteration. The arm itself touches no memory, so arming it here (while
        //    the slot may still be read by a retired-but-unwaited MMA) is safe.
        const int g_next = g + kRing - 1;
        const int slot_next = g_next % kRing;
        if (tid == 0 && g_next < ngrp)
            tc5_mbar_expect_tx(&s.tma_bar[slot_next], kStageTxBytes);

        // 3. expand the raw bytes into the MMA operands. [TODO-1 / TODO-2]
        tc5_unpack_a(s, sslot);
        tc5_permute_b(s, sslot);

        // 4. publish the operands to the async proxy (the MMA reads smem through
        //    it) and order step 2's arm before step 6's copies.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        // 5. the MMAs of this slot. One thread issues: tcgen05.mma is a
        //    CTA-level async instruction, not a per-thread workload. enable_d = 0
        //    only on the very first block of the very first ring iteration.
        if (tid == 0) {
#pragma unroll
            for (int st = 0; st < kNStep; ++st) {
                const int b = g * kNStep + st;  // global 32-element block index
                const uint64_t da = tc5_make_desc(smem_addr(s.a_op[sslot][st]));
                const uint64_t db = tc5_make_desc(smem_addr(s.b_op[sslot][st]));
                const uint32_t id = tc5_make_idesc((uint32_t)(b & 3), (uint32_t)(b & 3));
                tc5_mma(d_col, da, db, id, sfa_col + 4u * (uint32_t)(b >> 2),
                        sfb_col + (uint32_t)(b >> 2), b == 0 ? 0u : 1u);
            }
            tc_commit(&s.mma_bar[sslot]);
        }

        // 6. refill the slot we are about to overwrite. Its previous use was
        //    iteration g-1 (slot_next % kRing == g-1 for kRing >= 2), whose
        //    commit was issued at that iteration, so wait that use's parity. The
        //    wait covers BOTH buffers of the slot: the MMA reads the operand, and
        //    the unpack reads the raw, and the unpack of the previous use
        //    happened before its MMA was issued.
        if (g_next < ngrp) {
            if (g >= 1)
                mbar_wait(&s.mma_bar[slot_next], (uint32_t)(((g - 1) / kRing) & 1));
            issue(g_next);
        }
    }

    // =========================================================================
    // 4. EPILOGUE — D -> out. Wait for the LAST MMA first (the only barrier the
    // loop does not consume: each slot's commit is waited one revolution later,
    // except the final slot's).
    // =========================================================================
    mbar_wait(&s.mma_bar[(ngrp - 1) % kRing], (uint32_t)(((ngrp - 1) / kRing) & 1));
    {
        // D[m][n] lives at lane (m % 32) of partition (m / 32), column d_col + n
        // (Phase 0's read-back, proven). Only n = 0 carries the token, so only
        // v[0] is kept; v[1..7] are the dead columns (B rows 1..7 are zero).
        uint32_t v[kNTile];
        tc5_ld_x8(((uint32_t)(warp * 32) << 16) | d_col, v);
        tc_wait_ld();
        const int row = m0 + warp * 32 + lane;  // == the output column (swapAB)
        float x = __uint_as_float(v[0]);
        if (epi_mode == 1) {  // gate/up clamp, same convention as mxf4_gemm_kernel
            if (limit > 0.f) {
                if (split < 0) {
                    // interleaved (ILV) pool: even row = gate, odd row = up
                    x = (row & 1) ? fminf(fmaxf(x, -limit), limit) : fminf(x, limit);
                } else {
                    // row < split: gate (upper clamp only); else: up (both)
                    x = (row < split) ? fminf(x, limit) : fminf(fmaxf(x, -limit), limit);
                }
            }
        }
        // Phase 2 (down direction) adds row_weight[slot][row] here and the
        // epi_mode 3 accumulate, per mxf4_gemm_kernel:534-548.
        out_s[row] = x;
    }

    __syncthreads();
    if (warp == 0) tc_dealloc(tb, kTmemCols);
}

// =============================================================================
// LAUNCHER
// =============================================================================
// [Phase 2 territory] This is the Phase 1 shape: direct pointers, one launch,
// default OFF behind an env gate read ONCE per process. A per-call getenv() is a
// capture hazard (§5): the value must not change between capture and replay, and
// the Rust side flips it with a process-level env var anyway, so a function-local
// static is the correct form.
//
// [K-SPLIT TODO — read the OCCUPANCY note in the header before dismissing this]
// grid = (30, slots) is 240 CTAs at slots=8 with at most 2 CTAs/SM, i.e. it
// cannot put enough bytes in flight to reach the DRAM floor on its own. If the
// isolated number says so, add a third grid dimension that splits the K range
// and a deterministic ascending-order reduce (the plan §3 pins the same
// "fp addition is not associative, the order IS the contract" rule for the down
// direction's ascending-slot fixed-point reduce; gate/up needs the identical
// treatment). The kernel needs NO ring change for that — only the k0 range and
// the partial write — which is why the loop above is written against gngrp/nk
// rather than against dim.
// =============================================================================
inline cudaError_t tc5_launch_gateup(const uint8_t* w, const uint8_t* w_scale,
                                     const uint8_t* act, const uint8_t* act_scale, float* out,
                                     long out_slot_stride, int rows, int dim, int slots,
                                     float limit, int epi_mode, int split, cudaStream_t stream) {
    if (dim <= 0 || rows <= 0 || slots <= 0) return cudaErrorInvalidValue;
    if (dim % kPackK != 0 || dim % 128 != 0 || dim > kMaxDim) return cudaErrorInvalidValue;
    if (rows % kMTile != 0) return cudaErrorInvalidValue;
    const dim3 grid((unsigned)(rows / kMTile), (unsigned)slots, 1u);
    return dsv41_experts_pdl_or_plain(expert_tcgen05_gateup_kernel, grid, dim3(kThreads), 0,
                                      stream, w, w_scale, act, act_scale, out, out_slot_stride, dim,
                                      epi_mode, limit, split, (const uint8_t*)nullptr, 0L,
                                      (const uint8_t*)nullptr, 0L, (const int*)nullptr);
}

// gate/up, one dispatch per (layer, top-k slot) batch: `act` is the ONE shared
// quantised activation row, `out` holds `slots` consecutive [2*inter] blocks.
// Returns 0 (and does nothing) while disabled, so the caller can call it
// unconditionally and keep the old GEMV path as the fallback.
extern "C" int dsv41_expert_tcgen05_gate_up(const uint8_t* w, const uint8_t* w_scale,
                                            const uint8_t* act, const uint8_t* act_scale,
                                            float* out, long out_slot_stride, int inter, int dim,
                                            float limit, int slots, cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("DSV41_EXPERT_TCGEN05");
        return (e != nullptr && e[0] == '1') ? 1 : 0;  // default OFF until serve A/B
    }();
    if (!enabled) return 0;
    if (inter <= 0 || dim <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
    const cudaError_t e = tc5_launch_gateup(w, w_scale, act, act_scale, out, out_slot_stride,
                                            2 * inter, dim, slots, limit, /*epi_mode=*/1,
                                            /*split=*/inter, stream);
    (void)cudaGetLastError();  // never fail the step: the fallback GEMV is correctness
    return (int)e;
}

}  // namespace tc5

#endif  // DSV41_TCGEN05_GATEUP_SKELETON

// =============================================================================
// PHASE 1 (mxf4 revision) — tcgen05 swapAB gate/up, e2m1 x e2m1, scale_vec::2X
// =============================================================================
// STATUS: SKELETON, NOT IN THE BUILD, same gate style as the mxf8f6f4 block
// above (build.sh defines neither macro; each block compiles only under its own
// -D flag, so enabling one cannot affect the .so or the other block).
//
// Compile-verify (no GPU needed):
//     nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//          -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 -c kernels/cuda/dsv41_experts_mxf4.cu \
//          -o /tmp/t5m4.o -Xptxas -v
//     ptxas -v must show 0 spills and the
//     `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X` spelling.
//
// -----------------------------------------------------------------------------
// WHY A SECOND BLOCK INSTEAD OF A MACRO SWITCH OVER THE FIRST ONE
// -----------------------------------------------------------------------------
// docs/agent/expert-tcgen05-plan.md §1e (2026-09-12 revision) moved the Phase 1
// default from mxf8f6f4/1X/e4m3 to kind::mxf4/2X/e2m1. The two arms differ in
// FOUR places, and three of them are not textual substitutions:
//   * the operand FORMAT (packed 2/byte vs UNPACKED 1/byte) changes the ring
//     SIZING and deletes the 1:2 expansion pass;
//   * the MMA spelling and both descriptor formats change;
//   * the SF word is shared by a PAIR of atoms (2X) instead of a QUAD (1X),
//     which changes SFA_ID from `b & 3` to `2 * (a & 1)`.
// Keeping them in one block behind an #if would leave both arms' numbers
// interleaved in every comment; the arm that ships is the one whose comments and
// constants match its code. Same namespace FAMILY (tc5) as the block above, but a
// nested `mxf4` scope so that -D'ing both macros at once still compiles (the two
// arms' constant sets have the same names by design).
//
// -----------------------------------------------------------------------------
// WHAT CHANGES vs the mxf8f6f4 block (and what deliberately does not)
// -----------------------------------------------------------------------------
//  CHANGED
//    activation   e4m3 [dim]      -> e2m1 packed [dim/2]
//    MMA          kind::mxf8f6f4.block_scale.scale_vec::1X
//                                 -> kind::mxf4.block_scale.scale_vec::2X
//    K atom       32 (1X)         -> 64 (2X: one SF word per atom PAIR)
//    idesc        a_format=5 (MXF8F6F4Format::E2M1), b_format=0 (E4M3)
//                                 -> a_format=1, b_format=1 (MXF4Format::E2M1)
//    operands     UNPACKED 1/byte with a raw staging buffer + 1:2 expansion
//                                 -> PACKED 2/byte, no raw buffer, no expansion
//    ring         3 slots         -> 8 slots
//    smem/slot    13312 B         -> 4352 B
//  UNCHANGED (and this is the point of the exercise -- the SF addressing, the
//  TMEM budget, the mbarrier ring, the epilogue and the launcher contract are
//  all the same, so the serve A/B isolates ONE variable: the staging depth)
//    SF column density  4 SFA columns per 128 K elements in BOTH arms: a 32-bit
//                       SF word always holds 4 block scales, and a row group
//                       always owns one column. 1X spends the word on 4 single
//                       atoms via SFA_ID = 0..3; 2X spends it on 2 atom PAIRS
//                       via SFA_ID = 0/2. => SFA columns = 4 * (dim/128) = 160,
//                       SFB columns = dim/128 = 40, D = 8, total 208 <= 256.
//    TMEM alloc         256 columns (8 + 160 + 40), so still <= 2 CTA/SM.
//    D/TMEM read-back   D[m][n] at lane (m%32), partition (m/32), column d_col+n.
//    epilogue           gate/up clamp convention (mxf4_gemm_kernel:534-538).
//    grid/occupancy     (rows/128, slots); see the OCCUPANCY note in the block
//                       above -- it applies verbatim here.
//
// -----------------------------------------------------------------------------
// WHERE THE 3.06x COMES FROM (this is why there is no raw staging buffer)
// -----------------------------------------------------------------------------
// A packed fp4 operand carries 32 bytes per row per K=64 atom; the canonical
// UMMA core chunk is 16 bytes. So the raw checkpoint chunk size (16 B) and the
// canonical unit (16 B) coincide, and the layout transform is EXACTLY
//     "issue each 16-byte raw chunk to its canonical unit address".
// The mxf8f6f4 arm could not do that: its K=32 atom needs 32 UNPACKED bytes per
// row, i.e. each 16-byte raw chunk has to be expanded 1:2 into a 32-byte operand
// (a `__byte_perm` pass that a copy engine cannot perform), which forced a
// separate raw staging buffer per slot.
// Result: the permute segment is retained (m4_off() below IS it), the expansion
// segment is gone, and the per-slot operand smem drops
//     13312 B (a_raw 4096 + a_op 8192 + b_raw 512 + b_op 512, kPackK=64)
//   -> 4352 B (a_op 4096 + b_op 256)                            = 3.06x
// which is what buys kRing 3 -> 8 inside the same 48 KiB static window.
// ⚠️ The smem saving does NOT raise occupancy: the binding resource is the 256
// TMEM columns per CTA (plan §1d). The whole point is the deeper ring.
//
// -----------------------------------------------------------------------------
// THE ONE THING TO GET WRONG: A 16-BYTE-KERNEL TMA IS NOW THE STAGING
// -----------------------------------------------------------------------------
// Because there is no raw buffer, the TMA issues one 16-byte bulk copy per
// (row, K-chunk) instead of one 32-byte copy per row: 2*128 + 2 = 258 copies per
// ring stage (vs 129 in the mxf8f6f4 block). That is the price of kRing=8 and it
// is a bring-up design -- [TODO-3] below replaces it with the 2D TENSOR form,
// which copies an 8-row x 16-byte box into one 128-byte canonical core matrix
// per instruction (32 instructions per stage instead of 258).
// ⚠️ 16-byte alignment is now a hard contract on BOTH sides of every copy:
//   weights  wp + row*(dim/2) + k0/2 + 16*kb   (dim % 128 == 0 => dim/2 % 64 == 0)
//   act      act + k0/2 + 16*kb                (act must be 16-byte aligned)
// A misaligned bulk copy does not fault -- it silently misplaces bytes.
//
// -----------------------------------------------------------------------------
// NUMERIC CONTRACT: the 2X SF PAIRING IS FORCED BY THE CHECKPOINT LAYOUT
// -----------------------------------------------------------------------------
// The checkpoint stores one e8m0 scale per (row, 32 K elements). A 2X MMA
// consumes 64 K elements per atom, so two consecutive checkpoint scale bytes are
// consumed by one atom, and one 32-bit SF word holds the FOUR bytes of an atom
// pair: bytes [0,1] for the even atom (SFA_ID = 0), bytes [2,3] for the odd atom
// (SFA_ID = 2). The PTX ISA word layout is [SF0, SF1, SF0, SF1]. This is exactly
// what mxf4_gemm_kernel:430-514 already does, and what
// tests_tcgen05_mxf4.cu validates on the GPU (maxdiff == 0).
// => `dim % 128 == 0` is a correctness contract, not a tuning choice: a pair
//    spans 4 blocks = 128 K elements, so dim % 128 != 0 would let a 32-bit SF
//    word straddle the end of the row (the launcher rejects it).
//
// The golden reference for every layout constant below is
// dsv41_experts_mxf4.cu:24-70 (the file header) + :249-273 (desc/idesc) and
// tests_tcgen05_mxf4.cu (the only GPU-verified fp4 tcgen05 form we have).
// =============================================================================

#ifdef DSV41_TCGEN05_GATEUP_MXF4_SKELETON

#include <cstdlib>  // getenv (launcher gate)

namespace tc5 {
namespace mxf4 {

// ------------------------------------------------------------------ geometry
constexpr int kMTile = 128;  // MMA M, pinned by the instruction
constexpr int kNTile = 8;    // minimum legal N (mxf4 N range [8,256] step 8,
                             // dsv41_experts_mxf4.cu:17-18); only col 0 is kept
constexpr int kKStep = 64;   // one 2X atom == one MMA (dense K64, k_size = 0)
constexpr int kAtomBytes = kKStep / 2;   // 32 PACKED fp4 bytes per row per atom
constexpr int kPackK = 64;   // K elements staged per ring slot (== kKStep)
constexpr int kNStep = kPackK / kKStep;  // MMAs issued per slot (== 1)
constexpr int kRing = 8;     // ring depth -- 8 slots is the whole point of the
                             // mxf4 arm. See the SMEM BUDGET note below.
constexpr int kThreads = 128;  // 4 warps == the 4 TMEM lane partitions
// Descriptor strides, in bytes. A K=64 atom is 32 PACKED bytes per row = TWO
// 16-byte core chunks, so the canonical interleave is
//     unit16(m, kb) = (m % 8) + 8*kb + 16*(m / 8)          kb in {0,1}
// => LBO (K-chunk stride) = 8 units = 128 B, SBO (8-row-group stride) =
// 16 units = 256 B. These are the SAME numbers as the file-scope make_desc()
// (:249-254), which is why the production mxf4_gemm_kernel needs no desc change:
// the packed K=64 atom and the unpacked mxf8f6f4 K=32 atom both carry 32 bytes
// per row, so the two arms' descriptors coincide numerically even though the
// formats differ.
constexpr int kLboBytes = 128;
constexpr int kSboBytes = 256;
constexpr int kTmemCols = 256;  // power of two >= 8 + 160 + 40 = 208
constexpr int kMaxDim = 5120;   // gate/up dim; sizes the SF staging chunk
constexpr int kSfWords = kMaxDim / 32 / 4;  // 4 block scales per 32-bit word = 40
constexpr int kSfaCols = 4 * kSfWords;      // 160 (one column per (pair, row-group))
constexpr int kSfbCols = kSfWords;          // 40  (one column per pair, N=8 = 1 group)
// SFA columns used per ring stage: 4 per atom PAIR (one per 32-row group of
// M=128) -> kNStep=1 atom = half a pair, but the pair's word is written whole
// (both atoms share it), so the SF prologue walks ALL kSfWords words once.
constexpr int kSfChunkRows = 32;  // one 32-row group staged at a time

// Per-ring-stage operand bytes. A = 128 rows x 32 packed bytes; B = 8 rows x 32
// packed bytes (only row 0 is live -- rows 1..7 are zeroed once and never TMA'd,
// their D columns being mathematically dead; see the prologue).
constexpr int kAbBytes = kMTile * kAtomBytes * kNStep;  // 4096 (A operand smem)
constexpr int kBbBytes = kNTile * kAtomBytes * kNStep;  //  256 (B operand smem)
// TMA transaction bytes for one ring stage: the whole A operand + ONE B row
// (rows 1..7 of the B operand are zeroed once and never transferred). Every copy
// is 16 B, so this is also 258 * 16 (see the header note).
constexpr unsigned kStageTxBytes =
    (unsigned)(kMTile * kAtomBytes * kNStep + kAtomBytes * kNStep);  // 4096 + 32 = 4128
// --- SF staging chunk: ONE 32-row group of A scales (32 * dim/32 contiguous
// bytes in the pool, because the CTA's rows are consecutive) + the whole B
// scale row. Same shape and same size as the mxf8f6f4 block: the scale format
// did not change. Staged one group at a time so the buffer stays small.
constexpr int kSfAChunkBytes = kSfChunkRows * (kMaxDim / 32);  // 32 * 160 = 5120
constexpr int kSfBChunkBytes = kNTile * (kMaxDim / 32);        //  8 * 160 = 1280

struct Smem {
    // ---- ring: the canonical operand ONLY (no raw staging, no expansion) ----
    // a_op is kNStep K=64 atoms of 128 weight rows; b_op the matching atom of
    // the 8 activation rows. Every byte here is written by a 16-byte TMA that
    // lands directly on its canonical unit (m4_off), so the MMA can read it the
    // moment the mbarrier tx-count retires.
    alignas(1024) uint8_t a_op[kRing][kNStep][kAbBytes];
    alignas(128) uint8_t b_op[kRing][kNStep][kBbBytes];
    alignas(1024) uint8_t sf_stage[kSfAChunkBytes > kSfBChunkBytes ? kSfAChunkBytes
                                                                  : kSfBChunkBytes];
    alignas(8) uint64_t tma_bar[kRing];  // TMA tx-count completion, one per slot
    alignas(8) uint64_t mma_bar[kRing];  // tcgen05.commit retirement, one per slot
    uint32_t tmem_base;
};

// The 48 KiB static window. Same rule as the block above: a tuning change that
// trips this fails the BUILD instead of failing at launch. Going deeper than
// kRing=8 needs dynamic smem + an INIT-TIME cudaFuncSetAttribute (capture rules,
// plan §5), which is a different risk class -- see the DEPTH note.
static_assert(sizeof(Smem) <= 48 * 1024, "tc5::mxf4 ring does not fit static smem");
// TMEM budget: the accumulator D plus the full SFA and SFB range must fit the
// single power-of-two alloc (8 + 160 + 40 = 208 <= 256).
static_assert(kNTile + kSfaCols + kSfbCols <= kTmemCols, "tc5::mxf4 TMEM budget overflow");
// sizeof at the shipped constants: a_op 32768 + b_op 2048 + sf_stage 5120 +
// barriers 128 + tmem_base ~= 40068 B -> 40 KiB of the 48 KiB window, leaving
// kRing 9-10 as headroom if a future kPackK stays at one atom.

// DEPTH NOTE. kRing is the only lever that matters for the DRAM floor:
//   in flight ~= (kRing - 1) * kMTile * kAtomBytes = 7 * 4096 = 28 KiB per CTA
// vs 8 KiB at the mxf8f6f4 block's kRing=3. At slots=8 (240 CTAs) that is
// 6.7 MB in flight, i.e. the 8 TB/s x ~700 ns bandwidth-latency product -- which
// is the entire reason this arm exists. DO NOT trade kRing back for kPackK:
// staging two atoms per slot doubles a_op and forces kRing back to 4.

// ------------------------------------------------------------------- helpers
// --- the permute segment, expressed as an ADDRESS -----------------------------
// Byte offset of the canonical 16-byte unit that holds (row, kb) of a K=64 atom,
//     unit16(row, kb) = (row & 7) + 8*kb + 16*(row >> 3).
// One helper serves BOTH operands: A uses all 16 row groups (SBO term = 256 B),
// B has a single 8-row group so the (row >> 3) term is identically 0 (which is
// also why the hardware ignores SBO at N=8 -- Phase 0 says so and relies on it).
__device__ __forceinline__ int m4_off(int row, int kb) {
    return 16 * ((row & 7) + 8 * kb + 16 * (row >> 3));
}

// --- SMEM operand descriptor, SWIZZLE_NONE K-major, 16-byte units ------------
// Identical in form and value to the file-scope make_desc() (:249-254) and to
// the mxf8f6f4 block's tc5_make_desc(). Kept local (not shared) so this arm's
// constants cannot drift through a macro switch.
__device__ __forceinline__ uint64_t m4_make_desc(uint32_t smem_base) {
    const uint64_t start = (uint64_t)((smem_base >> 4) & 0x3FFFu);
    const uint64_t lbo = (uint64_t)((kLboBytes >> 4) & 0x3FFFu);  // 8
    const uint64_t sbo = (uint64_t)((kSboBytes >> 4) & 0x3FFFu);  // 16
    return start | (lbo << 16) | (sbo << 32) | ((uint64_t)1 << 46);  // version = 1
}

// --- instruction descriptor, block-scaled, kind::mxf4 -------------------------
// a_format = b_format = 1 (MXF4Format::E2M1). ⚠️ NOT 5: 5 is
// MXF8F6F4Format::E2M1, the value the mxf8f6f4 block uses for its fp4 WEIGHT
// with b_format = 0 (E4M3) for the activation. Bit layout is the one the file
// header documents at :58-67 and make_idesc() encodes at :256-273.
// For this geometry (M=128, N=8, sf ids 0) the value is 0x08820480 -- the
// mxf8f6f4 block's probe assertion is 0x08820280, differing in exactly the two
// format fields (b_format 1<<10 = 0x400 and a_format 1<<7 = 0x80).
__device__ __forceinline__ uint32_t m4_make_idesc(uint32_t a_sf_id, uint32_t b_sf_id) {
    uint32_t d = 0;
    d |= (b_sf_id & 0x3u) << 4;
    d |= 1u << 7;   // a_format = E2M1 (MXF4Format)
    d |= 1u << 10;  // b_format = E2M1  <- the mxf8f6f4 arm had 0 (E4M3) here
    d |= (uint32_t)(kNTile >> 3) << 17;
    d |= 1u << 23;  // scale_format = UE8M0 (forced, == the checkpoint format)
    d |= (uint32_t)(kMTile >> 4) << 24;
    d |= (a_sf_id & 0x3u) << 29;
    d |= 0u << 31;  // k_size = 0 -> dense K64
    return d;
}

// --- the MMA (byte-identical to the production tc_mma_mxf4 at :212-222) ------
__device__ __forceinline__ void m4_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
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

// --- TMEM access (same spellings the mxf8f6f4 block verified) -----------------
__device__ __forceinline__ void m4_st_x1(uint32_t taddr, uint32_t w0) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(taddr), "r"(w0)
                 : "memory");
}
__device__ __forceinline__ void m4_ld_x8(uint32_t taddr, uint32_t* v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7])
        : "r"(taddr)
        : "memory");
}

// --- TMA (1D bulk), 16 bytes per copy ----------------------------------------
// Same instruction, same rationale and the same 2D-tensor [TODO-3] as the
// mxf8f6f4 block; only the SIZE changes (16 B chunks instead of one 32 B copy
// per row). Requirements: 16-byte alignment on both sides, size % 16 == 0.
__device__ __forceinline__ void m4_bulk_g2s(void* smem_dst, const void* gmem_src, unsigned bytes,
                                            uint64_t* bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];" ::"r"((unsigned)__cvta_generic_to_shared(smem_dst)),
        "l"(gmem_src), "r"(bytes), "r"((unsigned)__cvta_generic_to_shared(bar))
        : "memory");
#else
    (void)smem_dst; (void)gmem_src; (void)bytes; (void)bar;
#endif
}

// Arms the byte count one stage's copies will retire, and performs the single
// arrival the barrier was initialised with. MUST be issued before the copies it
// covers (phase completes on arrivals == 1 AND tx count == 0).
__device__ __forceinline__ void m4_mbar_expect_tx(uint64_t* bar, unsigned bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     (unsigned)__cvta_generic_to_shared(bar)),
                 "r"(bytes)
                 : "memory");
#else
    (void)bar; (void)bytes;
#endif
}

// -----------------------------------------------------------------------------
// The kernel.
//
// Contract (checked by the launcher, not here):
//   dim % kPackK == 0    (the ring never stages a partial K atom)
//   dim % 128 == 0       (2X pairing: a 32-bit SF word never straddles a row end)
//   dim <= kMaxDim       (sizes the SF staging chunk)
//   dim % 2 == 0 and act 16-byte aligned (16-byte bulk-copy contract)
//   rows == 2 * split and split % 32 == 0  (see GAP 1 below)
//   (2*inter) % kMTile == 0    -> 30 tiles at the production shape
//   w1/w3 (+ their scales) rows are per-(row, k/32); act_scale is per-(k/32)
//   f32 (GAP 2: the quantiser emits f32 2^n, the kernel converts to e8m0).
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) expert_tcgen05_gateup_mxf4_kernel(
    const uint8_t* __restrict__ act,        // [dim/2] fp4 e2m1, the ONE token
    const float* __restrict__ act_scale,    // [dim/32] f32 power-of-two scales
    float* __restrict__ out,                // [slots][2*inter] (out_slot_stride apart)
    long out_slot_stride,
    int dim, int epi_mode, float limit, int split,
    // ---- GAP 1: TWO weight pools, not one ------------------------------------
    // The loader keeps w1 (gate) and w3 (up) in SEPARATE contiguous planes
    // (`load.rs:644`: [w1][w1.scale][w3][w3.scale][w2][w2.scale]). `split` is
    // the row boundary between them, NOT the row count of either pool: output
    // row `r < split` is gate pool row `r`, output row `r >= split` is up pool
    // row `r - split`. This is exactly `mxf4_gemm_kernel:626-632` (`b`/`b_hi` +
    // `b_split`), the production-verified form of the same split. Feeding the
    // single-pool form `w1_base` alone reads `w1.scale` as gate weights for
    // every row >= split: silent wrong values, never a fault.
    // Scales are [rows, dim/32] e8m0 and follow the same row split.
    const uint8_t* __restrict__ w1_base, long w1_stride,
    const uint8_t* __restrict__ w1s_base, long w1s_stride,
    const uint8_t* __restrict__ w3_base, long w3_stride,
    const uint8_t* __restrict__ w3s_base, long w3s_stride,
    // ---- GAP 3: per-slot expert indirection ----------------------------------
    // ids != nullptr: each grid.y slot derives its OWN four pool pointers from
    // `base + ids[slot] * stride`, so the routing never reaches the host and
    // the launch arguments stay independent of the routing (CUDA-graph safe).
    // ids == nullptr: the four bases ARE the direct pointers (one expert, every
    // slot the same answer) -- the parity-harness form.
    const int* __restrict__ ids) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot = (int)blockIdx.y;

    // ---- expert / row-block resolution (pure argument arithmetic) -------------
    const uint8_t* w1p = w1_base;
    const uint8_t* w1sp = w1s_base;
    const uint8_t* w3p = w3_base;
    const uint8_t* w3sp = w3s_base;
    if (ids != nullptr) {
        const size_t e = (size_t)ids[slot];
        w1p = w1_base + e * (size_t)w1_stride;
        w1sp = w1s_base + e * (size_t)w1s_stride;
        w3p = w3_base + e * (size_t)w3_stride;
        w3sp = w3s_base + e * (size_t)w3s_stride;
    }
    const int m0 = (int)blockIdx.x * kMTile;  // first weight row of this CTA
    const int kbytes = dim >> 1;              // packed weight bytes per row
    const int nsf = dim >> 5;                 // e8m0 bytes per weight row
    const int nwords = nsf >> 2;              // 32-bit SF words per row == pairs
    const int natoms = dim / kKStep;          // K=64 atoms over the whole K
    const int ngrp = natoms / kNStep;         // ring iterations
    float* out_s = out + (size_t)slot * (size_t)out_slot_stride;

    __shared__ Smem s;

    // ---- TMEM + barrier init --------------------------------------------------
    if (warp == 0) {
        tc_alloc(&s.tmem_base, kTmemCols);
        tc_relinquish();
    }
    if (tid < kRing) {
        mbar_init(&s.tma_bar[tid], 1u);  // 1 arrival (expect_tx) + the tx bytes
        mbar_init(&s.mma_bar[tid], 1u);  // 1 arrival (tcgen05.commit)
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();

    const uint32_t tb = s.tmem_base;
    const uint32_t d_col = tb;                     // kNTile columns
    const uint32_t sfa_col = tb + kNTile;          // kSfaCols columns
    const uint32_t sfb_col = sfa_col + kSfaCols;   // kSfbCols columns
    // 8 + 160 + 40 = 208 <= kTmemCols = 256. SAME budget as the 1X arm: an
    // atom pair consumes one SF word per row group exactly like a quad does.

    // =========================================================================
    // 1. STAGING PROLOGUE — arm + issue the first kRing-1 ring slots.
    // =========================================================================
    // FIRST, deliberately: the ring TMA is then in flight for the whole of the
    // SF prologue below, so the ~6 KB of scale traffic does not sit in front of
    // the first MMA with an idle memory system behind it.
    //
    // One ring stage = one K=64 atom = 258 16-byte bulk copies, landed DIRECTLY
    // on their canonical units (that IS the permute segment; there is no raw
    // buffer to permute out of). Thread t owns A row t (2 chunks); the two B
    // chunks go to threads 0..1.
    auto issue = [&](int g) {
        const int sslot = g % kRing;
        const size_t kb0 = (size_t)g * (kKStep / 2);  // packed byte offset of the atom
        if (tid < kMTile) {
            // GAP 1: per ROW, not per CTA. `rows % 128 == 0` only pins the split
            // to a multiple of 64, so a 128-row tile may straddle the two pools
            // (the production shape does not, but the kernel must not depend on
            // that). Row `r < split` is gate pool row `r`; `r >= split` is up
            // pool row `r - split`.
            const int row = m0 + tid;
            const uint8_t* wrow = (row < split)
                                      ? (w1p + (size_t)row * kbytes)
                                      : (w3p + (size_t)(row - split) * kbytes);
            const uint8_t* src = wrow + kb0;
#pragma unroll
            for (int st = 0; st < kNStep; ++st) {
                uint8_t* dst = s.a_op[sslot][st];
#pragma unroll
                for (int kb = 0; kb < 2; ++kb)
                    m4_bulk_g2s(dst + m4_off(tid, kb), src + st * kAtomBytes + 16 * kb, 16,
                                &s.tma_bar[sslot]);
            }
        }
        // B row 0 is the token and is the only row transferred; rows 1..7 are
        // zeroed ONCE (below) and never re-written, because their D columns are
        // dead by construction. This is what lets the activation stay a single
        // [dim/2] packed row instead of an [8, dim/2] padded buffer.
        if (tid < 2 * kNStep) {
            const int st = tid >> 1, kb = tid & 1;
            m4_bulk_g2s(s.b_op[sslot][st] + m4_off(0, kb),
                        act + kb0 + st * kAtomBytes + 16 * kb, 16, &s.tma_bar[sslot]);
        }
    };
    // Zero B rows 1..7, once, for every (slot, atom): never TMA'd again.
    // kRing * kNStep * 7 * 2 = 112 uint4 stores over 128 threads -> one each.
    for (int i = tid; i < kRing * kNStep * (kNTile - 1) * 2; i += kThreads) {
        const int kb = i & 1;
        const int n = 1 + ((i >> 1) % (kNTile - 1));
        const int rest = (i >> 1) / (kNTile - 1);  // which (slot, atom)
        *reinterpret_cast<uint4*>(s.b_op[rest / kNStep][rest % kNStep] + m4_off(n, kb)) =
            make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();  // the tx count must be armed, and the zeroed B rows in smem,
                      // before any copy is issued / any slot is consumed
    if (tid == 0) {
        for (int g = 0; g < kRing - 1; ++g)
            if (g < ngrp) m4_mbar_expect_tx(&s.tma_bar[g], kStageTxBytes);
    }
    __syncthreads();  // arm-before-issue
    for (int g = 0; g < kRing - 1; ++g)
        if (g < ngrp) issue(g);

    // =========================================================================
    // 2. SF PROLOGUE — the WHOLE scale block into TMEM, once, before the MMAs.
    // =========================================================================
    // Byte-for-byte the same pass as the mxf8f6f4 block: the SF FORMAT and the
    // SF COLUMN DENSITY are both unchanged by the arm switch. Every MMA of the
    // K loop reads all four row-group columns of its word, so the full 160 SFA
    // columns must exist before the first MMA; the scales cannot be staged
    // lazily per ring slot.
    //
    // The A block for 128 CONSECUTIVE rows is 128 * 160 = 20480 contiguous bytes
    // in the pool, so it is copied one 32-row group at a time (coalesced uint4s)
    // into sf_stage, and the 4-blocks-per-word assembly happens out of smem.
    // =========================================================================
    for (int j = 0; j < 4; ++j) {
        // GAP 1: the scales follow the same row split as the weights. A 32-row
        // group never straddles the pools: `split` is a multiple of 32 (it is
        // `rows/2` with `rows % 128 == 0`) and m0 is a multiple of 128.
        const int r0 = m0 + 32 * j;
        const bool hi = (r0 >= split);
        const int rr = hi ? (r0 - split) : r0;
        const uint8_t* src = (hi ? w3sp : w1sp) + (size_t)rr * nsf;
        // `w3sp`/`w1sp` are the TP-sharded W3/W1 SCALE views (`w3s_base +
        // e*w3s_stride`), i.e. the same plane the grouped arm calls `bhs_base` —
        // a shard boundary can leave one rank's view off by a few bytes. The
        // uint4 source read therefore goes through the alignment-safe helper
        // (byte-identical on both paths; see ld_uint4_a16).
        for (int i = tid; i < 2 * nsf; i += kThreads)  // nsf % 4 == 0 by contract
            reinterpret_cast<uint4*>(s.sf_stage)[i] = ld_uint4_a16(src + (size_t)i * 16);
        __syncthreads();  // the chunk is read by every warp below
        // Word for row (32j + lane), word index w (4 blocks 4w..4w+3), column
        // (sfa_col + 4w + j). Every warp writes its OWN 32-lane partition with
        // identical content: PTX requires the scale factors duplicated to all
        // four partitions, and it is what both Phase 0 harnesses validate.
        for (int w = 0; w < nwords; ++w) {
            const uint32_t word =
                *reinterpret_cast<const uint32_t*>(s.sf_stage + (size_t)lane * nsf + 4 * w);
            m4_st_x1(((uint32_t)(warp * 32) << 16) | (sfa_col + 4 * w + j), word);
        }
        __syncthreads();  // sf_stage is reused by the next row group
    }
    // B scales: one word per pair, replicated across the warps as above. Only
    // lane 0 (activation row 0 = the token) carries a value; lanes 1..7 keep 0,
    // which decodes to 2^-127 -- FINITE, so the zeroed B rows it scales stay
    // exactly 0. Never write 0xFF (NaN): NaN * 0 is NaN and would poison the
    // whole D column.
    for (int w = 0; w < nwords; ++w) {
        // GAP 2: `dsv41_quant_fp4` emits f32 powers of two; the SF word wants
        // e8m0 BYTES. Converting here (instead of demanding an e8m0 activation
        // scale producer) leaves the quantiser's ABI and the whole old SIMT path
        // untouched. Lossless by construction: a fast_round_scale output IS 2^n,
        // so `f_pow2_to_ue8m0` (:132) is an exact exponent copy, and the four
        // bytes are packed exactly as the weight path packs its checkpoint bytes
        // ([b0, b1, b2, b3] of the atom pair's four 32-K blocks).
        uint32_t word = 0u;
        if (lane == 0) {
            const float* s4 = act_scale + 4 * w;
            word = (uint32_t)f_pow2_to_ue8m0(s4[0]) | ((uint32_t)f_pow2_to_ue8m0(s4[1]) << 8) |
                   ((uint32_t)f_pow2_to_ue8m0(s4[2]) << 16) |
                   ((uint32_t)f_pow2_to_ue8m0(s4[3]) << 24);
        }
        m4_st_x1(((uint32_t)(warp * 32) << 16) | (sfb_col + w), word);
    }
    tc_wait_st();
    tc_fence_before_thread_sync();
    __syncthreads();
    tc_fence_after_thread_sync();

    // =========================================================================
    // 3. THE RING
    // =========================================================================
    // Per iteration: one slot's atom lands (TMA completion), is consumed by its
    // MMAs, and the slot is refilled. The barrier armed for the NEXT refill is
    // armed early (step 2) and the copies are issued last (step 6), so the
    // arm/copy order is preserved by the __syncthreads in step 4 -- no extra
    // barrier in the loop.
    //
    // Phase bookkeeping -- slot s is used by iterations g ≡ s (mod kRing); the
    // k-th use has parity k & 1.
    // =========================================================================
#pragma unroll 1
    for (int g = 0; g < ngrp; ++g) {
        const int sslot = g % kRing;
        const uint32_t ph = (uint32_t)((g / kRing) & 1);

        // 1. this slot's bytes have landed. No __syncthreads: the mbarrier wait
        //    has acquire semantics, so whoever observes the phase flip also
        //    observes the TMA writes. There is no expansion/permute step between
        //    the TMA and the MMA in this arm -- that is the mxf4 win.
        mbar_wait(&s.tma_bar[sslot], ph);

        // 2. arm the refill barrier of the slot we will fill at the end of this
        //    iteration. The arm touches no memory, so arming it here (while the
        //    slot may still be read by a retired-but-unwaited MMA) is safe.
        const int g_next = g + kRing - 1;
        const int slot_next = g_next % kRing;
        if (tid == 0 && g_next < ngrp)
            m4_mbar_expect_tx(&s.tma_bar[slot_next], kStageTxBytes);

        // 3. publish the operands to the async proxy. In this arm the only
        //    generic-proxy writer of the operands is the one-time B zeroing
        //    above; the fence is kept per iteration anyway because it also costs
        //    nothing next to the MMA and it keeps this loop structurally
        //    identical to the (fence-required) mxf8f6f4 arm.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        // 4. the MMAs of this slot. One thread issues: tcgen05.mma is a CTA-level
        //    async instruction, not a per-thread workload. enable_d = 0 only on
        //    the very first atom.
        if (tid == 0) {
#pragma unroll
            for (int st = 0; st < kNStep; ++st) {
                const int a = g * kNStep + st;  // global K=64 atom index
                // 2X SF: a 32-bit word holds the 4 block scales of an atom PAIR,
                // bytes [0,1] = atom 2pr (SFA_ID 0), bytes [2,3] = atom 2pr+1
                // (SFA_ID 2). Both atoms of the pair therefore read the SAME
                // column and differ only in the 2-bit sub-column selector.
                const int pr = a >> 1;
                const uint32_t sf = (uint32_t)(2 * (a & 1));
                const uint64_t da = m4_make_desc(smem_addr(s.a_op[sslot][st]));
                const uint64_t db = m4_make_desc(smem_addr(s.b_op[sslot][st]));
                const uint32_t id = m4_make_idesc(sf, sf);
                m4_mma(d_col, da, db, id, sfa_col + 4u * (uint32_t)pr, sfb_col + (uint32_t)pr,
                       a == 0 ? 0u : 1u);
            }
            tc_commit(&s.mma_bar[sslot]);
        }

        // 5. refill the slot we are about to overwrite. Its previous use was
        //    iteration g-1 (slot_next == g-1 mod kRing for kRing >= 2), whose
        //    commit was issued at that iteration, so wait that use's parity.
        if (g_next < ngrp) {
            if (g >= 1)
                mbar_wait(&s.mma_bar[slot_next], (uint32_t)(((g - 1) / kRing) & 1));
            issue(g_next);
        }
    }

    // =========================================================================
    // 4. EPILOGUE — D -> out. Wait for the LAST MMA first (the only barrier the
    // loop does not consume: each slot's commit is waited one revolution later,
    // except the final slot's).
    // =========================================================================
    mbar_wait(&s.mma_bar[(ngrp - 1) % kRing], (uint32_t)(((ngrp - 1) / kRing) & 1));
    {
        // D[m][n] lives at lane (m % 32) of partition (m / 32), column d_col + n.
        // Only n = 0 carries the token, so only v[0] is kept; v[1..7] are the
        // dead columns (B rows 1..7 are zero).
        uint32_t v[kNTile];
        m4_ld_x8(((uint32_t)(warp * 32) << 16) | d_col, v);
        tc_wait_ld();
        const int row = m0 + warp * 32 + lane;  // == the output column (swapAB)
        float x = __uint_as_float(v[0]);
        if (epi_mode == 1) {  // gate/up clamp, same convention as mxf4_gemm_kernel
            if (limit > 0.f) {
                if (split < 0) {
                    // interleaved (ILV) pool: even row = gate, odd row = up
                    x = (row & 1) ? fminf(fmaxf(x, -limit), limit) : fminf(x, limit);
                } else {
                    // row < split: gate (upper clamp only); else: up (both)
                    x = (row < split) ? fminf(x, limit) : fminf(fmaxf(x, -limit), limit);
                }
            }
        }
        // Phase 2 (down direction) adds row_weight[slot][row] here and the
        // epi_mode 3 accumulate, per mxf4_gemm_kernel:534-548.
        out_s[row] = x;
    }

    __syncthreads();
    if (warp == 0) tc_dealloc(tb, kTmemCols);
}

// =============================================================================
// LAUNCHER
// =============================================================================
// [Phase 2 territory] Phase 1 shape: the four pool bases (+ per-expert strides)
// and the per-slot `ids`, one launch, default OFF behind an env gate read ONCE
// per process (a per-call getenv() is a capture hazard, plan §5 -- the value
// must not change between capture and replay, and the Rust side flips it with a
// process-level env var anyway).
//
// [K-SPLIT TODO] The OCCUPANCY note in the mxf8f6f4 block applies verbatim:
// grid = (30, slots) at slots=8 with <= 2 CTAs/SM still has to put enough bytes
// in flight, and that is exactly what kRing=8 now supplies (28 KiB/CTA). If
// slots=8 still lands far from the floor, add the third grid dimension that
// splits the K range plus an ascending-order reduce (fp addition is not
// associative; the order IS the contract). The kernel needs no ring change --
// only the atom range and the partial write.
// =============================================================================
inline cudaError_t m4_launch_gateup(const uint8_t* act, const float* act_scale, float* out,
                                    long out_slot_stride, int rows, int dim, int slots,
                                    float limit, int epi_mode, int split,
                                    const uint8_t* w1_base, long w1_stride,
                                    const uint8_t* w1s_base, long w1s_stride,
                                    const uint8_t* w3_base, long w3_stride,
                                    const uint8_t* w3s_base, long w3s_stride,
                                    const int* ids, cudaStream_t stream) {
    if (dim <= 0 || rows <= 0 || slots <= 0) return cudaErrorInvalidValue;
    // dim % 128 == 0 IS the 2X pairing contract (a 32-bit SF word spans 4 blocks
    // = 128 K elements); dim % kPackK == 0 keeps the ring on atom boundaries.
    if (dim % kPackK != 0 || dim % 128 != 0 || dim > kMaxDim) return cudaErrorInvalidValue;
    if (rows % kMTile != 0) return cudaErrorInvalidValue;
    // GAP 1 contract: `split` is the row count of EACH pool, so `rows == 2*split`
    // is what makes the epilogue's clamp boundary and the weight row indexing
    // agree. `split % 32 == 0` is what keeps one 32-row SF group inside a single
    // pool (the SF prologue stages 32 rows at a time).
    if (split <= 0 || 2 * split != rows || split % 32 != 0) return cudaErrorInvalidValue;
    // 16-BYTE ALIGNMENT IS A HARD CONTRACT on both sides of every bulk copy: a
    // misaligned copy does not fault, it silently misplaces bytes. Every base
    // and every per-expert stride must be a multiple of 16 -- including w3,
    // which the dual-pointer split now feeds through the same TMA. Reject here
    // instead of producing silently wrong weights.
    const auto al16 = [](const void* p) { return ((uintptr_t)p & 0xF) == 0; };
    const auto str16 = [](long s) { return s == 0 || (s & 0xF) == 0; };
    if (!al16(w1_base) || !al16(w1s_base) || !al16(w3_base) || !al16(w3s_base) ||
        !str16(w1_stride) || !str16(w1s_stride) || !str16(w3_stride) || !str16(w3s_stride))
        return cudaErrorInvalidValue;
    const dim3 grid((unsigned)(rows / kMTile), (unsigned)slots, 1u);
    return dsv41_experts_pdl_or_plain(expert_tcgen05_gateup_mxf4_kernel, grid, dim3(kThreads), 0,
                                      stream, act, act_scale, out, out_slot_stride, dim, epi_mode,
                                      limit, split, w1_base, w1_stride, w1s_base, w1s_stride,
                                      w3_base, w3_stride, w3s_base, w3s_stride, ids);
}

// gate/up, one dispatch per (layer, top-k slot) batch: `act` is the ONE shared
// quantised activation row (packed e2m1, [dim/2]), `act_scale` its [dim/32] f32
// power-of-two scales, `out` holds `slots` consecutive [2*inter] blocks.
// Returns 0 (and does nothing) while disabled, so the caller can call it
// unconditionally and keep the proven GEMV path as the fallback.
//
// ABI (18 params, 2026-09-12 ABI-gap revision; was 11):
//   `*_base` + `*_stride` describe the four weight planes the loader actually
//   uses (`load.rs:644`), and `ids[slot]` selects the expert per grid.y slot.
//   There is deliberately NO separate "direct pointer" pair any more: ids
//   == nullptr makes the four bases BE the direct pointers (one expert, every
//   slot the same answer), which is exactly the parity-harness call, so one
//   form covers both. `rows` is derived (`2 * inter`) and `epi_mode`/`split`
//   are fixed (`1`, `inter`), so they are not ABI surface.
extern "C" int dsv41_expert_tcgen05_gate_up_mxf4(
    const uint8_t* act, const float* act_scale, float* out, long out_slot_stride, int inter,
    int dim, float limit, int slots, const uint8_t* w1_base, long w1_stride,
    const uint8_t* w1s_base, long w1s_stride, const uint8_t* w3_base, long w3_stride,
    const uint8_t* w3s_base, long w3s_stride, const int* ids, cudaStream_t stream) {
    // Runtime gate, read ONCE per process (a per-call getenv is a capture
    // hazard, plan §5). TWO NAMES arm the arm, both with the strict
    // "first char == '1'" rule so `=0` and the unset default are both OFF:
    //   DSV41_EXPERT_TCGEN05_MXF4  the long name (the historical one)
    //   DSV41_EXPERT_TCGEN05       the short alias the (b) task passes
    // ⚠️ MUST stay byte-for-byte equivalent to the Rust mirror
    // `chain_dev.rs::expert_tcgen05_mxf4()`. If the two disagree, the Rust side
    // believes the step runs the tcgen05 arm while the .so keeps the paired
    // GEMV, i.e. BOTH A/B arms measure the OLD path (the project's #1
    // measurement-bias trap).
    static const int enabled = [] {
        const char* names[2] = {"DSV41_EXPERT_TCGEN05_MXF4", "DSV41_EXPERT_TCGEN05"};
        for (const char* n : names) {
            const char* e = getenv(n);
            if (e != nullptr && e[0] == '1') return 1;
        }
        return 0;  // default OFF until the serve A/B
    }();
    if (!enabled) return 0;
    if (inter <= 0 || dim <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
    const cudaError_t e = m4_launch_gateup(act, act_scale, out, out_slot_stride, 2 * inter, dim,
                                           slots, limit, /*epi_mode=*/1, /*split=*/inter, w1_base,
                                           w1_stride, w1s_base, w1s_stride, w3_base, w3_stride,
                                           w3s_base, w3s_stride, ids, stream);
    (void)cudaGetLastError();  // never fail the step: the fallback GEMV is correctness
    return (int)e;
}

}  // namespace mxf4
}  // namespace tc5

#endif  // DSV41_TCGEN05_GATEUP_MXF4_SKELETON

// =============================================================================
// PHASE 1b — tcgen05 e4m3-ACTIVATION gate/up arm (SKELETON)
// =============================================================================
// WHY THIS ARM (the (b) task: "give the tcgen05 skeleton an e4m3 activation")
// -----------------------------------------------------------------------------
// The routed experts' activation can be quantised two ways, and until now only
// one of them could reach a tensor core on this part:
//   * e2m1 PACKED, consumed by `kind::mxf4` (the tc5::mxf4 arm above) -- the
//     cheap form: both operands packed, the TMA lands straight on the canonical
//     units, kRing=8, no expansion pass anywhere;
//   * e4m3, 1 byte per value (`DSV41_EXPERT_ACT_E4M3`) -- the OFFICIAL
//     `fp4_gemm` activation (`act_quant(e4m3, block=32)`), 3 mantissa bits and a
//     wide exponent instead of e2m1's single mantissa bit, i.e. exactly the
//     numeric safety margin a 5-row verify pass is sensitive to.
// `kind::mxf4` is e2m1 x e2m1 and cannot consume e4m3, so the Rust dispatch had
// to REFUSE the tcgen05 arm whenever the e4m3 activation was armed
// (`chain_dev.rs`: `let tcgen05 = !e4m3 && ...`). This arm removes that
// refusal: the SAME swapAB gate/up mapping, with an e4m3 B operand.
//
// -----------------------------------------------------------------------------
// THE KIND IS `kind::mxf8f6f4`, **NOT** `kind::f8f6f4`  (ptxas-verified)
// -----------------------------------------------------------------------------
// The task that asked for this arm named `tcgen05.mma ... kind::f8f6f4` (the
// FP8 x FP4 mixed kind). That spelling exists, but it CANNOT be block-scaled:
//     tcgen05.mma.cta_group::1.kind::f8f6f4.block_scale.scale_vec::1X
//       -> ptxas: "Modifier '.kind::f8f6f4' cannot be combined with modifier
//                  '.block_scale'"
// Without `.block_scale` the MMA sees ONE global scale, so the checkpoint's
// per-(row, k/32) UE8M0 factors cannot be fed to it -- the result would not be
// the checkpoint's quantisation at all. The block-scaled FP8 x FP4 kind is the
// MX variant, and it is what this arm uses:
//     tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X   [OK]
// Probed on this host with nvcc 13.3 `-gencode arch=compute_103a,code=sm_103a`
// (the same form the tc5 arm above and Phase 0's tests_tcgen05_mxf8f6f4_1x.cu
// use). Also probed, all REJECTED: `kind::mxf8f6f4` + `scale_vec::2X` ("cannot
// be combined"), `kind::mxf4` + `scale_vec::1X`, and `kind::mxf4nvf4` +
// `scale_vec::1X`. Consequence: for mxf8f6f4 the scale vector is FORCED to 1X,
// i.e. K = 32 elements per MMA and one e8m0 byte per 32-element block.
//
// -----------------------------------------------------------------------------
// WHAT DIFFERS FROM THE tc5::mxf4 ARM (everything else is reused verbatim)
// -----------------------------------------------------------------------------
//  (1) KIND / FORMATS: kind::mxf8f6f4 + scale_vec::1X, idesc a_format = 5
//      (MXF8F6F4Format::E2M1 -- the fp4 WEIGHT) and b_format = 0
//      (MXF8F6F4Format::E4M3 -- the activation).  ** NOT MXF4Format::E2M1 = 1 **
//      -- the two enums number E2M1 differently (CUTLASS mma_sm100_desc.hpp:
//      MXF4Format::E2M1 = 1, MXF8F6F4Format::E2M1 = 5). Expected idesc for this
//      geometry (M=128, N=8, sf ids 0): 0x08820280.
//  (2) K PER ATOM: 32 instead of 64 (1X vs 2X), so `dim=5120` is 160 MMAs
//      instead of 80 and the SF sub-column selector is `block & 3` instead of
//      `2 * (atom & 1)`. The SF *word* layout, the SF column density, the TMEM
//      budget and the whole SF prologue are UNCHANGED -- one 32-bit word still
//      holds four consecutive checkpoint e8m0 bytes and still occupies one
//      column per 32-row group (tc5::mxf4's note: "an atom pair consumes one SF
//      word per row group exactly like a quad does").
//  (3) THE fp4 OPERAND IS UNPACKED -- the one real cost of this arm, and it is
//      FORCED by the hardware, not chosen. In MXF8F6F4Format the fp4 type is the
//      "unpacksmem" one (CUTLASS: float_e2m1_unpacksmem_t -> MXF8F6F4Format::
//      E2M1 = 5, while the packed float_e2m1_t -> MXF4Format::E2M1 = 1), so each
//      e2m1 element needs its own byte in smem. The TMA cannot perform that
//      expansion (the checkpoint packs 2 fp4 per byte), so every ring slot needs
//      BOTH the raw packed staging area the TMA fills AND the unpacked operand
//      the MMA reads, plus one expansion pass between them.
//      => smem per unit of K is 3x the tc5::mxf4 arm's, so kRing is 6, not 8:
//         in-flight packed bytes (kRing-1)*kARawBytes = 5*2048 = 10 KiB per CTA
//         against 7*4096 = 28 KiB for tc5::mxf4. THAT is the price of the e4m3
//         margin on this part, and it is the first number to measure.
//  (4) B (the activation) is e4m3, one byte per value, and needs NO expansion:
//      the TMA lands its 16-byte chunks directly on the canonical units.
//
// INVARIANT -- `act` FOR THIS ARM IS `dim` BYTES PER ROW (one e4m3 byte per
// value, the `dsv41_quant_fp8` / `act_quant(e4m3, block=32)` output), NOT
// `dim/2` packed bytes. `act_scale` is `[dim/32]` f32 powers of two, exactly as
// for tc5::mxf4 (the kernel converts them to e8m0). A caller that hands over the
// e2m1 row instead would have half of its bytes decoded as garbage: silent wrong
// values, never a fault. The kernel cannot detect it, which is WHY this arm has
// its own SYMBOL and its own runtime gate (`DSV41_EXPERT_TCGEN05_E4M3`) rather
// than a trailing `act_e4m3` argument on the tc5::mxf4 entry point: a stale .so
// would silently ignore a trailing argument, whereas a missing symbol is loud.
//
// -----------------------------------------------------------------------------
// [OPEN -- must be settled on an sm_103a GPU before this arm is trusted]
// -----------------------------------------------------------------------------
//  * THE BYTE ARRANGEMENT INSIDE THE 16-BYTE CORE MATRIX of the unpacked fp4
//    operand has TWO plausible readings, and this arm follows the project's
//    existing prior art (tc5_unpack_a / ph0_parity_kernel: "one e2m1 element per
//    BYTE", element 2i in the low nibble of packed byte i).
//    The other reading comes from the CUDA driver's own description of the TMA
//    data type this format maps to (cuda.h: CU_TENSOR_MAP_DATA_TYPE_16U4_
//    ALIGN16B copies "16 x U4 packed values ... There are 8 byte gaps between
//    every 8 byte chunk"), i.e. the 16 nibble-packed bytes in the LOW half of the
//    16-byte core matrix and a gap in the high half. Both readings agree on
//    DENSITY (32 bytes per 32-element row -- so the smem budget above holds
//    either way) but they disagree on WHERE the nibbles sit. Only a numeric check
//    can pick one; run Phase 0 for this arm too before believing any number.
//  * [TODO-3 analogue] the 2D TENSOR form (a TMA tensor map with
//    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) writes the unpacked layout DIRECTLY:
//    no raw staging, no expansion pass, 2/3 less a_op smem, kRing back to ~9 and
//    16 KiB in flight. It needs a host-side tensor map per weight plane, i.e. the
//    launch arguments would start depending on the routing -- an ARCHITECTURE
//    decision (the current design deliberately stays on plain 1D bulk + raw
//    pointers), so it is flagged here and NOT taken.
// =============================================================================

#ifdef DSV41_TCGEN05_GATEUP_E4M3_SKELETON

#include <cstdlib>  // getenv (launcher gate)

namespace tc5 {
namespace e4 {

// ------------------------------------------------------------------ geometry
constexpr int kMTile = 128;  // MMA M, pinned by the instruction
constexpr int kNTile = 8;    // minimum legal N (mxf8f6f4: [8,256] step 8);
                             // only column 0 carries the token
constexpr int kKStep = 32;   // one 1X scale block == one MMA (dense K32, k_size = 0)
constexpr int kAtomBytes = kKStep;       // 32 UNPACKED fp4 bytes per row per atom
constexpr int kPackK = 32;   // K elements staged per ring slot (== kKStep)
constexpr int kNStep = kPackK / kKStep;  // MMAs issued per slot (== 1)
constexpr int kRing = 6;     // ring depth. 3x the smem per unit of K (see (3)
                             // above) caps this at 6 inside the 48 KiB window.
constexpr int kThreads = 128;  // 4 warps == the 4 TMEM lane partitions
// Descriptor strides. A K=32 atom of the UNPACKED fp4 operand is 32 bytes per
// row = TWO 16-byte core chunks, so the canonical interleave is
//     unit16(m, kb) = (m % 8) + 8*kb + 16*(m / 8)          kb in {0,1}
// => LBO (K-chunk stride) = 8 units = 128 B, SBO (8-row-group stride) =
// 16 units = 256 B. Numerically identical to both other arms: a K=64 PACKED
// atom and a K=32 UNPACKED atom both carry 32 bytes per row.
constexpr int kLboBytes = 128;
constexpr int kSboBytes = 256;
constexpr int kTmemCols = 256;  // power of two >= 8 + 160 + 40 = 208
constexpr int kMaxDim = 5120;   // gate/up dim; sizes the SF staging chunk
constexpr int kSfWords = kMaxDim / 32 / 4;  // 4 block scales per 32-bit word = 40
constexpr int kSfaCols = 4 * kSfWords;      // 160 (one column per (quad, row-group))
constexpr int kSfbCols = kSfWords;          // 40  (one column per quad, N=8 = 1 group)
constexpr int kSfChunkRows = 32;            // one 32-row group staged at a time

// Per-ring-stage operand bytes. A: 128 rows x 32 unpacked bytes. The raw staging
// the TMA fills is the PACKED source of the same atom: 128 rows x 16 bytes.
// B: 8 rows x 32 e4m3 bytes (only row 0 is live; rows 1..7 are zeroed once).
constexpr int kAbBytes = kMTile * kAtomBytes * kNStep;      // 4096 (A operand smem)
constexpr int kARawBytes = kMTile * (kPackK / 2) * kNStep;  // 2048 (packed staging)
constexpr int kBbBytes = kNTile * kAtomBytes * kNStep;      //  256 (B operand smem)
// TMA transaction bytes for one ring stage: the whole packed A atom + the
// 32-byte B row (ONE activation row). Every copy is 16 B.
constexpr unsigned kStageTxBytes =
    (unsigned)(kMTile * (kPackK / 2) * kNStep + kPackK * kNStep);  // 2048 + 32 = 2080
// SF staging chunk: ONE 32-row group of A scales (32 * dim/32 contiguous bytes in
// the pool) + the whole B scale row. Same shape and same size as both other arms
// -- the scale format and the SF column density did not change.
constexpr int kSfAChunkBytes = kSfChunkRows * (kMaxDim / 32);  // 32 * 160 = 5120
constexpr int kSfBChunkBytes = kNTile * (kMaxDim / 32);        //  8 * 160 = 1280

struct Smem {
    // ---- ring: raw packed staging (what the TMA lands) + the unpacked operand
    // (what the MMA reads). The expansion pass in the ring body converts one
    // into the other; there is no way around the second buffer (see (3)).
    alignas(1024) uint8_t a_raw[kRing][kARawBytes];
    alignas(1024) uint8_t a_op[kRing][kNStep][kAbBytes];
    // b_op is written ONLY by the TMA (row 0) and the one-time zeroing (rows
    // 1..7), so the MMA can read it the moment the tx-count retires.
    alignas(128) uint8_t b_op[kRing][kNStep][kBbBytes];
    alignas(1024) uint8_t sf_stage[kSfAChunkBytes > kSfBChunkBytes ? kSfAChunkBytes
                                                                  : kSfBChunkBytes];
    alignas(8) uint64_t tma_bar[kRing];  // TMA tx-count completion, one per slot
    alignas(8) uint64_t mma_bar[kRing];  // tcgen05.commit retirement, one per slot
    uint32_t tmem_base;
};

// The 48 KiB static window. Same rule as both other arms: a tuning change that
// trips this fails the BUILD instead of failing at launch. Going deeper needs
// dynamic smem + an INIT-TIME (never capture-time) cudaFuncSetAttribute -- a
// different risk class, see the DEPTH note in tc5::mxf4.
static_assert(sizeof(Smem) <= 48 * 1024, "tc5::e4 ring does not fit static smem");
static_assert(kNTile + kSfaCols + kSfbCols <= kTmemCols, "tc5::e4 TMEM budget overflow");
// e4_expand_a maps one (row, K_STEP) item to one thread; keep the identity
// explicit so a kThreads change cannot silently leave rows unexpanded.
static_assert(kMTile == kThreads, "tc5::e4 expansion assumes one A row per thread");
// sizeof at the shipped constants: a_raw 12288 + a_op 24576 + b_op 1536 +
// sf_stage 5120 + barriers 96 + tmem_base ~= 43620 B of the 48 KiB window.

// ------------------------------------------------------------------- helpers
// --- the permute segment, expressed as an ADDRESS -----------------------------
// Byte offset of the canonical 16-byte unit holding (row, kb) of a K=32 atom.
// One helper serves BOTH operands (A uses all 16 row groups, the N=8 B operand
// has a single 8-row group so its (row >> 3) term is identically 0 -- which is
// also why the hardware ignores SBO at N=8).
__device__ __forceinline__ int e4_off(int row, int kb) {
    return 16 * ((row & 7) + 8 * kb + 16 * (row >> 3));
}

// --- SMEM operand descriptor, SWIZZLE_NONE K-major, 16-byte units ------------
// Identical in form and value to both other arms' (kLboBytes/kSboBytes above).
// Kept local so this arm's constants cannot drift through a macro switch.
__device__ __forceinline__ uint64_t e4_make_desc(uint32_t smem_base) {
    const uint64_t start = (uint64_t)((smem_base >> 4) & 0x3FFFu);
    const uint64_t lbo = (uint64_t)((kLboBytes >> 4) & 0x3FFFu);  // 8
    const uint64_t sbo = (uint64_t)((kSboBytes >> 4) & 0x3FFFu);  // 16
    return start | (lbo << 16) | (sbo << 32) | ((uint64_t)1 << 46);  // version = 1
}

// --- instruction descriptor, block-scaled, kind::mxf8f6f4 --------------------
// a_format = 5 (MXF8F6F4Format::E2M1, the fp4 WEIGHT, UNPACKED smem form),
// b_format = 0 (MXF8F6F4Format::E4M3, the activation).  ** NOT 1 ** -- 1 is
// MXF4Format::E2M1, the value tc5::mxf4 uses for its PACKED operands; the two
// enums are not interchangeable. Bit layout is the one the file header documents
// at :58-67. For this geometry (M=128, N=8, sf ids 0) the value is 0x08820280:
//   b_sf_id 0 | a_format 5<<7 | b_format 0<<10 | n_dim 1<<17 | scale_format
//   1<<23 (UE8M0, forced) | m_dim 8<<24 | a_sf_id 0 | k_size 0.
__device__ __forceinline__ uint32_t e4_make_idesc(uint32_t a_sf_id, uint32_t b_sf_id) {
    uint32_t d = 0;
    d |= (b_sf_id & 0x3u) << 4;
    d |= 5u << 7;   // a_format = E2M1 (MXF8F6F4Format, the unpacked fp4 weight)
    d |= 0u << 10;  // b_format = E4M3 (MXF8F6F4Format, the activation)
    d |= (uint32_t)(kNTile >> 3) << 17;
    d |= 1u << 23;  // scale_format = UE8M0 (forced, == the checkpoint format)
    d |= (uint32_t)(kMTile >> 4) << 24;
    d |= (a_sf_id & 0x3u) << 29;
    d |= 0u << 31;  // k_size = 0 -> dense K32 for mxf8f6f4
    return d;
}

// --- the MMA (byte-identical to the tc5 arm's tc5_mma / Phase 0's) ------------
__device__ __forceinline__ void e4_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                       uint32_t idesc, uint32_t sfa_tmem, uint32_t sfb_tmem,
                                       uint32_t enable_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
        : "memory");
}

// --- TMEM access (same spellings both other arms verified) --------------------
__device__ __forceinline__ void e4_st_x1(uint32_t taddr, uint32_t w0) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(taddr), "r"(w0)
                 : "memory");
}
__device__ __forceinline__ void e4_ld_x8(uint32_t taddr, uint32_t* v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7])
        : "r"(taddr)
        : "memory");
}

// --- TMA (1D bulk), 16 bytes per copy ----------------------------------------
// Same instruction and the same 2D-tensor [TODO-3] as both other arms; only the
// SOURCE offsets change (16 B per weight row per atom -- a K=32 atom is 16
// PACKED bytes -- and 32 B per activation row). Requirements: 16-byte alignment
// on both sides, size % 16 == 0.
__device__ __forceinline__ void e4_bulk_g2s(void* smem_dst, const void* gmem_src, unsigned bytes,
                                            uint64_t* bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];" ::"r"((unsigned)__cvta_generic_to_shared(smem_dst)),
        "l"(gmem_src), "r"(bytes), "r"((unsigned)__cvta_generic_to_shared(bar))
        : "memory");
#else
    (void)smem_dst; (void)gmem_src; (void)bytes; (void)bar;
#endif
}

// Arms the byte count one stage's copies will retire, and performs the single
// arrival the barrier was initialised with. MUST be issued before the copies it
// covers (the phase completes on arrivals == 1 AND tx count == 0).
__device__ __forceinline__ void e4_mbar_expect_tx(uint64_t* bar, unsigned bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     (unsigned)__cvta_generic_to_shared(bar)),
                 "r"(bytes)
                 : "memory");
#else
    (void)bar; (void)bytes;
#endif
}

// --- the packing-free re-layout of the fp4 operand ---------------------------
// NOTE ON SEMANTICS: this is a NIBBLE->BYTE EXPANSION, not a copy. The canonical
// operand wants one e2m1 element per byte (the "unpacksmem" form); the checkpoint
// stores two per byte. The widening is lane-parallel because it is only a
// byte-level shuffle:
//   p = [b0 b1 b2 b3]  (b = packed byte, low nibble = even element)
//   ev = p & 0x0F0F0F0F  = [e0 e2 e4 e6]   (even elements)
//   od = (p >> 4) & mask = [e1 e3 e5 e7]   (odd elements)
//   out bytes [e0 e1 e2 e3] = __byte_perm(ev, od, 0x5140)
//   out bytes [e4 e5 e6 e7] = __byte_perm(ev, od, 0x7362)
// A 16-byte (uint4) load therefore becomes 8 u32 halves -> 32 unpacked bytes =
// two uint4, i.e. exactly the two K=32 core chunks of one A row. Byte-identical
// to tc5_unpack_a / tc5_expand / tc5_ilv_* above (kept local to this arm so the
// two copies cannot drift through a macro switch).
__device__ __forceinline__ void e4_expand(uint32_t p, uint32_t& ev, uint32_t& od) {
    ev = p & 0x0F0F0F0Fu;
    od = (p >> 4) & 0x0F0F0F0Fu;
}
__device__ __forceinline__ uint32_t e4_ilv_lo(uint32_t ev, uint32_t od) {
    return __byte_perm(ev, od, 0x5140u);  // [e0 e1 e2 e3]
}
__device__ __forceinline__ uint32_t e4_ilv_hi(uint32_t ev, uint32_t od) {
    return __byte_perm(ev, od, 0x7362u);  // [e4 e5 e6 e7]
}

// raw packed A -> the unpacked canonical operand, one (row, K_STEP) per thread.
// kMTile*kNStep items over kThreads threads (== 1 each at the shipped constants).
__device__ __forceinline__ void e4_expand_a(Smem& s, int slot) {
    for (int i = threadIdx.x; i < kMTile * kNStep; i += kThreads) {
        const int m = i / kNStep;   // weight row within this CTA's 128-row tile
        const int st = i % kNStep;  // K_STEP index within the slot
        const uint4 p =
            *reinterpret_cast<const uint4*>(s.a_raw[slot] + (size_t)m * (kPackK / 2) + st * 16);
        uint32_t ev[4], od[4];
        e4_expand(p.x, ev[0], od[0]);
        e4_expand(p.y, ev[1], od[1]);
        e4_expand(p.z, ev[2], od[2]);
        e4_expand(p.w, ev[3], od[3]);
        uint4 o0, o1;  // elements [0,16) and [16,32) of this K_STEP
        o0.x = e4_ilv_lo(ev[0], od[0]);
        o0.y = e4_ilv_hi(ev[0], od[0]);
        o0.z = e4_ilv_lo(ev[1], od[1]);
        o0.w = e4_ilv_hi(ev[1], od[1]);
        o1.x = e4_ilv_lo(ev[2], od[2]);
        o1.y = e4_ilv_hi(ev[2], od[2]);
        o1.z = e4_ilv_lo(ev[3], od[3]);
        o1.w = e4_ilv_hi(ev[3], od[3]);
        // kb = 0 (elements [0,16)) and kb = 1 (elements [16,32)) of the atom.
        uint8_t* dst = s.a_op[slot][st];
        *reinterpret_cast<uint4*>(dst + ((size_t)((m & 7) + 16 * (m >> 3))) * 16) = o0;
        *reinterpret_cast<uint4*>(dst + ((size_t)((m & 7) + 8 + 16 * (m >> 3))) * 16) = o1;
    }
}

// -----------------------------------------------------------------------------
// The kernel. SwapAB, exactly like tc5::mxf4: the WEIGHTS are the M operand
// (A = W1W3 [2*inter, dim] -> M = 2*inter = 30 full 128-row tiles, ZERO
// redundancy) and the activation is the N operand (B = act [dim, 8] e4m3 -> N=8,
// only row 0 live, so the activation side is 8x redundant -- 40 KB of reads per
// step, negligible). D = TMEM [128, 8], only column 0 kept.
//
// Contract (checked by the launcher, not here):
//   dim % kPackK == 0    (the ring never stages a partial K atom)
//   dim % 128 == 0       (1X pairing: a 32-bit SF word spans 4 blocks = 128
//                         elements, so it can never straddle a row end)
//   dim <= kMaxDim       (sizes the SF staging chunk)
//   dim % 2 == 0 and act 16-byte aligned (16-byte bulk-copy contract)
//   rows == 2 * split and split % 32 == 0  (one 32-row SF group stays in a pool)
//   (2*inter) % kMTile == 0    -> 30 tiles at the production shape
//   w1/w3 (+ their scales) rows are per-(row, k/32); act_scale is per-(k/32) f32
//   act is `dim` BYTES (e4m3, 1 byte/value) -- NOT the e2m1 packed row.
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) expert_tcgen05_gateup_e4_kernel(
    const uint8_t* __restrict__ act,        // [dim] e4m3, the ONE token
    const float* __restrict__ act_scale,    // [dim/32] f32 power-of-two scales
    float* __restrict__ out,                // [slots][2*inter] (out_slot_stride apart)
    long out_slot_stride,
    int dim, int epi_mode, float limit, int split,
    // GAP 1 (inherited): TWO weight pools. `split` is the row boundary between
    // gate and up, NOT the row count of either pool: output row r < split is gate
    // pool row r, r >= split is up pool row r - split. Scales follow the split.
    const uint8_t* __restrict__ w1_base, long w1_stride,
    const uint8_t* __restrict__ w1s_base, long w1s_stride,
    const uint8_t* __restrict__ w3_base, long w3_stride,
    const uint8_t* __restrict__ w3s_base, long w3s_stride,
    // GAP 3 (inherited): per-slot expert indirection, so the routing never reaches
    // the host and the launch arguments stay independent of it (CUDA-graph safe).
    // ids == nullptr: the four bases ARE the direct pointers (one expert).
    const int* __restrict__ ids) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot = (int)blockIdx.y;

    // ---- expert / row-block resolution (pure argument arithmetic) -------------
    const uint8_t* w1p = w1_base;
    const uint8_t* w1sp = w1s_base;
    const uint8_t* w3p = w3_base;
    const uint8_t* w3sp = w3s_base;
    if (ids != nullptr) {
        const size_t e = (size_t)ids[slot];
        w1p = w1_base + e * (size_t)w1_stride;
        w1sp = w1s_base + e * (size_t)w1s_stride;
        w3p = w3_base + e * (size_t)w3_stride;
        w3sp = w3s_base + e * (size_t)w3s_stride;
    }
    const int m0 = (int)blockIdx.x * kMTile;  // first weight row of this CTA
    const int kbytes = dim >> 1;              // PACKED weight bytes per row
    const int nsf = dim >> 5;                 // e8m0 bytes per weight row
    const int nwords = nsf >> 2;              // 32-bit SF words per row == quads
    const int natoms = dim / kKStep;          // K=32 atoms over the whole K
    const int ngrp = natoms / kNStep;         // ring iterations
    float* out_s = out + (size_t)slot * (size_t)out_slot_stride;

    __shared__ Smem s;

    // ---- TMEM + barrier init --------------------------------------------------
    if (warp == 0) {
        tc_alloc(&s.tmem_base, kTmemCols);
        tc_relinquish();
    }
    if (tid < kRing) {
        mbar_init(&s.tma_bar[tid], 1u);  // 1 arrival (expect_tx) + the tx bytes
        mbar_init(&s.mma_bar[tid], 1u);  // 1 arrival (tcgen05.commit)
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();

    const uint32_t tb = s.tmem_base;
    const uint32_t d_col = tb;                     // kNTile columns
    const uint32_t sfa_col = tb + kNTile;          // kSfaCols columns
    const uint32_t sfb_col = sfa_col + kSfaCols;   // kSfbCols columns
    // 8 + 160 + 40 = 208 <= kTmemCols = 256. SAME budget as both other arms: a
    // 1X atom consumes one SF word per row group exactly like a quad does.

    // =========================================================================
    // 1. STAGING PROLOGUE -- arm + issue the first kRing-1 ring slots.
    // =========================================================================
    // FIRST, deliberately: the ring TMA is then in flight for the whole of the SF
    // prologue below, so the ~6 KB of scale traffic does not sit in front of the
    // first MMA with an idle memory system behind it.
    //
    // One ring stage = one K=32 atom. The A half is 128 sixteen-byte copies, one
    // per weight row, landed in the PACKED staging area (the expansion pass in
    // the ring body turns it into the operand). The B half is the activation's
    // 32 bytes: 2 sixteen-byte chunks, landed DIRECTLY on their canonical units
    // (e4m3 is already one byte per value -- no expansion for B).
    auto issue = [&](int g) {
        const int sslot = g % kRing;
        const size_t pk0 = (size_t)g * (kPackK / 2);  // packed byte offset of the atom
        if (tid < kMTile) {
            // GAP 1: per ROW, not per CTA -- a 128-row tile may straddle the two
            // pools (the production shape does not, but the kernel must not
            // depend on that).
            const int row = m0 + tid;
            const uint8_t* wrow = (row < split)
                                      ? (w1p + (size_t)row * kbytes)
                                      : (w3p + (size_t)(row - split) * kbytes);
            const uint8_t* src = wrow + pk0;
#pragma unroll
            for (int st = 0; st < kNStep; ++st)
                e4_bulk_g2s(s.a_raw[sslot] + (size_t)tid * (kPackK / 2) + st * 16,
                            src + st * (kKStep / 2), 16, &s.tma_bar[sslot]);
        }
        // B row 0 is the token and the only row transferred; rows 1..7 are zeroed
        // ONCE (below) and never re-written, because their D columns are dead by
        // construction. That is what keeps the activation a single packed-free
        // [dim] byte row instead of an [8, dim] padded buffer.
        if (tid < 2 * kNStep) {
            const int st = tid >> 1, kb = tid & 1;
            e4_bulk_g2s(s.b_op[sslot][st] + e4_off(0, kb),
                        act + (size_t)(g * kPackK + st * kKStep) + 16 * kb, 16, &s.tma_bar[sslot]);
        }
    };
    // Zero B rows 1..7, once, for every (slot, atom): never TMA'd again.
    for (int i = tid; i < kRing * kNStep * (kNTile - 1) * 2; i += kThreads) {
        const int kb = i & 1;
        const int n = 1 + ((i >> 1) % (kNTile - 1));
        const int rest = (i >> 1) / (kNTile - 1);  // which (slot, atom)
        *reinterpret_cast<uint4*>(s.b_op[rest / kNStep][rest % kNStep] + e4_off(n, kb)) =
            make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();  // the tx count must be armed, and the zeroed B rows in smem,
                      // before any copy is issued / any slot is consumed
    if (tid == 0) {
        for (int g = 0; g < kRing - 1; ++g)
            if (g < ngrp) e4_mbar_expect_tx(&s.tma_bar[g], kStageTxBytes);
    }
    __syncthreads();  // arm-before-issue
    for (int g = 0; g < kRing - 1; ++g)
        if (g < ngrp) issue(g);

    // =========================================================================
    // 2. SF PROLOGUE -- the WHOLE scale block into TMEM, once, before the MMAs.
    // =========================================================================
    // Byte-for-byte the same pass as the other two arms: neither the SF FORMAT,
    // nor the SF column density, nor the 4-blocks-per-32-bit-word packing changes
    // with the scale vector (1X vs 2X). Every MMA reads all four row-group
    // columns of its word, so the full 160 SFA columns must exist before the
    // first MMA; the scales cannot be staged lazily per ring slot.
    //
    // The A block for 128 CONSECUTIVE rows is 128 * 160 = 20480 contiguous bytes in
    // the pool, so it is copied one 32-row group at a time (coalesced uint4s) into
    // sf_stage, and the 4-blocks-per-word assembly happens out of smem.
    for (int j = 0; j < 4; ++j) {
        const int r0 = m0 + 32 * j;
        const bool hi = (r0 >= split);
        const int rr = hi ? (r0 - split) : r0;
        const uint8_t* src = (hi ? w3sp : w1sp) + (size_t)rr * nsf;
        // `w3sp`/`w1sp` are the TP-sharded W3/W1 SCALE views (`w3s_base +
        // e*w3s_stride`) — the grouped arm's `bhs_base` under another name. A
        // shard boundary can leave one rank's view a few bytes off while every
        // other rank is fine, and a uint4 read of an odd base is err 716. So the
        // source read goes through the alignment-safe helper; the bytes are
        // IDENTICAL either way (see ld_uint4_a16).
        for (int i = tid; i < 2 * nsf; i += kThreads)  // nsf % 4 == 0 by contract
            reinterpret_cast<uint4*>(s.sf_stage)[i] = ld_uint4_a16(src + (size_t)i * 16);
        __syncthreads();  // the chunk is read by every warp below
        // Word for row (32j + lane), word index w (4 blocks 4w..4w+3), column
        // (sfa_col + 4w + j). Every warp writes its OWN 32-lane partition with
        // identical content: PTX requires the scale factors duplicated to all four
        // partitions, and it is what both Phase 0 harnesses validate.
        for (int w = 0; w < nwords; ++w) {
            const uint32_t word =
                *reinterpret_cast<const uint32_t*>(s.sf_stage + (size_t)lane * nsf + 4 * w);
            e4_st_x1(((uint32_t)(warp * 32) << 16) | (sfa_col + 4 * w + j), word);
        }
        __syncthreads();  // sf_stage is reused by the next row group
    }
    // B scales: one word per quad, replicated across the warps as above. Only lane
    // 0 (activation row 0 = the token) carries a value; lanes 1..7 keep 0, which
    // decodes to 2^-127 -- FINITE, so the zeroed B rows it scales stay exactly 0.
    // Never write 0xFF (NaN): NaN * 0 is NaN and would poison the whole D column.
    for (int w = 0; w < nwords; ++w) {
        // GAP 2 (inherited): `dsv41_quant_fp4`/`quant_fp8` emit f32 powers of two;
        // the SF word wants e8m0 BYTES. Converting here keeps the quantisers' ABI
        // and the whole old SIMT path untouched. Lossless by construction: a
        // fast_round_scale output IS 2^n, so `f_pow2_to_ue8m0` (:132) is an exact
        // exponent copy, and the four bytes are packed exactly as the weight path
        // packs its checkpoint bytes ([b0, b1, b2, b3] of the quad's four 32-K
        // blocks).
        uint32_t word = 0u;
        if (lane == 0) {
            const float* s4 = act_scale + 4 * w;
            word = (uint32_t)f_pow2_to_ue8m0(s4[0]) | ((uint32_t)f_pow2_to_ue8m0(s4[1]) << 8) |
                   ((uint32_t)f_pow2_to_ue8m0(s4[2]) << 16) |
                   ((uint32_t)f_pow2_to_ue8m0(s4[3]) << 24);
        }
        e4_st_x1(((uint32_t)(warp * 32) << 16) | (sfb_col + w), word);
    }
    tc_wait_st();
    tc_fence_before_thread_sync();
    __syncthreads();
    tc_fence_after_thread_sync();

    // =========================================================================
    // 3. THE RING
    // =========================================================================
    // Per iteration: one slot's packed atom lands (TMA completion), is EXPANDED
    // into the operand, is consumed by its MMA, and the slot is refilled. This is
    // the one structural difference from tc5::mxf4: the expansion pass sits
    // between the wait and the fence, because the MMA can only read the unpacked
    // form. Everything else -- the arm-early / issue-late order, the parity
    // bookkeeping, the commit-protects-the-slot argument -- is unchanged.
    //
    // Phase bookkeeping -- slot s is used by iterations g = s (mod kRing); the
    // k-th use has parity k & 1.
    // =========================================================================
#pragma unroll 1
    for (int g = 0; g < ngrp; ++g) {
        const int sslot = g % kRing;
        const uint32_t ph = (uint32_t)((g / kRing) & 1);

        // 1. this slot's bytes have landed. No __syncthreads: the mbarrier wait has
        //    acquire semantics, so whoever observes the phase flip also observes the
        //    TMA writes.
        mbar_wait(&s.tma_bar[sslot], ph);

        // 2. EXPANSION: packed staging -> the unpacked canonical operand the MMA
        //    reads. Safe against the refill of this slot one revolution later: the
        //    refill (issue(g_next), step 6 of iteration g + kRing - 1) is ordered
        //    after the __syncthreads in step 3 of the NEXT iteration, hence after
        //    every thread has finished THIS expansion.
        e4_expand_a(s, sslot);

        // 3. publish the operands to the async proxy. Both the expansion (above)
        //    and the one-time B zeroing are generic-proxy writes of what the MMA
        //    reads.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        // 4. the MMAs of this slot. One thread issues: tcgen05.mma is a CTA-level
        //    async instruction, not a per-thread workload. enable_d = 0 only on the
        //    very first atom.
        if (tid == 0) {
#pragma unroll
            for (int st = 0; st < kNStep; ++st) {
                const int b = g * kNStep + st;  // global 32-block index == the atom
                // 1X SF: the 32-bit word holds the four block scales of a QUAD
                // (bytes 0..3 = blocks 4q..4q+3), so the column advances once per
                // quad and the 2-bit selector picks the block's byte within it.
                const uint32_t sf = (uint32_t)(b & 3);
                const uint64_t da = e4_make_desc(smem_addr(s.a_op[sslot][st]));
                const uint64_t db = e4_make_desc(smem_addr(s.b_op[sslot][st]));
                const uint32_t id = e4_make_idesc(sf, sf);
                e4_mma(d_col, da, db, id, sfa_col + 4u * (uint32_t)(b >> 2),
                       sfb_col + (uint32_t)(b >> 2), b == 0 ? 0u : 1u);
            }
            tc_commit(&s.mma_bar[sslot]);
        }

        // 5. arm the refill barrier of the slot we will fill at the end of this
        //    iteration. The arm touches no memory, so arming it here (while the
        //    slot may still be read by a retired-but-unwaited MMA) is safe.
        const int g_next = g + kRing - 1;
        const int slot_next = g_next % kRing;
        if (tid == 0 && g_next < ngrp)
            e4_mbar_expect_tx(&s.tma_bar[slot_next], kStageTxBytes);

        // 6. refill the slot we are about to overwrite. Its previous use was
        //    iteration g-1 (slot_next == g-1 mod kRing for kRing >= 2), whose commit
        //    was issued at that iteration, so wait that use's parity. The wait also
        //    proves that use's MMA has retired, hence that a_op[slot_next] is free
        //    (its expansion completed BEFORE that MMA).
        if (g_next < ngrp) {
            if (g >= 1)
                mbar_wait(&s.mma_bar[slot_next], (uint32_t)(((g - 1) / kRing) & 1));
            issue(g_next);
        }
    }

    // =========================================================================
    // 4. EPILOGUE -- D -> out. Wait for the LAST MMA first (the only barrier the
    // loop does not consume: each slot's commit is waited one revolution later,
    // except the final slot's).
    // =========================================================================
    mbar_wait(&s.mma_bar[(ngrp - 1) % kRing], (uint32_t)(((ngrp - 1) / kRing) & 1));
    {
        // D[m][n] lives at lane (m % 32) of partition (m / 32), column d_col + n.
        // Only n = 0 carries the token, so only v[0] is kept; v[1..7] are the dead
        // columns (B rows 1..7 are zero).
        uint32_t v[kNTile];
        e4_ld_x8(((uint32_t)(warp * 32) << 16) | d_col, v);
        tc_wait_ld();
        const int row = m0 + warp * 32 + lane;  // == the output column (swapAB)
        float x = __uint_as_float(v[0]);
        if (epi_mode == 1) {  // gate/up clamp, same convention as mxf4_gemm_kernel
            if (limit > 0.f) {
                if (split < 0) {
                    // interleaved (ILV) pool: even row = gate, odd row = up
                    x = (row & 1) ? fminf(fmaxf(x, -limit), limit) : fminf(x, limit);
                } else {
                    // row < split: gate (upper clamp only); else: up (both)
                    x = (row < split) ? fminf(x, limit) : fminf(fmaxf(x, -limit), limit);
                }
            }
        }
        // Phase 2 (down direction) adds row_weight[slot][row] here and the epi_mode
        // 3 accumulate, per mxf4_gemm_kernel:534-548.
        out_s[row] = x;
    }

    __syncthreads();
    if (warp == 0) tc_dealloc(tb, kTmemCols);
}

// =============================================================================
// LAUNCHER
// =============================================================================
// Same shape as tc5::mxf4's m4_launch_gateup (and the same capture rules: the
// gate is read ONCE per process, a per-call getenv is a capture hazard).
//
// [K-SPLIT TODO] The occupancy argument is the one recorded in tc5::mxf4 and the
// mxf8f6f4 block: grid = (30, slots) at slots=8 with <= 2 CTAs/SM, and kRing=6
// supplies 10 KiB in flight per CTA. If slots=8 still lands far from the DRAM
// floor, add the third grid dimension that splits the K range plus an ascending
// order reduce (fp addition is not associative; the order IS the contract). The
// kernel needs no ring change -- only the atom range and the partial write.
// =============================================================================
inline cudaError_t e4_launch_gateup(const uint8_t* act, const float* act_scale, float* out,
                                    long out_slot_stride, int rows, int dim, int slots,
                                    float limit, int epi_mode, int split,
                                    const uint8_t* w1_base, long w1_stride,
                                    const uint8_t* w1s_base, long w1s_stride,
                                    const uint8_t* w3_base, long w3_stride,
                                    const uint8_t* w3s_base, long w3s_stride,
                                    const int* ids, cudaStream_t stream) {
    if (dim <= 0 || rows <= 0 || slots <= 0) return cudaErrorInvalidValue;
    // dim % 128 == 0 IS the 1X pairing contract (a 32-bit SF word spans 4 blocks =
    // 128 K elements); dim % kPackK == 0 keeps the ring on atom boundaries.
    if (dim % kPackK != 0 || dim % 128 != 0 || dim > kMaxDim) return cudaErrorInvalidValue;
    if (rows % kMTile != 0) return cudaErrorInvalidValue;
    // GAP 1 contract: `split` is the row count of EACH pool, so `rows == 2*split`
    // is what makes the epilogue's clamp boundary and the weight row indexing
    // agree. `split % 32 == 0` keeps one 32-row SF group inside a single pool.
    if (split <= 0 || 2 * split != rows || split % 32 != 0) return cudaErrorInvalidValue;
    // 16-BYTE ALIGNMENT IS A HARD CONTRACT on both sides of every bulk copy: a
    // misaligned copy does not fault, it silently misplaces bytes. Every base and
    // every per-expert stride must be a multiple of 16 -- INCLUDING `act`, whose
    // 16-byte chunks are addressed directly (the activation row here is twice as
    // long as the e2m1 one, so a 16-byte-aligned base is not implied by anything
    // else).
    const auto al16 = [](const void* p) { return ((uintptr_t)p & 0xF) == 0; };
    const auto str16 = [](long s) { return s == 0 || (s & 0xF) == 0; };
    if (!al16(act) || !al16(w1_base) || !al16(w1s_base) || !al16(w3_base) || !al16(w3s_base) ||
        !str16(w1_stride) || !str16(w1s_stride) || !str16(w3_stride) || !str16(w3s_stride))
        return cudaErrorInvalidValue;
    const dim3 grid((unsigned)(rows / kMTile), (unsigned)slots, 1u);
    return dsv41_experts_pdl_or_plain(expert_tcgen05_gateup_e4_kernel, grid, dim3(kThreads), 0,
                                      stream, act, act_scale, out, out_slot_stride, dim, epi_mode,
                                      limit, split, w1_base, w1_stride, w1s_base, w1s_stride,
                                      w3_base, w3_stride, w3s_base, w3s_stride, ids);
}

// gate/up, one dispatch per (layer, top-k slot) batch, with the OFFICIAL e4m3
// activation: `act` is the ONE shared quantised activation row (e4m3, 1 byte per
// value, [dim]), `act_scale` its [dim/32] f32 power-of-two scales, `out` holds
// `slots` consecutive [2*inter] blocks.
//
// Returns 0 (and does nothing) while disabled, so the caller can call it
// unconditionally and keep the proven GEMV path as the fallback.
//
// ABI (18 params): byte-for-byte the same signature as
// `dsv41_expert_tcgen05_gate_up_mxf4` -- deliberately, so the Rust binding is a
// mirror and the two arms are interchangeable at the call site. The ONLY
// difference is the activation's BYTE LAYOUT, which no argument can express,
// which is why this arm has its own symbol: the Rust side probes
// `dsv41_expert_tcgen05_gate_up_e4m3` and a `.so` without it keeps the e2m1 arm
// (reported once) instead of silently decoding e4m3 bytes as fp4 nibbles.
extern "C" int dsv41_expert_tcgen05_gate_up_e4m3(
    const uint8_t* act, const float* act_scale, float* out, long out_slot_stride, int inter,
    int dim, float limit, int slots, const uint8_t* w1_base, long w1_stride,
    const uint8_t* w1s_base, long w1s_stride, const uint8_t* w3_base, long w3_stride,
    const uint8_t* w3s_base, long w3s_stride, const int* ids, cudaStream_t stream) {
    // Runtime gate, read ONCE per process (a per-call getenv is a capture hazard,
    // plan §5). ONE name, with the same strict "first char == '1'" rule as the
    // tc5::mxf4 arm so `=0` and the unset default are both OFF:
    //   DSV41_EXPERT_TCGEN05_E4M3
    // ⚠️ MUST stay byte-for-byte equivalent to the Rust mirror
    // `chain_dev.rs::expert_tcgen05_e4m3()`. If the two disagree, the Rust side
    // believes the step runs this arm while the `.so` keeps the paired GEMV, i.e.
    // BOTH A/B arms measure the OLD path (the project's #1 measurement-bias trap).
    static const int enabled = [] {
        const char* e = getenv("DSV41_EXPERT_TCGEN05_E4M3");
        return (e != nullptr && e[0] == '1') ? 1 : 0;
    }();
    if (!enabled) return 0;
    if (inter <= 0 || dim <= 0 || slots <= 0) return (int)cudaErrorInvalidValue;
    const cudaError_t e = e4_launch_gateup(act, act_scale, out, out_slot_stride, 2 * inter, dim,
                                           slots, limit, /*epi_mode=*/1, /*split=*/inter, w1_base,
                                           w1_stride, w1s_base, w1s_stride, w3_base, w3_stride,
                                           w3s_base, w3s_stride, ids, stream);
    (void)cudaGetLastError();  // never fail the step: the fallback GEMV is correctness
    return (int)e;
}

}  // namespace e4
}  // namespace tc5

#endif  // DSV41_TCGEN05_GATEUP_E4M3_SKELETON

// =============================================================================
// PHASE 1c — tcgen05 e4m3-activation MASKED M=128 TILE GEMM with the per-32-block
// scales EXTERNALIZED to the accumulation step (kind::f8f6f4 DENSE — no
// block-scale operands).   [SKELETON — compile-gated, GPU parity pending]
// =============================================================================
// WHY THIS ARM (docs/agent/verify-family-fusion.md W5 + dspark-correctness-chain.md
// "tcgen05 f8f6f4 判词")
// -----------------------------------------------------------------------------
// The routed experts' activation is quantised by the OFFICIAL path as
// `act_quant(e4m3, block=32)` — one e4m3 byte per value, with one e8m0
// (power-of-two) scale per (row, 32 K elements). A tensor-core path that eats
// exactly that activation therefore needs kind::f8f6f4 (FP8 x FP4), and THAT
// kind has no block-scale operands:
//     tcgen05.mma.cta_group::1.kind::f8f6f4.block_scale...  -> ptxas REJECTS
//         "'.kind::f8f6f4' cannot be combined with '.block_scale'"
// (probed; see expert-tcgen05-plan.md §8.1 and tests_tcgen05_mxf8f6f4_1x.cu).
// The only block-scaled FP8 x FP4 kind is kind::mxf8f6f4, which is what the
// tc5::e4 swapAB arm above uses — at the price of forcing scale_vec::1X and of
// UNPACKING the fp4 operand in smem. This arm takes the other road: the MMA runs
// RAW (dense, no scale operand) and the per-32-block scales are applied AFTER it,
// in the accumulation step — the official tilelang inner loop
//     C_local_accum += C_local * scale_a * scale_b
// (dspark-correctness-chain.md:711, "scale 外提"). That is what makes the e4m3
// activation reachable for THIS file's masked M=128 tile GEMM.
//
// -----------------------------------------------------------------------------
// WHAT IT IS: the e4m3 twin of mxf4_gemm_kernel (A = activation)
// -----------------------------------------------------------------------------
// Same masked M=128 tile GEMM, same tile machinery (M=128 / N=64 / K staging in
// 64-element atoms), same B side (the fp4 WEIGHT, with b_split + per-slot `ids`),
// same epilogue conventions. TWO things change:
//   * the A operand is e4m3 — 1 byte per value, the official act_quant output —
//     instead of packed e2m1 (mxf4_gemm_kernel's kAtomBytes / 2 form);
//   * the MMA is kind::f8f6f4 (dense), so the scales can NOT ride in the
//     instruction and are folded by the accumulator instead.
// The scale ROLES are unchanged from mxf4_gemm_kernel, which is exactly why they
// can be moved out instead of needing the TMEM SF lanes the block-scaled arms
// stage:
//     scale_a = the ACTIVATION scale, per (row, 32-block) — f32 powers of two
//     scale_b = the WEIGHT scale,     per (col, 32-block) — e8m0 bytes
//
// -----------------------------------------------------------------------------
// THE DESIGN, AND THE ONE PLACE THIS IMPLEMENTATION DIFFERS FROM IT (honest)
// -----------------------------------------------------------------------------
// W5 reads:
//     for each K-atom (64 elements = 2 x 32-blocks):
//         MMA(A_atom[e4m3], B_atom[fp4]) -> tmem partial C_local[128x64]
//         for each 32-K block of C_local: C_accum += C_local * sa * sb
// A SINGLE 64-element MMA cannot be split that way: its C_local[m][n] is the sum
// of ALL 64 K products, so a per-32 scale applied afterwards would also scale the
// other block's contribution. This implementation therefore issues TWO K=32 MMAs
// per atom — one per 32-block, each into its OWN C_local region (d0/d1) — then
// ONE tcgen05.commit, ONE mbarrier wait and two scale-folds. Consequences:
//   * the tmem ROUND TRIP count is exactly the design's: one commit+wait+read per
//     atom = K / kAtomK = 2304 / 64 = 36. (The two loads of a round trip are
//     separate but share the single wait: the two MMAs retire on one arrival.)
//   * `enable_input_d = 0` on BOTH MMAs: each C_local IS the RAW partial, and the
//     accumulator lives in registers (there is nothing left for the tensor core
//     to accumulate, because the scale would have to be applied in between).
//   * the scales are constant per 32-block for the whole K loop, so they are
//     staged ONCE per stage into smem (coalesced) rather than re-read per fold.
//
// -----------------------------------------------------------------------------
// NUMERIC DOMAIN (why the multiply order is safe)
// -----------------------------------------------------------------------------
// 1. The MMA accumulates in f32: C_local is the f32 sum of 32 products of an
//    e4m3 activation value and an e2m1 weight value, computed by the tensor core
//    (its internal order is the hardware's, exactly as in the block-scaled arms).
// 2. scale_a is a fast_round_scale output and scale_b is an e8m0 byte: BOTH are
//    powers of two (2^p, 2^q), so every multiply by them is an EXACT exponent
//    shift — no rounding, and no reassociation risk under --use_fast_math (which
//    this .so builds with). Only the final `acc += ...` rounds, once per
//    (atom, 32-block) contribution. This is why `(v*sa)*sb` and `v*(sa*sb)` are
//    the SAME f32 value here (the product sa*sb is itself exact), i.e. the
//    design's "乘法可交换" holds numerically, not just algebraically.
// 3. The accumulation order is FIXED and ascending in K: sub-block 0 then 1
//    inside the atom, atoms ascending inside the stage, stages ascending over K.
//    One f32 chain per (m, n) — no split accumulators, no shuffle tree — so the
//    order is a property of the source, not of the scheduler.
// 4. Range: sa*sb underflows the f32 exponent only if p + q < -126. The
//    checkpoint's scales are chosen so each block's values land in the operand
//    type's range, and a flushed-to-zero product is the same edge the SIMT path
//    has. e8m0 0x00 decodes to 2^-127, which is FINITE — the weight path never
//    writes 0xFF (NaN), which would poison the whole fold.
//
// -----------------------------------------------------------------------------
// [OPEN — settle on an sm_103a GPU before this arm is trusted]
// -----------------------------------------------------------------------------
//  * the non-block-scaled idesc format codes. This arm takes a_format = E4M3 = 0
//    and b_format = E2M1 = 5 from MXF8F6F4Format — the only enum this repo has
//    decoded (tests_tcgen05_mxf8f6f4_1x.cu:836). The non-block-scaled form
//    spends bits [4,6) on the D type instead of b_sf_id; f32 (1) is used because
//    a f32 accumulate is the whole point of the parity with the SIMT path.
//  * the fp4 operand of the f8f6f4 family is the UNPACKED ("unpacksmem") form:
//    one e2m1 element per BYTE, i.e. 32 bytes per row per K=32 atom = the same
//    geometry as the e4m3 A operand. This is the file's own documented finding
//    for kind::mxf8f6f4 (CUTLASS float_e2m1_unpacksmem_t -> MXF8F6F4Format::E2M1
//    while the packed float_e2m1_t -> MXF4Format::E2M1, (:3030)) and it is what
//    keeps ONE descriptor form (LBO 128 B / SBO 256 B) valid for both operands.
//    The alternative reading (packed fp4, 16 bytes per row = a single core
//    chunk) would halve B's smem but invent an SBO=128 B descriptor this repo
//    has no precedent for — it is one numeric parity run away either way.
// =============================================================================

#ifdef DSV41_TCGEN05_GATEUP_E4M3_SKELETON

#include <cstdlib>  // getenv (launcher gate)

namespace tc5 {
namespace e4x {

// ------------------------------------------------------------------ geometry
constexpr int kMTile = 128;  // MMA M (1-CTA kind::f8f6f4 is fixed at 128)
constexpr int kNTile = 64;   // output columns per CTA (N of the MMA)
constexpr int kSubK = 32;    // dense K of ONE kind::f8f6f4 MMA: 32 elements
                             // (== 32 e4m3 bytes == 32 unpacked fp4 bytes per row)
constexpr int kAtomK = 64;   // K per staging atom == 2 sub-blocks == 2 scale blocks
constexpr int kSubs = kAtomK / kSubK;  // 2 sub-blocks (2 scale blocks) per atom
// Atoms per smem stage. This is SMALLER than mxf4_gemm_kernel's 4 on purpose:
// the e4m3 activation is 1 byte/element AND the fp4 weight has to be expanded to
// the unpacked form, so a sub-block costs 4096 (A) + 2048 (B) bytes against 4096
// for a whole K=64 fp4 atom there. 2 atoms (128 K elements) is what fits the
// same 48 KiB window with the scales staged.
constexpr int kStageAtoms = 2;
constexpr int kSubsPerStage = kStageAtoms * kSubs;  // 4
constexpr int kThreads = 128;                       // 4 warps == 4 TMEM partitions
// Descriptor strides in bytes, for the canonical K-major SWIZZLE_NONE atom
//     unit16(row, kb) = (row % 8) + 8*kb + 16*(row / 8)        kb in {0,1}
// Both operands carry 32 bytes per row per K=32 sub-block (e4m3 = 1 byte/element;
// unpacked fp4 = 1 byte/element), i.e. 2 x 16-byte core chunks, so LBO (chunk
// stride) = 128 B and SBO (8-row-group stride) = 256 B — the SAME numbers as the
// file-scope make_desc() and mxf4_gemm_kernel's operand (only the element size
// changed, so the descriptor did not).
constexpr int kLboBytes = 128;
constexpr int kSboBytes = 256;
// Per-K=32-sub-block operand bytes.
constexpr int kASubBytes = kMTile * kSubK;        // 128 x 32 = 4096 (e4m3 A)
constexpr int kBSubBytes = kNTile * kSubK;        //  64 x 32 = 2048 (unpacked fp4 B)
constexpr int kABytes = kSubsPerStage * kASubBytes;  // 16384
constexpr int kBBytes = kSubsPerStage * kBSubBytes;  //  8192
// TMEM. D is no longer an accumulator: it is TWO per-sub-block PARTIAL regions
// (C_local), one per 32-block of the atom in flight. tc_alloc wants a power of
// two, so the pair is allocated as 128 columns.
constexpr int kDCols = kNTile;         // 64 columns per C_local region
constexpr int kTmemCols = 2 * kDCols;  // 128
static_assert((kTmemCols & (kTmemCols - 1)) == 0, "tc_alloc wants a power of two");

struct Smem {
    // ---- operands, canonical K-major, one block per K=32 sub-block ---------
    alignas(1024) uint8_t a[kABytes];  // e4m3 activation, 1 byte per value
    alignas(1024) uint8_t b[kBBytes];  // fp4 weight, UNPACKED 1 element/byte
    // ---- scales of the stage in flight -------------------------------------
    // [sub][row] and [sub][col], TRANSPOSED so the fold's per-lane A-scale read
    // is stride 1 (bank-conflict free) and its per-column B-scale read is a
    // uniform address (broadcast). f32 because that is the precision the fold
    // multiplies in; the e8m0 bytes are decoded on the way in.
    float sa[kSubsPerStage][kMTile];
    float sb[kSubsPerStage][kNTile];
    alignas(8) uint64_t mbar;  // tcgen05.commit retirement (one arrival per atom)
    uint32_t tmem_base;
};
// The 48 KiB static window, same rule as every other arm in this file: a tuning
// change that trips this fails the BUILD instead of failing at launch.
static_assert(sizeof(Smem) <= 48 * 1024, "tc5::e4x stage does not fit static smem");
// 16384 + 8192 + 2048 + 1024 + 12 ~= 27.7 KiB.

// ------------------------------------------------------------------- helpers
// Byte offset of the canonical 16-byte unit holding (row, kb) of a K=32 atom.
__device__ __forceinline__ int e4x_off(int row, int kb) {
    return 16 * ((row & 7) + 8 * kb + 16 * (row >> 3));
}
// SMEM operand descriptor. Kept local (not the file-scope make_desc) so this
// arm's constants cannot drift through a macro switch; the VALUES are identical.
__device__ __forceinline__ uint64_t e4x_make_desc(uint32_t smem_base) {
    const uint64_t start = (uint64_t)((smem_base >> 4) & 0x3FFFu);
    const uint64_t lbo = (uint64_t)((kLboBytes >> 4) & 0x3FFFu);  // 8
    const uint64_t sbo = (uint64_t)((kSboBytes >> 4) & 0x3FFFu);  // 16
    return start | (lbo << 16) | (sbo << 32) | ((uint64_t)1 << 46);  // version = 1
}
// Instruction descriptor, kind::f8f6f4, NON-block-scaled. Bits [4,6) carry the D
// type here (the block-scaled form spends them on b_sf_id), bits 7-9 / 10-12 the
// A/B formats (MXF8F6F4Format: E4M3 = 0, E2M1 = 5), bits 13-16 the negates and
// majors (all 0 = K-major, no negation). See [OPEN] in the header.
__device__ __forceinline__ uint32_t e4x_make_idesc() {
    uint32_t d = 0;
    d |= 1u << 4;   // D / accumulator type = f32
    d |= 0u << 7;   // a_format = E4M3 (A = the activation)
    d |= 5u << 10;  // b_format = E2M1 (B = the fp4 weight)
    d |= (uint32_t)(kNTile >> 3) << 17;  // n_dim
    d |= (uint32_t)(kMTile >> 4) << 24;  // m_dim
    return d;
}
// The MMA. NOTE the operand list: FIVE operands, no [sf_a]/[sf_b] — that is the
// whole point of this arm (kind::f8f6f4 cannot carry them, so the scales are
// folded by the caller). `enable_d` is always 0 here: every MMA produces the RAW
// partial of its own 32-block.
__device__ __forceinline__ void e4x_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                        uint32_t idesc, uint32_t enable_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d)
        : "memory");
}
// Packed fp4 -> UNPACKED e2m1 (one element per byte). Same nibble->byte widening
// as tc5_expand/e4_expand above (kept local to this arm for the same reason):
//   p = [b0 b1 b2 b3], bi = e(2i) | e(2i+1) << 4
//   ev = p & 0x0F0F0F0F = [e0 e2 e4 e6]   od = (p >> 4) & mask = [e1 e3 e5 e7]
//   out bytes [e0 e1 e2 e3] = __byte_perm(ev, od, 0x5140)
//   out bytes [e4 e5 e6 e7] = __byte_perm(ev, od, 0x7362)
// A uint4 (16 packed bytes = 32 elements = ONE K=32 sub-block) therefore becomes
// two uint4, i.e. exactly the two core chunks (kb 0/1) of one B row.
__device__ __forceinline__ void e4x_expand(uint32_t p, uint32_t& ev, uint32_t& od) {
    ev = p & 0x0F0F0F0Fu;
    od = (p >> 4) & 0x0F0F0F0Fu;
}
__device__ __forceinline__ uint32_t e4x_ilv_lo(uint32_t ev, uint32_t od) {
    return __byte_perm(ev, od, 0x5140u);  // [e0 e1 e2 e3]
}
__device__ __forceinline__ uint32_t e4x_ilv_hi(uint32_t ev, uint32_t od) {
    return __byte_perm(ev, od, 0x7362u);  // [e4 e5 e6 e7]
}

// -----------------------------------------------------------------------------
// The kernel.
//
// Contract (checked by the launcher, not here):
//   k % kAtomK == 0        (a stage never carries a half atom)
//   rows % kMTile == 0, n_total % kNTile == 0   (full CTA tiles)
//   a       [rows, k]    e4m3, ONE byte per value (the act_quant(e4m3, 32) row)
//   a_scale [rows, k/32] f32 powers of two
//   b/b_hi  [r, k/2]     fp4, packed 2 values/byte, + [r, k/32] e8m0 scales
//   k % 32 == 0 (the scale-block granularity), 16-byte alignment on every base
// -----------------------------------------------------------------------------
// NOTE on the launch bounds: the plain form leaves ptxas at 162 registers
// (0 spill) = 3 CTAs/SM; adding a `, 4` min-blocks target buys 128 registers
// (4 CTAs/SM) at the cost of 8 bytes of spill. Latency is this arm's risk, so
// the occupancy knob is real — but the file's arms are expected to compile with
// 0 spill, so the hint stays off and the GPU A/B flips it if occupancy wins.
__global__ void __launch_bounds__(kThreads) e4m3_gemm_kernel(
    const uint8_t* __restrict__ a,        // [rows, k] e4m3, one byte per value
    const float* __restrict__ a_scale,    // [rows, k/32] f32 powers of two
    const uint8_t* __restrict__ b,        // [b_rows, k/2] fp4, packed
    const uint8_t* __restrict__ b_scale,  // [b_rows, k/32] e8m0
    const uint8_t* __restrict__ b_hi,     // second half (b_split >= 0), else b
    const uint8_t* __restrict__ b_hi_scale,
    float* __restrict__ out,              // [rows, n_total]
    int rows, int n_total, int k, int b_split, int epi_mode, float limit,
    const float* __restrict__ row_weight,
    // Indirect (graph-friendly) B addressing, same convention as
    // mxf4_gemm_kernel: ids != nullptr derives the four B pointers per CTA from
    // `base + ids[slot] * stride`, so the routing never reaches the host;
    // ids == nullptr keeps the direct path (one expert, every slot identical).
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
    const int kbytes = k >> 1;  // PACKED fp4 weight bytes per row
    const int nk_blk = k >> 5;  // 32-element blocks == scale columns

    __shared__ Smem s;

    // ---------------------------------------------------------- tmem alloc
    if (warp == 0) {
        tc_alloc(&s.tmem_base, kTmemCols);
        tc_relinquish();
    }
    if (tid == 0) mbar_init(&s.mbar, 1);
    __syncthreads();

    const uint32_t tmem_base = s.tmem_base;
    const uint32_t d0_col = tmem_base;           // C_local, sub-block 0 (64 cols)
    const uint32_t d1_col = tmem_base + kDCols;  // C_local, sub-block 1 (64 cols)

    // --------------------------------------------------- register accumulator
    // acc[c] holds row (m_base + warp*32 + lane) x column (n_base + c) for the
    // WHOLE K loop. It has to live in registers: the scale fold is not something
    // the tensor core can do, so the partials must come out to where a multiply
    // can happen. One f32 chain per (m, n), ascending K (see NUMERIC DOMAIN).
    float acc[kNTile];
#pragma unroll
    for (int c = 0; c < kNTile; ++c) acc[c] = 0.f;

    // ------------------------------------------------------------- K loop
    uint32_t phase = 0;
    for (int k0 = 0; k0 < k; k0 += kStageAtoms * kAtomK) {
        const int natoms = min(kStageAtoms, (k - k0 + kAtomK - 1) / kAtomK);

        // ---- 1. stage the A operand (e4m3) --------------------------------
        // One 16-byte chunk per (atom, row, sub, kb). The e4m3 row is already one
        // byte per value, so this is a straight 16-byte copy onto the canonical
        // unit — no expansion pass (that is this arm's advantage over every fp4
        // operand in the file: 2 values per byte become 1 byte per value here).
        for (int c = tid; c < kStageAtoms * kMTile * 2 * kSubs; c += kThreads) {
            const int atom = c / (kMTile * 2 * kSubs);
            if (atom >= natoms) continue;
            const int r = c % (kMTile * 2 * kSubs);
            const int m = r >> 2;  // row within this CTA's M tile
            const int q = r & 3;   // chunk within the atom: 2 subs x 2 kb
            const int sub = q >> 1, kb = q & 1;
            const int row = m_base + m;
            const size_t kk = (size_t)k0 + (size_t)atom * kAtomK + sub * kSubK + kb * 16;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (row < rows && kk + 16 <= (size_t)k)
                val = ld_uint4_a16(a + (size_t)row * k + kk);
            *reinterpret_cast<uint4*>(s.a + (atom * kSubs + sub) * kASubBytes +
                                      e4x_off(m, kb)) = val;
        }

        // ---- 2. stage + expand the B operand (packed fp4 -> unpacked) ------
        // The source is the checkpoint's 16 packed bytes of this row's K=32
        // sub-block; the destination is the two 16-byte core chunks of one B row
        // (kb 0 = elements 0..15, kb 1 = elements 16..31). See the [OPEN] note in
        // the header: the f8f6f4 family's fp4 operand is the unpacked form, which
        // is why this expansion exists at all.
        for (int c = tid; c < kStageAtoms * kNTile * kSubs; c += kThreads) {
            const int atom = c / (kNTile * kSubs);
            if (atom >= natoms) continue;
            const int r = c % (kNTile * kSubs);
            const int n = r >> 1, sub = r & 1;
            const int n_glob = n_base + n;
            const size_t ko = (size_t)(k0 + atom * kAtomK + sub * kSubK) >> 1;
            uint4 o0 = make_uint4(0, 0, 0, 0), o1 = make_uint4(0, 0, 0, 0);
            if (n_glob < n_total && ko + 16 <= (size_t)kbytes) {
                const uint8_t* src_base = b_use;
                int row = n_glob;
                if (b_split >= 0 && n_glob >= b_split) {
                    src_base = bhi_use;
                    row = n_glob - b_split;
                }
                if (row >= 0) {
                    const uint4 p = ld_uint4_a16(src_base + (size_t)row * kbytes + ko);
                    uint32_t ev[4], od[4];
                    e4x_expand(p.x, ev[0], od[0]);
                    e4x_expand(p.y, ev[1], od[1]);
                    e4x_expand(p.z, ev[2], od[2]);
                    e4x_expand(p.w, ev[3], od[3]);
                    o0.x = e4x_ilv_lo(ev[0], od[0]);
                    o0.y = e4x_ilv_hi(ev[0], od[0]);
                    o0.z = e4x_ilv_lo(ev[1], od[1]);
                    o0.w = e4x_ilv_hi(ev[1], od[1]);
                    o1.x = e4x_ilv_lo(ev[2], od[2]);
                    o1.y = e4x_ilv_hi(ev[2], od[2]);
                    o1.z = e4x_ilv_lo(ev[3], od[3]);
                    o1.w = e4x_ilv_hi(ev[3], od[3]);
                }
            }
            uint8_t* dst = s.b + (atom * kSubs + sub) * kBSubBytes;
            *reinterpret_cast<uint4*>(dst + e4x_off(n, 0)) = o0;
            *reinterpret_cast<uint4*>(dst + e4x_off(n, 1)) = o1;
        }

        // ---- 3. stage this stage's scales (once, coalesced) ---------------
        // The scales are the SAME for every MMA of the K loop given the block
        // index, so they are decoded once per stage instead of once per fold: the
        // fold then touches only smem. A: [sub][row] f32; B: [sub][col] f32
        // (e8m0 -> f32 on the way in; ue8m0_to_f is the file's 2^(b-127), 0x00
        // stays FINITE so a zeroed/unwritten block can never poison the sum).
        for (int c = tid; c < kSubsPerStage * kMTile; c += kThreads) {
            const int sub = c / kMTile;
            const int m = c % kMTile;
            const int row = m_base + m;
            const int bb = (k0 >> 5) + sub;  // global 32-block of this sub
            float v = 0.f;
            if (row < rows && bb < nk_blk) v = a_scale[(size_t)row * nk_blk + bb];
            s.sa[sub][m] = v;
        }
        for (int c = tid; c < kSubsPerStage * kNTile; c += kThreads) {
            const int sub = c / kNTile;
            const int n = c % kNTile;
            const int n_glob = n_base + n;
            const int bb = (k0 >> 5) + sub;
            float v = 0.f;
            if (n_glob < n_total && bb < nk_blk) {
                const uint8_t* sc = bsc_use;
                int row = n_glob;
                if (b_split >= 0 && n_glob >= b_split) {
                    sc = bhs_use;
                    row = n_glob - b_split;
                }
                if (row >= 0) v = ue8m0_to_f(sc[(size_t)row * nk_blk + bb]);
            }
            s.sb[sub][n] = v;
        }

        // publish every smem write to the async proxy (the MMA reads it there)
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        // ---- 4. per atom: 2 MMAs (one per 32-block) + ONE commit/wait + fold
        // This is the design's inner loop: one tmem round trip per K-atom, with
        // the scale applied on the way out of tmem (C_accum += C_local * sa * sb).
        for (int atom = 0; atom < natoms; ++atom) {
            if (tid == 0) {
                const uint32_t id = e4x_make_idesc();
                const uint64_t da0 =
                    e4x_make_desc(smem_addr(s.a + (atom * kSubs + 0) * kASubBytes));
                const uint64_t db0 =
                    e4x_make_desc(smem_addr(s.b + (atom * kSubs + 0) * kBSubBytes));
                const uint64_t da1 =
                    e4x_make_desc(smem_addr(s.a + (atom * kSubs + 1) * kASubBytes));
                const uint64_t db1 =
                    e4x_make_desc(smem_addr(s.b + (atom * kSubs + 1) * kBSubBytes));
                // enable_input_d = 0 on BOTH: each region holds the RAW partial of
                // its own 32-block, and no tensor-core accumulate can happen
                // across two blocks with different scales.
                e4x_mma(d0_col, da0, db0, id, 0u);
                e4x_mma(d1_col, da1, db1, id, 0u);
                tc_commit(&s.mbar);  // ONE arrival covers BOTH MMAs of the atom
            }
            mbar_wait(&s.mbar, phase);
            phase ^= 1u;

            const int mrow = warp * 32 + lane;  // this thread's row in the M tile
#pragma unroll
            for (int sub = 0; sub < kSubs; ++sub) {
                const int ssub = atom * kSubs + sub;  // stage-local sub index
                const uint32_t dcol = (sub == 0) ? d0_col : d1_col;
                const float sa = s.sa[ssub][mrow];  // [row, 32-block] activation
#pragma unroll
                for (int c0 = 0; c0 < kNTile; c0 += 16) {
                    uint32_t v[16];
                    tc_ld_x16((((uint32_t)(warp * 32)) << 16) | (dcol + c0), v);
                    tc_wait_ld();
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        // sa and sb are powers of two, so sa*sb is EXACT: the
                        // association below is numerically irrelevant, and the
                        // only rounding is the accumulate (see NUMERIC DOMAIN).
                        acc[c0 + i] += __uint_as_float(v[i]) * (sa * s.sb[ssub][c0 + i]);
                    }
                }
            }

            // Every warp has now consumed both C_local regions; the NEXT atom's
            // MMAs overwrite them, so all four warps' tcgen05.ld must be retired
            // CTA-wide before tid 0 issues them. Same idiom the SF prologue uses
            // for a generic-proxy write feeding the async proxy.
            tc_fence_before_thread_sync();
            __syncthreads();
            tc_fence_after_thread_sync();
        }
    }

    // ------------------------------------------------------------ epilogue
    // No final tmem wait is needed: every atom's MMAs were waited inside the loop
    // (that IS the 36 round trips), so `acc` is complete here. Read-out mapping
    // is mxf4_gemm_kernel's: lane (m % 32) of partition (m / 32) is row m and the
    // column index is the N index (this arm does NOT use the swapAB mapping).
    // The mask (row/col bounds) and every epi_mode convention are inherited
    // verbatim, so a caller can A/B this arm against the e2m1 one unchanged.
#pragma unroll
    for (int c = 0; c < kNTile; ++c) {
        const int row = m_base + warp * 32 + lane;
        const int col = n_base + c;
        if (row >= rows || col >= n_total) continue;
        float x = acc[c];
        if (epi_mode == 1) {  // gate/up clamps (training convention)
            if (limit > 0.f) {
                if (col < b_split) x = fminf(x, limit);           // gate
                else x = fminf(fmaxf(x, -limit), limit);          // up
            }
        } else if (epi_mode == 2 || epi_mode == 3) {  // down: routing weight
            if (row_weight != nullptr) x *= row_weight[row];
        }
        // epi_mode 3 accumulates straight into the caller's MoE accumulator.
        if (epi_mode == 3) out[(size_t)row * n_total + col] += x;
        else out[(size_t)row * n_total + col] = x;
    }

    __syncthreads();
    if (warp == 0) tc_dealloc(tmem_base, kTmemCols);
}

// =============================================================================
// LAUNCHER
// =============================================================================
// Same shape as mxf4_gemm_kernel's launcher (one CTA per (n-tile, m-tile), the
// grid iterating n fastest) and the same capture rule: the runtime gate is read
// ONCE per process, because a per-call getenv is a CUDA-graph capture hazard.
// =============================================================================
inline cudaError_t e4x_launch_gemm(const uint8_t* a, const float* a_scale, const uint8_t* b,
                                   const uint8_t* b_scale, const uint8_t* b_hi,
                                   const uint8_t* b_hi_scale, float* out, int rows, int n_total,
                                   int k, int b_split, int epi_mode, float limit,
                                   const float* row_weight, const uint8_t* b_base, long b_stride,
                                   const uint8_t* bs_base, long bs_stride, const uint8_t* bh_base,
                                   long bh_stride, const uint8_t* bhs_base, long bhs_stride,
                                   const int* ids, int slot, cudaStream_t stream) {
    if (rows <= 0 || n_total <= 0 || k <= 0) return cudaErrorInvalidValue;
    // k % kAtomK: a stage never carries a half atom (and k % 32 == 0 is implied,
    // the scale block being the MMA's own K). The two tile contracts keep every
    // CTA on a full tile — the kernel masks out-of-range rows/columns anyway, but
    // a partial tile would silently change the folding pattern, so it is refused
    // here rather than tolerated.
    if (k % kAtomK != 0 || rows % kMTile != 0 || n_total % kNTile != 0)
        return cudaErrorInvalidValue;
    // 16-byte alignment is a hard contract on the operand bases: the K=32 e4m3
    // row is addressed in 16-byte chunks and the packed fp4 source in uint4s. A
    // misaligned access does not fault, it silently misplaces bytes.
    const auto al16 = [](const void* p) { return ((uintptr_t)p & 0xF) == 0; };
    if (!al16(a) || !al16(b) || !al16(b_hi)) return cudaErrorInvalidValue;
    const dim3 grid((unsigned)(n_total / kNTile), (unsigned)(rows / kMTile));
    e4m3_gemm_kernel<<<grid, kThreads, 0, stream>>>(
        a, a_scale, b, b_scale, b_hi, b_hi_scale, out, rows, n_total, k, b_split, epi_mode, limit,
        row_weight, b_base, b_stride, bs_base, bs_stride, bh_base, bh_stride, bhs_base, bhs_stride,
        ids, slot);
    return cudaGetLastError();
}

// "e4m3 activation x fp4 weight, masked M=128 tile GEMM, per-32-block scales
// externalized to the accumulation step". Returns 0 (and does nothing) while
// disabled, so a caller can invoke it unconditionally and keep the SIMT/GEMV
// path as the fallback — the same contract as every other arm in this file.
//
// Runtime gate: DSV41_EXPERT_TCGEN05_E4M3 (default OFF, strict first-char '1'
// rule, read once). This arm is the w=row sibling of the tc5::e4 swapAB arm and
// shares its activation FORMAT, so it shares the gate name: which of the two a
// step runs is a launch-shape decision (a dense M=128 tile of activation rows vs
// the single-token swapAB form), NOT a numeric-format decision. Each has its own
// symbol, so a .so carrying only one of them still behaves (the Rust side probes
// `dsv41_expert_tcgen05_gate_up_e4m3` today; this arm's symbol is additive and
// carries no ABI bump).
extern "C" int dsv41_expert_gemm_e4m3_ext(
    const uint8_t* a, const float* a_scale, const uint8_t* b, const uint8_t* b_scale,
    const uint8_t* b_hi, const uint8_t* b_hi_scale, float* out, int rows, int n_total, int k,
    int b_split, int epi_mode, float limit, const float* row_weight, const uint8_t* b_base,
    long b_stride, const uint8_t* bs_base, long bs_stride, const uint8_t* bh_base, long bh_stride,
    const uint8_t* bhs_base, long bhs_stride, const int* ids, int slot, cudaStream_t stream) {
    static const int enabled = [] {
        const char* g = getenv("DSV41_EXPERT_TCGEN05_E4M3");
        return (g != nullptr && g[0] == '1') ? 1 : 0;  // default OFF until serve A/B
    }();
    if (!enabled) return 0;
    const cudaError_t rc = e4x_launch_gemm(a, a_scale, b, b_scale, b_hi, b_hi_scale, out, rows,
                                           n_total, k, b_split, epi_mode, limit, row_weight,
                                           b_base, b_stride, bs_base, bs_stride, bh_base,
                                           bh_stride, bhs_base, bhs_stride, ids, slot, stream);
    (void)cudaGetLastError();  // never fail the step: the fallback path is correctness
    return (int)rc;
}

// =============================================================================
// GROUP-INDEXED MASKED GEMM — the `m_grouped_gemm_nt_masked` form
// =============================================================================
// WHY THIS EXISTS. `e4m3_gemm_kernel` above is EXPERT-CENTRIC: its whole B side
// is derived from ONE `ids[slot]`, so one launch covers ONE expert over a DENSE
// row block, while `moe_rows`' routing is per-(row, slot) (`route_idx_r[m][topk]`).
// Handing that table to the dense arm would apply slot s's expert to rows that
// route to a DIFFERENT expert — a SILENT WRONG ANSWER, which is exactly why the
// dense arm declines today (`e4x_tile = false` in chain_dev.rs::moe_rows).
//
// The missing form is DeepGEMM's `m_grouped_gemm_nt_masked`: the routed rows are
// re-ordered so that every expert's rows are CONTIGUOUS (the grouped layout of
// `dsv41_route_group`), a CTA owns ONE kMTile=128 row MMA tile OF ONE EXPERT, and
// the rows past that expert's real row count are MASKED — the A operand is
// zero-filled there and the epilogue writes nothing. At this engine's shapes
// (m <= VERIFY_ROWS = 6, topk = 6) an expert owns O(1) rows, so the masking is
// what makes the tensor-core arm viable at all: a per-expert DENSE launch would
// have to pad 1-3 real rows out to 128 (85x the work of the whole step), while
// this kernel still runs the 128-row MMA but only ~1/128 of its columns carry a
// non-zero A.
//
// GRID (blockIdx.y is the intra-expert m-tile, blockIdx.z the group-table
// cursor, blockIdx.x the N tile — the SAME (n, m) plane as the dense arm, with
// the group axis appended rather than a second m-tile sum):
//   x : n_total / kNTile                 — N tiles of ONE expert's weight planes
//   y : ceil(m_cap / kMTile)             — m-tiles INSIDE one expert's block
//   z : the ACTIVE-list cursor (n_assign entries; the caller passes `m*topk`,
//       the number of grouped positions the layout can ever hold)
// Per CTA, from the group table (all four reads are uniform across the CTA, so
// each is ONE L2 broadcast per warp):
//   e        = active[z]                 — exit when z >= *n_active
//   row_base = starts[e] + y*kMTile      — the A row base, from the GROUP TABLE
//   m_valid  = min(kMTile, counts[e] - y*kMTile)   — real rows of this tile
//   m_valid <= 0                          — exit (empty expert, or past its block)
//
// ⚠️ WHY `starts[e]` AND NOT `z * kMTile`: the grouped buffer is COMPACT
// (`dsv41_route_gather_rows` writes the `n_assign` gathered rows back to back,
// with no per-expert padding), so expert e's first row sits at `starts[e]`, and
// the tile's rows are `[starts[e] + y*kMTile, +m_valid)`. `active`/`n_active` is
// why the grid does not have to be n_routed deep: only the ~topk*m live experts
// are listed, and the caller already knows `n_assign = m*topk >= n_active`
// (every live expert owns at least one assignment), so the z bound is static
// (CUDA-graph friendly — no D2H, no host-side scan of `counts`).
//
// ⚠️ EARLY EXIT IS BEFORE tc_alloc: an out-of-range CTA does one uniform load
// (`*n_active`) plus one branch, and touches no tensor memory. That matters
// because the plan's naive `grid.z = n_routed` would launch ~10x more CTAs than
// there are live experts at decode shapes (n_routed = 384 vs n_assign <= 36).
//
// NUMERIC DOMAIN (identical to the dense arm, element for element): the K loop
// is the SAME atom-ascending walk with the SAME two-kind::f8f6f4 MMAs per atom
// and the SAME `C_accum += C_local * sa * sb` fold at the SAME point, so an
// output element's f32 chain is the one the dense arm would produce for that
// (activation row, expert, column) triple. The row index only selects WHICH
// contiguous A row is read (the gather is byte-verbatim, see
// dsv41_route.cu's NUMERIC DOMAIN note) and WHICH out cell is written (the
// scatter is a pure copy). Masked rows contribute nothing: their A chunks and
// their per-32-block scales are ZERO-filled in smem, and their accumulator is
// never written back, so no masked lane can reach a live output cell.
//
// The epilogue keeps mode 0 (plain store) and mode 1 (gate/up clamp) only. Mode
// 2 (multiply by `row_weight[row]`) and mode 3 (accumulate into `out`) are
// per-(row, slot) concepts — `row_weight` is indexed by the MODEL row and every
// slot accumulates into the same `out` cell, while a grouped row's output cell
// belongs to ONE (row, slot) pair whose weight lives in the ungrouped table.
// Rather than index the WRONG row (a silent wrong answer) the launcher REFUSES
// them and the grouped pipeline applies the weight after the scatter.
__global__ void __launch_bounds__(kThreads) e4m3_gemm_grouped_kernel(
    const uint8_t* __restrict__ a,        // [n_assign, k] e4m3, GROUPED row order
    const float* __restrict__ a_scale,    // [n_assign, k/32] f32 powers of two
    float* __restrict__ out,              // [n_assign, n_total], grouped row order
    const int* __restrict__ active,       // [..] live expert ids, ascending
    const int* __restrict__ n_active,     // [1] live length of `active`
    const int* __restrict__ counts,       // [n_experts] rows per expert
    const int* __restrict__ starts,       // [n_experts + 1] exclusive prefix sum
    int n_experts, int n_total, int k, int b_split, int epi_mode, float limit,
    // The four B planes, derived per CTA from `base + e * stride` exactly as the
    // dense arm derives them from `base + ids[slot] * stride`. There is no
    // direct-pointer path here: the expert ALWAYS comes from the group table.
    const uint8_t* __restrict__ b_base, long b_stride,
    const uint8_t* __restrict__ bs_base, long bs_stride,
    const uint8_t* __restrict__ bh_base, long bh_stride,
    const uint8_t* __restrict__ bhs_base, long bhs_stride) {
    // ---- 0. group-table lookup + mask (the whole point of this arm) -------
    if ((int)blockIdx.z >= *n_active) return;  // compact list exhausted
    const int e = active[blockIdx.z];
    if (e < 0 || e >= n_experts) return;  // belt-and-braces (the table is ours)
    const int m_off = (int)blockIdx.y * kMTile;
    const int m_valid = min(kMTile, counts[e] - m_off);
    if (m_valid <= 0) return;  // empty expert, or a tile past its row block
    const int row_base = starts[e] + m_off;  // first grouped row of this tile

    const uint8_t* b_use = b_base + e * (size_t)b_stride;
    const uint8_t* bsc_use = bs_base + e * (size_t)bs_stride;
    const uint8_t* bhi_use = bh_base + e * (size_t)bh_stride;
    const uint8_t* bhs_use = bhs_base + e * (size_t)bhs_stride;

    const int n_base = (int)blockIdx.x * kNTile;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kbytes = k >> 1;  // PACKED fp4 weight bytes per row
    const int nk_blk = k >> 5;  // 32-element blocks == scale columns

    __shared__ Smem s;

    // ---------------------------------------------------------- tmem alloc
    if (warp == 0) {
        tc_alloc(&s.tmem_base, kTmemCols);
        tc_relinquish();
    }
    if (tid == 0) mbar_init(&s.mbar, 1);
    __syncthreads();

    const uint32_t tmem_base = s.tmem_base;
    const uint32_t d0_col = tmem_base;           // C_local, sub-block 0 (64 cols)
    const uint32_t d1_col = tmem_base + kDCols;  // C_local, sub-block 1 (64 cols)

    // --------------------------------------------------- register accumulator
    // One f32 chain per (m, n) of the tile, ascending K; `m` is the row WITHIN
    // the tile, so a masked row's chain starts and stays at 0.
    float acc[kNTile];
#pragma unroll
    for (int c = 0; c < kNTile; ++c) acc[c] = 0.f;

    // ------------------------------------------------------------- K loop
    uint32_t phase = 0;
    for (int k0 = 0; k0 < k; k0 += kStageAtoms * kAtomK) {
        const int natoms = min(kStageAtoms, (k - k0 + kAtomK - 1) / kAtomK);

        // ---- 1. stage the A operand (e4m3), MASKED -------------------------
        // Identical to the dense arm's chunk-per-(atom, row, sub, kb) walk, with
        // the row guarded twice: `m < m_valid` keeps the masked lanes away from
        // memory (their smem chunk stays the zero it was initialised to), and
        // `kk + 16 <= k` is the dense arm's own K tail guard.
        for (int c = tid; c < kStageAtoms * kMTile * 2 * kSubs; c += kThreads) {
            const int atom = c / (kMTile * 2 * kSubs);
            if (atom >= natoms) continue;
            const int r = c % (kMTile * 2 * kSubs);
            const int m = r >> 2;  // row within this CTA's M tile
            const int q = r & 3;   // chunk within the atom: 2 subs x 2 kb
            const int sub = q >> 1, kb = q & 1;
            const size_t kk = (size_t)k0 + (size_t)atom * kAtomK + sub * kSubK + kb * 16;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (m < m_valid && kk + 16 <= (size_t)k)
                val = ld_uint4_a16(a + (size_t)(row_base + m) * k + kk);
            *reinterpret_cast<uint4*>(s.a + (atom * kSubs + sub) * kASubBytes +
                                      e4x_off(m, kb)) = val;
        }

        // ---- 2. stage + expand the B operand (packed fp4 -> unpacked) ------
        // Unchanged from the dense arm: the B side is ONE expert's weight from
        // `b_use`/`bhi_use`, selected by `b_split` (gate|up).
        for (int c = tid; c < kStageAtoms * kNTile * kSubs; c += kThreads) {
            const int atom = c / (kNTile * kSubs);
            if (atom >= natoms) continue;
            const int r = c % (kNTile * kSubs);
            const int n = r >> 1, sub = r & 1;
            const int n_glob = n_base + n;
            const size_t ko = (size_t)(k0 + atom * kAtomK + sub * kSubK) >> 1;
            uint4 o0 = make_uint4(0, 0, 0, 0), o1 = make_uint4(0, 0, 0, 0);
            if (n_glob < n_total && ko + 16 <= (size_t)kbytes) {
                const uint8_t* src_base = b_use;
                int row = n_glob;
                if (b_split >= 0 && n_glob >= b_split) {
                    src_base = bhi_use;
                    row = n_glob - b_split;
                }
                if (row >= 0) {
                    const uint4 p = ld_uint4_a16(src_base + (size_t)row * kbytes + ko);
                    uint32_t ev[4], od[4];
                    e4x_expand(p.x, ev[0], od[0]);
                    e4x_expand(p.y, ev[1], od[1]);
                    e4x_expand(p.z, ev[2], od[2]);
                    e4x_expand(p.w, ev[3], od[3]);
                    o0.x = e4x_ilv_lo(ev[0], od[0]);
                    o0.y = e4x_ilv_hi(ev[0], od[0]);
                    o0.z = e4x_ilv_lo(ev[1], od[1]);
                    o0.w = e4x_ilv_hi(ev[1], od[1]);
                    o1.x = e4x_ilv_lo(ev[2], od[2]);
                    o1.y = e4x_ilv_hi(ev[2], od[2]);
                    o1.z = e4x_ilv_lo(ev[3], od[3]);
                    o1.w = e4x_ilv_hi(ev[3], od[3]);
                }
            }
            uint8_t* dst = s.b + (atom * kSubs + sub) * kBSubBytes;
            *reinterpret_cast<uint4*>(dst + e4x_off(n, 0)) = o0;
            *reinterpret_cast<uint4*>(dst + e4x_off(n, 1)) = o1;
        }

        // ---- 3. stage this stage's scales (once, coalesced), MASKED --------
        // A scales: masked rows read 0.f (so a masked row's fold is 0 * partial =
        // 0 and can never leak a real value into a live accumulator). B scales
        // are the expert's own rows, unchanged.
        for (int c = tid; c < kSubsPerStage * kMTile; c += kThreads) {
            const int sub = c / kMTile;
            const int m = c % kMTile;
            const int bb = (k0 >> 5) + sub;  // global 32-block of this sub
            float v = 0.f;
            if (m < m_valid && bb < nk_blk)
                v = a_scale[(size_t)(row_base + m) * nk_blk + bb];
            s.sa[sub][m] = v;
        }
        for (int c = tid; c < kSubsPerStage * kNTile; c += kThreads) {
            const int sub = c / kNTile;
            const int n = c % kNTile;
            const int n_glob = n_base + n;
            const int bb = (k0 >> 5) + sub;
            float v = 0.f;
            if (n_glob < n_total && bb < nk_blk) {
                const uint8_t* sc = bsc_use;
                int row = n_glob;
                if (b_split >= 0 && n_glob >= b_split) {
                    sc = bhs_use;
                    row = n_glob - b_split;
                }
                if (row >= 0) v = ue8m0_to_f(sc[(size_t)row * nk_blk + bb]);
            }
            s.sb[sub][n] = v;
        }

        // publish every smem write to the async proxy (the MMA reads it there)
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        // ---- 4. per atom: 2 MMAs (one per 32-block) + ONE commit/wait + fold
        for (int atom = 0; atom < natoms; ++atom) {
            if (tid == 0) {
                const uint32_t id = e4x_make_idesc();
                const uint64_t da0 =
                    e4x_make_desc(smem_addr(s.a + (atom * kSubs + 0) * kASubBytes));
                const uint64_t db0 =
                    e4x_make_desc(smem_addr(s.b + (atom * kSubs + 0) * kBSubBytes));
                const uint64_t da1 =
                    e4x_make_desc(smem_addr(s.a + (atom * kSubs + 1) * kASubBytes));
                const uint64_t db1 =
                    e4x_make_desc(smem_addr(s.b + (atom * kSubs + 1) * kBSubBytes));
                e4x_mma(d0_col, da0, db0, id, 0u);
                e4x_mma(d1_col, da1, db1, id, 0u);
                tc_commit(&s.mbar);  // ONE arrival covers BOTH MMAs of the atom
            }
            mbar_wait(&s.mbar, phase);
            phase ^= 1u;

            const int mrow = warp * 32 + lane;  // this thread's row in the M tile
#pragma unroll
            for (int sub = 0; sub < kSubs; ++sub) {
                const int ssub = atom * kSubs + sub;  // stage-local sub index
                const uint32_t dcol = (sub == 0) ? d0_col : d1_col;
                const float sa = s.sa[ssub][mrow];  // [row, 32-block] activation
#pragma unroll
                for (int c0 = 0; c0 < kNTile; c0 += 16) {
                    uint32_t v[16];
                    tc_ld_x16((((uint32_t)(warp * 32)) << 16) | (dcol + c0), v);
                    tc_wait_ld();
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        acc[c0 + i] += __uint_as_float(v[i]) * (sa * s.sb[ssub][c0 + i]);
                    }
                }
            }

            tc_fence_before_thread_sync();
            __syncthreads();
            tc_fence_after_thread_sync();
        }
    }

    // ------------------------------------------------------------ epilogue
    // Masked rows are SKIPPED: a grouped position that no assignment reached is
    // never a `perm_map` target, so the scatter cannot read it, and the cells
    // past `starts[e] + counts[e]` belong to another expert's tile (or to the
    // tail of the grouped buffer) — writing them would race that expert's CTA.
#pragma unroll
    for (int c = 0; c < kNTile; ++c) {
        const int m = warp * 32 + lane;
        if (m >= m_valid) continue;
        const int row = row_base + m;  // GROUPED row of this output element
        const int col = n_base + c;
        if (col >= n_total) continue;
        float x = acc[c];
        if (epi_mode == 1) {  // gate/up clamps (training convention)
            if (limit > 0.f) {
                if (col < b_split) x = fminf(x, limit);           // gate
                else x = fminf(fmaxf(x, -limit), limit);          // up
            }
        }
        out[(size_t)row * n_total + col] = x;
    }

    __syncthreads();
    if (warp == 0) tc_dealloc(tmem_base, kTmemCols);
}

// Launcher for the grouped arm. Contract (checked here, not in the kernel):
//   k % kAtomK == 0            a stage never carries a half atom
//   n_total % kNTile == 0      the MMA's N is the CTA's N
//   epi_mode in {0, 1}         see the kernel's note on modes 2/3
//   a / a_scale / every B base 16-byte aligned, B strides multiples of 16
// There is DELIBERATELY no `rows % kMTile == 0` term here: that requirement is
// exactly what this arm removes. The tile's real row count is `m_valid`, which
// is derived IN-KERNEL from `counts[e]`.
//
// `m_cap` is the per-expert row capacity the `active` list was built against
// (`grp_m_cap` on the Rust side); the launcher turns it into grid.y, so the
// tile-per-expert formula lives in ONE place. `n_assign` (= m*topk) is the z
// bound: it is >= `n_active` by construction, so no live expert can be missed,
// and a z past the live list reads `*n_active` and exits.
inline cudaError_t e4x_launch_gemm_grouped(
    const uint8_t* a, const float* a_scale, float* out, const int* active, const int* n_active,
    const int* counts, const int* starts, int n_experts, int n_assign, int m_cap, int n_total,
    int k, int b_split, int epi_mode, float limit, const uint8_t* b_base, long b_stride,
    const uint8_t* bs_base, long bs_stride, const uint8_t* bh_base, long bh_stride,
    const uint8_t* bhs_base, long bhs_stride, cudaStream_t stream) {
    if (a == nullptr || a_scale == nullptr || out == nullptr || active == nullptr ||
        n_active == nullptr || counts == nullptr || starts == nullptr || b_base == nullptr ||
        bs_base == nullptr)
        return cudaErrorInvalidValue;
    if (n_total <= 0 || k <= 0 || n_experts <= 0 || n_assign <= 0 || m_cap <= 0)
        return cudaErrorInvalidValue;
    if (k % kAtomK != 0 || n_total % kNTile != 0) return cudaErrorInvalidValue;
    if (epi_mode != 0 && epi_mode != 1) return cudaErrorInvalidValue;  // see the kernel note
    // `b_split >= 0` is the two-pool (gate|up) form: the SECOND pool is then a
    // hard requirement, not an option.
    if (b_split >= 0 && (bh_base == nullptr || bhs_base == nullptr))
        return cudaErrorInvalidValue;
    const auto al16 = [](const void* p) { return ((uintptr_t)p & 0xF) == 0; };
    if (!al16(a) || !al16(a_scale) || !al16(b_base) || !al16(bs_base)) return cudaErrorInvalidValue;
    // The B plane's expert stride is `inter * k/2` bytes; a stride that is not a
    // 16-byte multiple would silently misplace the uint4 rows.
    if ((b_stride % 16) != 0 || (bs_stride % 16) != 0) return cudaErrorInvalidValue;
    const int m_tiles = (m_cap + kMTile - 1) / kMTile;
    const dim3 grid((unsigned)(n_total / kNTile), (unsigned)m_tiles, (unsigned)n_assign);
    e4m3_gemm_grouped_kernel<<<grid, kThreads, 0, stream>>>(
        a, a_scale, out, active, n_active, counts, starts, n_experts, n_total, k, b_split,
        epi_mode, limit, b_base, b_stride, bs_base, bs_stride, bh_base, bh_stride, bhs_base,
        bhs_stride);
    return cudaGetLastError();
}

// "e4m3 activation x fp4 weight, GROUP-INDEXED MASKED M=128 tile GEMM" — the
// form `moe_rows`' per-(row, slot) routing needs (see the block comment above).
// Returns 0 (and does nothing) while EITHER runtime gate is OFF, so a caller can
// invoke it unconditionally and keep its proven per-(row, slot) path — the same
// contract as every other arm in this file.
//
// Runtime gates (both read ONCE per process, strict first-char '1', default OFF
// — a per-call getenv is a CUDA-graph capture hazard, exactly as in `e4x`):
//   DSV41_EXPERT_TCGEN05_E4M3 — the e4m3-activation tcgen05 family (this arm
//                               shares it with the two arms above: which of them
//                               a step runs is a LAUNCH-SHAPE decision, not a
//                               numeric-format one);
//   DSV41_EXPERT_GROUPED      — the grouped (permuted) routed layout.
// The AND is deliberate: this kernel CONSUMES `counts`/`starts`/`active`/
// `n_active`, so without the grouped layout it would index tables the step never
// built. Both gates must be armed before a single CTA runs.
extern "C" int dsv41_expert_gemm_e4m3_grouped(
    const uint8_t* a, const float* a_scale, float* out, const int* active, const int* n_active,
    const int* counts, const int* starts, int n_experts, int n_assign, int m_cap, int n_total,
    int k, int b_split, int epi_mode, float limit, const uint8_t* b_base, long b_stride,
    const uint8_t* bs_base, long bs_stride, const uint8_t* bh_base, long bh_stride,
    const uint8_t* bhs_base, long bhs_stride, cudaStream_t stream) {
    static const int enabled = [] {
        const char* g = getenv("DSV41_EXPERT_TCGEN05_E4M3");
        const char* h = getenv("DSV41_EXPERT_GROUPED");
        return (g != nullptr && g[0] == '1' && h != nullptr && h[0] == '1') ? 1 : 0;
    }();
    if (!enabled) return 0;
    const cudaError_t rc = e4x_launch_gemm_grouped(a, a_scale, out, active, n_active, counts,
                                                   starts, n_experts, n_assign, m_cap, n_total, k,
                                                   b_split, epi_mode, limit, b_base, b_stride,
                                                   bs_base, bs_stride, bh_base, bh_stride,
                                                   bhs_base, bhs_stride, stream);
    (void)cudaGetLastError();  // never fail the step: the fallback path is correctness
    return (int)rc;
}

}  // namespace e4x
}  // namespace tc5

#endif  // DSV41_TCGEN05_GATEUP_E4M3_SKELETON
