// =============================================================================
// tests_bs_impulse.cu — IMPULSE-RESPONSE + liveness instrument for
//     tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale[.scale_vec::1X]  (sm_103a)
//
// WHY THIS SHAPE
// -----------------------------------------------------------------------------
// A dense GEMM vs a golden only says "wrong". This harness drives the MMA with
// DELTA inputs so that every unknown convention turns into an OBSERVABLE response,
// one link at a time:
//
//   mode            stimulus                                  expected if our
//                                                             convention is right
//   ------------    ---------------------------------------   -------------------
//   selfcheck       poison TMEM D cols with 1000+row, NO MMA  read-back == poison
//                                                                 (proves the read
//                                                                  path + that D
//                                                                  cols are ours)
//   impulse m0 k0   A = ONE e4m3 1.0 at (m0,k0), B all 1.0    row m0 = 1.0 x128
//   impulseB n0 k0  B = ONE e2m1 1.0 at (n0,k0), A all 1.0    col n0 = 1.0 x128
//   const           A = B = 1.0 codes, SF = 127 (1.0)          D == 128 everywhere
//   sfprobe         A = B = 1.0 codes, SF byte j = 127+j      D == sum_k 4^j(k)
//                                                                 = 2720 everywhere
//   random          dense random A/B, SF = 127                == host double golden
//   random_sf       dense random A/B, per-K-block scales      == host double golden
//
// A is e4m3 (activation, M rows), B is e2m1-unpacked (weight, N rows), SF is one
// uint32 per row whose byte j is the ue8m0 scale of K-block j — exactly the
// production/micro-harness convention.
//
// CONFIG UNDER TEST (the canonical set we believe is right)
//   smem A/B layout (canonical UMMA K-major, SWIZZLE_NONE):
//     idx(m,kk) = (kk>>5)*4096 + (m>>3)*256 + (m&7)*16 + ((kk&31)>>4)*128 + (kk&15)
//   smem descriptor : lbo = 8 (16B units) = 128 B, sbo = 16 (16B units) = 256 B,
//                     layout_type = 0, version = 1
//   K-block advance : ki * 4096 B == ki * 256 (16B units)
//   idesc           : a_fmt = 0 (E4M3), b_fmt = 5 (E2M1), n_dim = N/8,
//                     scale_format = UE8M0, m_dim = M/16,
//                     a_sf_id = ki<<29, b_sf_id = ki<<4
//   SF path         : tcgen05.st, 32x32b.x4, identical content in all four lane
//                     partitions (the "PH0-verified" path)
//
// THREE tcgen05 FENCES are present (they were missing in tests_bs_mma_micro.cu):
//   F1  tcgen05.alloc -> generic smem read of the TMEM base   (before/after sync pair)
//   F2  thread sync point -> async tcgen05 op                (tcgen05.fence::after_thread_sync)
//   F3  mbarrier.init visibility: fence.mbarrier_init.release.cluster BEFORE the barrier
// -DIMP_NO_FENCES=1 removes them again, so the fence hypothesis can be A/B-ed.
//
// BUILD + RUN
//   nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//        -o /tmp/bsimp kernels/cuda/tests_bs_impulse.cu
//   CUDA_VISIBLE_DEVICES=6 /tmp/bsimp                    # full (m0,k0) sweep
//   CUDA_VISIBLE_DEVICES=6 /tmp/bsimp selfcheck
//   CUDA_VISIBLE_DEVICES=6 /tmp/bsimp const
//   CUDA_VISIBLE_DEVICES=6 /tmp/bsimp sfprobe
//   CUDA_VISIBLE_DEVICES=6 /tmp/bsimp random 1
//   CUDA_VISIBLE_DEVICES=6 /tmp/bsimp random_sf 1
//
// ⚠️ Use the explicit -gencode form: `-arch=sm_103a` alone is silently downgraded
//    to sm_103 on some nvcc builds and ptxas then rejects every tcgen05 op.
// ⚠️ `misaligned address` is STICKY on this part: it poisons the CUDA context, so
//    every later call fails too. Each case here detects that and calls
//    cudaDeviceReset() so one bad case cannot fake the results of the others.
// =============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------- geometry
constexpr int BM = 128;        // M tile (8 x 16-row atoms)
constexpr int BN = 128;        // N tile
constexpr int BK = 128;        // K per k-iteration (4 x K32 atoms)
constexpr int NKB = BK / 32;   // 4 K-blocks of 32 == scale_vec::1X granularity

constexpr uint8_t kE4M3One = 0x38;  // e4m3 1.0
constexpr uint8_t kE2M1One = 0x02;  // e2m1 code 2 == 1.0
constexpr uint32_t kSFOne = 0x7F7F7F7Fu;  // every byte 127 == 2^0 == 1.0

#ifdef IMP_NO_FENCES
constexpr bool kFencesDefault = false;
#else
constexpr bool kFencesDefault = true;
#endif

// =============================================================================
// device primitives — verbatim from tests_bs_mma_micro.cu / the PH0 probe
// =============================================================================
__device__ __forceinline__ void hw_tc_alloc(uint32_t *dst, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(dst)),
                 "r"(ncols));
}
__device__ __forceinline__ void hw_tc_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void hw_tc_dealloc(uint32_t tmem, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(ncols));
}

// CANONICAL UMMA K-major SWIZZLE_NONE index for (row, kk) in a 128x128-byte tile.
__device__ __forceinline__ int hw_canon_idx(int row, int kk) {
    return (kk >> 5) * 4096 + (row >> 3) * 256 + (row & 7) * 16 + (((kk & 31) >> 4) * 128) +
           (kk & 15);
}

// PACKED fp4 (e2m1) B-tile index: element k of row n.  (v1 guess, kept for A/B)
__device__ __forceinline__ int hw_pack_idx(int row, int k) {
    return (row >> 3) * 512 + (k >> 6) * 128 + (row & 7) * 16 + ((k & 63) >> 1);
}

// =============================================================================
// MEASURED packed-fp4 read geometry (see the DERIVATION block at the top).
//
// The hardware reads, per K=32 MMA, TWO "atoms" of EIGHT bytes per row; each atom
// is 16 fp4 elements (2/byte).  The atoms of one row sit at (row%8)*16 + {0, LBO}
// and the row-groups are SBO apart, i.e. the SAME skeleton as the canonical e4m3
// layout -- only the atom (16 elements) is 8 bytes wide instead of 16, and the
// 16-byte chunk holding it is only half used.
//
//   chunk c (c = 0/1, at +c*128) holds logical elements 32*kb + 16*c + [0,16)
//   packed byte b  (b = 0..7)  of that chunk holds elements (2b, 2b+1)
//
// p = packed byte index of the row, p in [0,64) == elements 2p, 2p+1:
//   kb  = p>>4      (K-block 0..3, one 4096 B region each)
//   c   = (p>>3)&1  (which of the two atoms -> +128 B)
//   b   = p&7       (byte inside the 8-byte atom)
// =============================================================================
__device__ __forceinline__ int hw_pack_canon_idx(int row, int p, int kbs) {
    return (p >> 4) * kbs + (row >> 3) * 256 + (row & 7) * 16 + ((p >> 3) & 1) * 128 + (p & 7);
}

// Candidate D (main-agent proposal): DENSE 64 B row + SWIZZLE_64B, chunk c XOR (row&3).
__device__ __forceinline__ int hw_pack_sw64_idx(int row, int p) {
    return (row >> 3) * 512 + (row & 7) * 64 + (((((p >> 4) ^ (row & 3))) & 3) << 4) + (p & 15);
}

// SW128 + OFFICIAL-TMA-shaped packed data: each 16 B swizzled slot carries 8 packed
// bytes (16 fp4 elements) in its FIRST HALF -- exactly what TMA's
// CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B writes (16 fp4 elements per 16 B container).
// p = packed byte of the row [0,64): slot c = p>>3, byte in slot = p&7.
__device__ __forceinline__ int hw_pack_sw128_idx(int row, int p) {
    const int c = (p >> 3) & 7;
    return (row >> 3) * 1024 + (row & 7) * 128 + (((c ^ (row & 7)) & 7) << 4) + (p & 7);
}

// SW128 (CU_TENSOR_MAP_SWIZZLE_128B) index — the layout TileLang's TMA actually
// produces for the A operand (moe_bs_up_tl.cu:115 uses lbo=1 sbo=64 layout=2).
__device__ __forceinline__ int hw_sw128_idx(int row, int kk) {
    return (row >> 3) * 1024 + (row & 7) * 128 + ((((kk >> 4) ^ (row & 7))) << 4) + (kk & 15);
}

__device__ __forceinline__ void hw_tc_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                          uint32_t idesc, uint32_t sfa_tmem, uint32_t sfb_tmem,
                                          uint32_t enable_d, int sv1x) {
    if (sv1x) {
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
            "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
            : "memory");
    } else {
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t}" ::"r"(d_tmem),
            "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
            : "memory");
    }
}

__device__ __forceinline__ void hw_tc_commit(void *mbar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
            (uint32_t)__cvta_generic_to_shared(mbar))
        : "memory");
}

// smem descriptor: start_addr[0:14) | lbo[16:30) | sbo[32:46) | version=1[46] | layout[61:64)
__device__ __forceinline__ uint64_t hw_make_desc(const void *smem_ptr, uint32_t lbo_16B,
                                                 uint32_t sbo_16B, uint32_t layout) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)(lbo_16B & 0x3FFF) << 16;
    d |= (uint64_t)(sbo_16B & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)(layout & 0x7) << 61;
    return d;
}

// idesc: b_sf_id[4:6) | a_fmt[7:10) | b_fmt[10:13) | n_dim[17:23) | sf_fmt[23)
//        | m_dim[24:29) | a_sf_id[29:31)
__device__ __forceinline__ uint32_t hw_make_idesc(int m, int n, int a_fmt, int b_fmt, int sf_id) {
    uint32_t d = 0;
    d |= (uint32_t)(sf_id & 3) << 4;
    d |= (uint32_t)(a_fmt & 7) << 7;
    d |= (uint32_t)(b_fmt & 7) << 10;
    d |= (uint32_t)((n >> 3) & 63) << 17;
    d |= (uint32_t)1 << 23;
    d |= (uint32_t)((m >> 4) & 31) << 24;
    d |= (uint32_t)(sf_id & 3) << 29;
    return d;
}

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
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(bar)),
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
// kernel — ONE case: stage -> (poison) -> SF -> MMA chain (4 x K32) -> read D
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 1) bs_probe_kernel(
    const uint8_t *__restrict__ A,     // [128][128] e4m3 codes (row-major, logical (m,kk))
    const uint8_t *__restrict__ B,     // [128][128] e2m1 codes, 1 code/byte (logical (n,kk))
    const uint32_t *__restrict__ SFA,  // [128] word per row; byte j = K-block j scale
    const uint32_t *__restrict__ SFB,  // [128]
    int do_mma, int do_poison, int use_fences, int sv1x, int layout_mode, int bspack, int lbo16,
    int sbo16, int blbo16, int bsbo16, int blay, int badv, int kbs, int imp_off_a, int imp_off_b,
    float *__restrict__ D, uint64_t *__restrict__ dbg) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(1024) uint8_t s_a[BM * BK];
    __shared__ __align__(1024) uint8_t s_b[BN * BK];
    __shared__ __align__(8) uint64_t s_mbar;
    __shared__ uint32_t s_tmem_d;
    __shared__ uint32_t s_tmem_sf;

    if (warp == 0) {
        hw_tc_alloc(&s_tmem_d, 128);  // D: 128 columns (128 lanes x 128 cols)
        hw_tc_alloc(&s_tmem_sf, 32);  // SF: 8 used (SFA +0..3, SFB +4..7)
        hw_tc_relinquish();
    }
    // FENCE F1: tcgen05.alloc writes the TMEM base into smem; the generic read
    // below must be ordered against it.
    if (use_fences) tc_fence_before_thread_sync();
    __syncthreads();
    if (use_fences) tc_fence_after_thread_sync();
    const uint32_t D_tmem = s_tmem_d;
    const uint32_t SF_tmem = s_tmem_sf;

    if (tid == 0) {
        mbar_init(&s_mbar, 1);
        // FENCE F3: mbarrier-init visibility uses the dedicated fence, BEFORE the
        // first barrier.
        if (use_fences) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // ---- (1) stage A and B in the selected layout
    for (int i = tid; i < BM * BK; i += 128) {
        const int m = i >> 7, kk = i & 127;
        s_a[layout_mode ? hw_sw128_idx(m, kk) : hw_canon_idx(m, kk)] = A[(size_t)m * BK + kk];
    }
    for (int i = tid; i < BN * BK; i += 128) {
        const int n = i >> 7, kk = i & 127;
        s_b[layout_mode ? hw_sw128_idx(n, kk) : hw_canon_idx(n, kk)] = B[(size_t)n * BK + kk];
    }
    if (bspack) {
        // fp4 packing: 2 K-elements per byte (low nibble = even k, high = odd k).
        // bspack selects the WRITE FORMULA of the candidate geometry (see main()).
        for (int i = tid; i < BN * BK; i += 128) s_b[i] = 0;
        __syncthreads();
        for (int i = tid; i < BN * 64; i += 128) {
            const int n = i >> 6, kp = i & 63;
            const uint8_t lo = (uint8_t)(B[(size_t)n * BK + 2 * kp] & 0xF);
            const uint8_t hi = (uint8_t)(B[(size_t)n * BK + 2 * kp + 1] & 0xF);
            const int idx = (bspack == 2) ? hw_pack_canon_idx(n, kp, kbs)
                          : (bspack == 3) ? hw_pack_sw64_idx(n, kp)
                          : (bspack == 4) ? hw_pack_sw128_idx(n, kp)
                                          : hw_pack_idx(n, 2 * kp);
            s_b[idx] = (uint8_t)(lo | (hi << 4));
        }
    }
    // optional: place the impulse at a RAW smem byte offset (physical probe)
    if (imp_off_a >= 0 && tid == 0) s_a[imp_off_a] = kE4M3One;
    if (imp_off_b >= 0 && tid == 0) s_b[imp_off_b] = kE2M1One;

    // ---- (2) poison the D TMEM columns with a per-row known pattern. If the MMA
    //          never runs, the read-back below must still show this pattern (that
    //          is the read path's self-check); if it runs with enable_d=0 on ki=0
    //          the pattern is fully replaced.
    if (do_poison) {
        const uint32_t p = __float_as_uint((float)(1000 + warp * 32 + lane));
        for (int q = 0; q < 128; q += 4)
            tc_st_x4(((uint32_t)(warp * 32) << 16) | (D_tmem + q), p, p, p, p);
        tc_wait_st();
    }

    // ---- (3) scale factors -> TMEM SF columns (the "verified" tcgen05.st path:
    //          lane l gets SFA[32*j+l] in column j, identical in every partition)
    {
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

    // ---- (4) publish to the async proxy and issue the MMA chain
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (use_fences) tc_fence_before_thread_sync();
    __syncthreads();
    if (use_fences) tc_fence_after_thread_sync();

    if (do_mma) {
        if (warp == 1 && lane == 0) {
            const uint64_t a_desc_base = layout_mode ? hw_make_desc(s_a, 1, 64, 2)
                                                     : hw_make_desc(s_a, (uint32_t)lbo16, (uint32_t)sbo16, 0);
            // bspack: choose the descriptor family of the packed candidate.
            //   1 = v1 guess (canonical lbo=8 sbo=16)
            //   2 = MEASURED canonical packed (lbo=8 sbo=16, atoms 8 B at +0/+LBO)
            //   3 = candidate D: dense SW64 (lbo=1 sbo=32 layout=4)
            const uint64_t b_desc_base =
                layout_mode ? hw_make_desc(s_b, 1, 64, 2)
                : (bspack == 3) ? hw_make_desc(s_b, 1, 32, 4)
                : bspack ? hw_make_desc(s_b, (uint32_t)blbo16, (uint32_t)bsbo16, (uint32_t)blay)
                         : hw_make_desc(s_b, (uint32_t)lbo16, (uint32_t)sbo16, 0);
            dbg[0] = a_desc_base;
            dbg[1] = b_desc_base;
            for (int ki = 0; ki < NKB; ++ki) {
                const uint32_t idesc = hw_make_idesc(BM, BN, /*a_fmt=*/0, /*b_fmt=*/5, ki);
                const uint64_t a_desc = a_desc_base + (uint64_t)(ki * (layout_mode ? 2 : 256));
                // K-block descriptor advance, in 16 B units (reg32_[0] += offset>>4).
                //   bspack==2: the 4 K-block regions are 4096 B apart (ki*256 units)
                //   bspack==3: one K-block == one 16 B chunk        (ki*1   unit)
                //   bspack==1: the old v1 guess
                const uint64_t b_adv = badv >= 0
                    ? (uint64_t)(ki * badv)
                    : bspack == 2 ? (uint64_t)(ki * 256)
                    : bspack == 3 ? (uint64_t)(ki * 1)
                    : bspack == 4 ? (uint64_t)(ki * 2)
                    : bspack      ? (uint64_t)((ki / 2) * 256 + (ki % 2) * 8)
                                  : (uint64_t)(ki * (layout_mode ? 2 : 256));
                const uint64_t b_desc = b_desc_base + b_adv;
                const uint32_t enable_d = (ki == 0) ? 0u : 1u;
                if (ki == 0) dbg[2] = idesc;
                if (ki == NKB - 1) dbg[3] = idesc;
                hw_tc_mma(D_tmem, a_desc, b_desc, idesc, SF_tmem + 0, SF_tmem + 4, enable_d, sv1x);
            }
            hw_tc_commit(&s_mbar);
        }
        // how long does the wait actually block? (0-ish => the wait is a no-op)
        const unsigned long long t0 = clock64();
        mbar_wait(&s_mbar, 0);
        const unsigned long long t1 = clock64();
        if (tid == 0) dbg[5] = t1 - t0;
    }
    tc_fence_before_thread_sync();
    __syncthreads();
    tc_fence_after_thread_sync();

    // ---- (5) read D back: D[m][n] at TMEM lane m, column D_tmem+n
    {
        const int row = warp * 32 + lane;
        for (int q = 0; q < 16; ++q) {
            uint32_t v[8];
            tc_ld_x8(((uint32_t)(warp * 32) << 16) | (D_tmem + 8 * q), v);
            tc_wait_ld();
            for (int i = 0; i < 8; ++i) D[(size_t)row * BN + 8 * q + i] = __uint_as_float(v[i]);
        }
    }
    __syncthreads();
    if (warp == 0) {
        hw_tc_dealloc(D_tmem, 128);
        hw_tc_dealloc(SF_tmem, 32);
    }
}

// =============================================================================
// host side
// =============================================================================
namespace {

uint32_t g_rng = 0x12345678u;
uint32_t xrand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

const float kTabE2M1[16] = {0.f,  0.5f, 1.f,  1.5f, 2.f,  3.f,  4.f,  6.f,
                            0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

double e4m3_to_d(uint8_t b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    double v = (e == 0) ? std::ldexp((double)m / 8.0, -6) : std::ldexp(1.0 + (double)m / 8.0, e - 7);
    return s ? -v : v;
}
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
inline uint8_t sf_byte(uint32_t w, int j) { return (uint8_t)((w >> (8 * j)) & 0xFF); }
inline double sf_val(uint8_t b) { return std::ldexp(1.0, (int)b - 127); }

struct Stim {
    std::vector<uint8_t> A, B;
    std::vector<uint32_t> SFA, SFB;
    std::string name;
};

uint8_t g_bcode = kE2M1One;  // B code byte (env BSB, e.g. 0x22 = BOTH nibbles 1.0)

void all_ones_codes(Stim &s) {
    s.A.assign((size_t)BM * BK, kE4M3One);
    s.B.assign((size_t)BN * BK, g_bcode);
    s.SFA.assign(BM, kSFOne);
    s.SFB.assign(BN, kSFOne);
}

void random_codes(Stim &s, bool varying_sf) {
    s.A.resize((size_t)BM * BK);
    s.B.resize((size_t)BN * BK);
    for (size_t i = 0; i < s.A.size(); ++i) {
        uint8_t v;
        do { v = (uint8_t)xrand(); } while (v == 0x7F || v == 0xFF);
        s.A[i] = v;
    }
    for (size_t i = 0; i < s.B.size(); ++i) s.B[i] = (uint8_t)(xrand() & 0xF);
    s.SFA.assign(BM, kSFOne);
    s.SFB.assign(BN, kSFOne);
    if (varying_sf) {
        for (int m = 0; m < BM; ++m) {
            uint32_t w = 0;
            for (int j = 0; j < NKB; ++j) w |= (uint32_t)(127 + ((m + j) % 5) - 2) << (8 * j);
            s.SFA[m] = w;
        }
        for (int n = 0; n < BN; ++n) {
            uint32_t w = 0;
            for (int j = 0; j < NKB; ++j) w |= (uint32_t)(127 + ((n + 2 * j) % 5) - 2) << (8 * j);
            s.SFB[n] = w;
        }
    }
}

// exact double golden over the DEQUANTIZED operands
std::vector<double> golden(const Stim &s) {
    std::vector<double> ref((size_t)BM * BN, 0.0);
    for (int m = 0; m < BM; ++m)
        for (int n = 0; n < BN; ++n) {
            double acc = 0;
            for (int kk = 0; kk < BK; ++kk) {
                const double a = e4m3_to_d(s.A[(size_t)m * BK + kk]);
                const double b = (double)kTabE2M1[s.B[(size_t)n * BK + kk] & 0xF];
                const double sa = sf_val(sf_byte(s.SFA[m], kk >> 5));
                const double sb = sf_val(sf_byte(s.SFB[n], kk >> 5));
                acc += a * b * sa * sb;
            }
            ref[(size_t)m * BN + n] = acc;
        }
    return ref;
}

const char *cuErrStr(cudaError_t e) { return cudaGetErrorString(e); }

}  // namespace

int main(int argc, char **argv) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("== tcgen05 mxf8f6f4 BLOCK_SCALE probe (impulse / liveness / parity) ==\n");
    printf("   device: %s cc=%d.%d   M=%d N=%d K=%d (4 x K32)\n", prop.name, prop.major,
           prop.minor, BM, BN, BK);
    printf("   canonical smem layout, desc(lbo=8,sbo=16,layout=0), K-advance=4096B, "
           "idesc(a_fmt=0,b_fmt=5,ki<<4,ki<<29)\n");
    printf("   tcgen05 fences: %s\n\n", kFencesDefault ? "ON (F1+F2+F3)" : "OFF (-DIMP_NO_FENCES)");

    std::string mode = (argc >= 2) ? argv[1] : "sweep";
    const int g_layout = getenv("BSLAYOUT") ? atoi(getenv("BSLAYOUT")) : 0;
    const int g_bspack = getenv("BSPACK") ? atoi(getenv("BSPACK")) : 0;
    // packed-B geometry knobs (all optional; defaults = the MEASURED canonical packed)
    const int g_blbo16 = getenv("BSLBO") ? atoi(getenv("BSLBO")) : 8;      // 16 B units
    const int g_bsbo16 = getenv("BSSBO") ? atoi(getenv("BSSBO")) : 16;     // 16 B units
    const int g_blay = getenv("BSLAY") ? atoi(getenv("BSLAY")) : 0;        // layout_type field
    const int g_badv = getenv("BSADV") ? atoi(getenv("BSADV")) : -1;       // per-ki advance, 16 B units
    const int g_kbs = getenv("BSKBS") ? atoi(getenv("BSKBS")) : 4096;      // packed K-block region stride
    if (getenv("BSB")) g_bcode = (uint8_t)strtoul(getenv("BSB"), nullptr, 0);
    printf("   layout=%s  B packing=%s  B code byte=0x%02x  (BSLAYOUT/BSPACK/BSB env)\n",
           g_layout ? "SW128" : "canonical", g_bspack ? "PACKED 2/byte" : "unpacked 1/byte",
           g_bcode);
    if (g_bspack)
        printf("   packed-B geom: write=%s  desc(lbo=%d sbo=%d layout=%d)  ki-advance=%d units%s\n",
               g_bspack == 2 ? "canonical-packed" : g_bspack == 3 ? "dense-SW64" : "v1",
               g_bspack == 3 ? 1 : g_blbo16, g_bspack == 3 ? 32 : g_bsbo16, g_bspack == 3 ? 4 : g_blay,
               g_badv >= 0 ? g_badv : (g_bspack == 2 ? 256 : g_bspack == 3 ? 1 : -1),
               g_badv >= 0 ? " (BSADV)" : "");

    float *dD = nullptr;
    uint64_t *dDbg = nullptr;
    if (cudaMalloc(&dD, (size_t)BM * BN * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dDbg, 6 * sizeof(uint64_t)) != cudaSuccess) {
        printf("[FATAL] cudaMalloc: %s\n", cuErrStr(cudaGetLastError()));
        return 2;
    }
    std::vector<float> D((size_t)BM * BN);
    std::vector<uint64_t> dbg(6);

    auto reset_ctx = [&]() {
        cudaDeviceReset();
        if (cudaMalloc(&dD, (size_t)BM * BN * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&dDbg, 6 * sizeof(uint64_t)) != cudaSuccess) {
            printf("[FATAL] re-alloc after reset failed: %s\n", cuErrStr(cudaGetLastError()));
            exit(3);
        }
    };

    // run one stimulus; returns false + the error on failure
    auto run = [&](const Stim &s, int do_mma, int do_poison, bool want_dbg, int lbo16 = 8,
                   int sbo16 = 16, int imp_off_a = -1, int imp_off_b = -1) -> bool {        static uint8_t *dA = nullptr, *dB = nullptr;
        static uint32_t *dSFA = nullptr, *dSFB = nullptr;
        if (!dA) {
            cudaMalloc(&dA, (size_t)BM * BK);
            cudaMalloc(&dB, (size_t)BN * BK);
            cudaMalloc(&dSFA, BM * 4);
            cudaMalloc(&dSFB, BN * 4);
        }
        cudaMemcpy(dA, s.A.data(), (size_t)BM * BK, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, s.B.data(), (size_t)BN * BK, cudaMemcpyHostToDevice);
        cudaMemcpy(dSFA, s.SFA.data(), BM * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dSFB, s.SFB.data(), BN * 4, cudaMemcpyHostToDevice);
        cudaGetLastError();
        bs_probe_kernel<<<1, 128>>>(dA, dB, dSFA, dSFB, do_mma, do_poison, kFencesDefault,
                                    /*sv1x=*/1, g_layout, g_bspack, lbo16, sbo16, g_blbo16,
                                    g_bsbo16, g_blay, g_badv, g_kbs, imp_off_a, imp_off_b, dD, dDbg);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("   [CUDA ERROR] %s\n", cuErrStr(e));
            reset_ctx();
            return false;
        }
        cudaMemcpy(D.data(), dD, (size_t)BM * BN * sizeof(float), cudaMemcpyDeviceToHost);
        if (want_dbg) cudaMemcpy(dbg.data(), dDbg, 6 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        return true;
    };

    auto report_vs_golden = [&](const std::vector<double> &ref) {
        double maxabs = 0, maxref = 0, maxd = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            maxref = std::fmax(maxref, std::fabs(ref[i]));
            maxd = std::fmax(maxd, std::fabs((double)D[i] - ref[i]));
        }
        maxabs = maxd;
        const double rel = maxabs / std::fmax(1e-3 * maxref, 1e-30);
        const double reln = maxabs / std::fmax(maxref, 1e-30);
        printf("   max|ref|=%.6g  max|D-ref|=%.6g  relerr(scaled)=%.4g  relerr/|ref|max=%.4g  %s\n",
               maxref, maxabs, rel, reln, rel < 1e-3 ? "[PASS]" : "[FAIL]");
        // show the worst element
        int wi = 0;
        double wd = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (std::fabs((double)D[i] - ref[i]) > wd) { wd = std::fabs((double)D[i] - ref[i]); wi = (int)i; }
        printf("   worst (m=%d,n=%d): D=%.6f ref=%.6f  (D[0][0]=%.6f ref[0][0]=%.6f, "
               "D[127][127]=%.6f ref=%.6f)\n",
               wi / BN, wi % BN, D[wi], ref[wi], D[0], ref[0], D[(size_t)BM * BN - 1],
               ref[(size_t)BM * BN - 1]);
        return rel < 1e-3;
    };

    int failures = 0;

    // ------------------------------------------------------------------ selfcheck
    if (mode == "selfcheck") {
        Stim s;
        all_ones_codes(s);
        printf("--- selfcheck: poison D cols with 1000+row, do_mma=0, read back ---\n");
        if (!run(s, /*do_mma=*/0, /*do_poison=*/1, true)) return 1;
        int bad = 0;
        for (int m = 0; m < BM; ++m)
            for (int n = 0; n < BN; ++n) {
                const float want = (float)(1000 + m);
                if (D[(size_t)m * BN + n] != want) {
                    if (bad < 5)
                        printf("   mismatch D[%d][%d]=%.6f want %.6f\n", m, n,
                               D[(size_t)m * BN + n], want);
                    ++bad;
                }
            }
        printf("   poison echo: %d/%d elements correct  %s\n", BM * BN - bad, BM * BN,
               bad == 0 ? "[PASS] read path + TMEM D columns verified"
                        : "[FAIL] the D read-back does NOT return what tcgen05.st wrote");
        if (bad) ++failures;
        printf("   dbg: D_tmem=0x%llx a_desc=0x%016llx\n", (unsigned long long)dbg[3],
               (unsigned long long)dbg[0]);
    }
    // ------------------------------------------------------------------ const
    else if (mode == "const" || mode == "sfprobe") {
        Stim s;
        all_ones_codes(s);
        if (mode == "sfprobe") {
            for (int m = 0; m < BM; ++m) {
                uint32_t w = 0;
                for (int j = 0; j < NKB; ++j) w |= (uint32_t)(127 + j) << (8 * j);
                s.SFA[m] = w;
            }
            for (int n = 0; n < BN; ++n) {
                uint32_t w = 0;
                for (int j = 0; j < NKB; ++j) w |= (uint32_t)(127 + j) << (8 * j);
                s.SFB[n] = w;
            }
        }
        printf("--- %s: A=B=1.0 codes, SF byte j = %s ---\n", mode.c_str(),
               mode == "sfprobe" ? "127+j (2^j)" : "127 (1.0)");
        if (!run(s, /*do_mma=*/1, /*do_poison=*/1, true)) return 1;
        const std::vector<double> ref = golden(s);
        printf("   dbg a_desc=0x%016llx b_desc=0x%016llx idesc0=0x%08x idesc3=0x%08x "
               "D_tmem=0x%llx mbar_wait_cycles=%llu\n",
               (unsigned long long)dbg[0], (unsigned long long)dbg[1], (unsigned)dbg[2],
               (unsigned)dbg[3], (unsigned long long)dbg[3], (unsigned long long)dbg[5]);
        if (!report_vs_golden(ref)) ++failures;
        printf("   D[0][0..7]= ");
        for (int n = 0; n < 8; ++n) printf("%.4f ", D[n]);
        printf("\n");
    }
    // ------------------------------------------------------------------ random
    else if (mode == "random" || mode == "random_sf") {
        const bool vary = (mode == "random_sf");
        if (argc >= 3) g_rng = (uint32_t)strtoul(argv[2], nullptr, 0);
        Stim s;
        random_codes(s, vary);
        printf("--- %s: dense random A/B, SF %s ---\n", mode.c_str(),
               vary ? "distinct per K-block (the micro-harness data shape)" : "all 127 (1.0)");
        if (!run(s, /*do_mma=*/1, /*do_poison=*/1, true)) return 1;
        const std::vector<double> ref = golden(s);
        printf("   dbg a_desc=0x%016llx b_desc=0x%016llx idesc0=0x%08x idesc3=0x%08x "
               "mbar_wait_cycles=%llu\n",
               (unsigned long long)dbg[0], (unsigned long long)dbg[1], (unsigned)dbg[2],
               (unsigned)dbg[3], (unsigned long long)dbg[5]);
        if (!report_vs_golden(ref)) ++failures;
    }
    // ------------------------------------------------------------------ sfscan
    // Which byte of the SF word actually selects which K-block? Impulse at
    // (0, k0) with k0 inside K-block kb; double ONE scale byte (127 -> 128 = 2^1)
    // of the impulse row. If byte j feeds K-block kb, D reads 2.0 instead of 1.0.
    else if (mode == "sfscan") {
        printf("--- sfscan: impulse in the operand's row 0 at K-block kb; one SF byte j doubled ---\n");
        printf("    (D value 2.0 => byte j is the scale that K-block kb actually uses; 1.0 => not)\n\n");
        for (int side = 0; side < 2; ++side) {
            const char *want = side == 0 ? "A (SFA)" : "B (SFB)";
            printf("  === %s side: impulse on that operand's row 0 ===\n", want);
            for (int probe = 0; probe < 2; ++probe) {  // impulse start / middle of the block
                const int koff = probe == 0 ? 0 : 16;
                printf("    impulse k0 = 32*kb + %d\n", koff);
                printf("      kb      ");
                for (int j = 0; j < NKB; ++j) printf(" byte%d=%d  ", j, probe ? -1 : j);
                printf("   (blank = not measured)\n");
                for (int kb = 0; kb < NKB; ++kb) {
                    printf("      kb=%d   ", kb);
                    for (int j = 0; j < NKB; ++j) {
                        Stim s;
                        all_ones_codes(s);
                        const int k0 = 32 * kb + koff;
                        if (side == 0) {
                            std::fill(s.A.begin(), s.A.end(), 0);
                            s.A[(size_t)0 * BK + k0] = kE4M3One;
                            s.SFA[0] = kSFOne | (uint32_t)(128u << (8 * j));  // byte j = 2^1
                        } else {
                            std::fill(s.B.begin(), s.B.end(), 0);
                            s.B[(size_t)0 * BK + k0] = kE2M1One;
                            s.SFB[0] = kSFOne | (uint32_t)(128u << (8 * j));
                        }
                        if (!run(s, /*do_mma=*/1, /*do_poison=*/1, j == 0 && kb == 0))
                            { printf(" ERR "); continue; }
                        // response should be exactly 1.0 (unused byte) or 2.0 (used byte)
                        printf(" %8.5f", D[0]);
                    }
                    printf("\n");
                }
            }
            // control: baseline with all bytes 127
            {
                Stim s; all_ones_codes(s);
                if (side == 0) { std::fill(s.A.begin(), s.A.end(), 0); s.A[0] = kE4M3One; }
                else { std::fill(s.B.begin(), s.B.end(), 0); s.B[0] = kE2M1One; }
                if (run(s, 1, 1, false)) printf("    control (all SF 127, impulse at k=0): D[0][0]=%.5f\n", D[0]);
            }
        }
    }
    // ------------------------------------------------------------------ sweep1d
    // Which k / m / n indices does the MMA actually read? One impulse per index,
    // B (or A) constant 1.0, SF all 127 -> "1" = that index is read, "0" = blind.
    else if (mode == "sweep1d") {
        const std::string which = (argc >= 3) ? argv[2] : "k";
        printf("--- sweep1d %s: impulse at (0,%s0) for %s0 = 0..127, read-back value ---\n",
               which.c_str(), which.c_str(), which.c_str());
        std::vector<int> lit_vals(128, 0);
        int lit = 0;
        for (int idx = 0; idx < 128; ++idx) {
            Stim s;
            all_ones_codes(s);
            if (which == "k") {
                std::fill(s.A.begin(), s.A.end(), 0);
                s.A[(size_t)0 * BK + idx] = kE4M3One;
            } else if (which == "m") {
                std::fill(s.A.begin(), s.A.end(), 0);
                s.A[(size_t)idx * BK + 0] = kE4M3One;
            } else {
                std::fill(s.B.begin(), s.B.end(), 0);
                s.B[(size_t)idx * BK + 0] = kE2M1One;
            }
            if (!run(s, 1, 1, false)) continue;
            const int m = (which == "m") ? idx : 0;
            const int n = (which == "n") ? idx : 0;
            const float v = D[(size_t)m * BN + n];
            lit_vals[idx] = (std::fabs(v) > 1e-6f) ? (int)std::lround(v * 1000.0f) : 0;
            if (lit_vals[idx]) ++lit;
        }
        printf("   read-map (128 bits, '1' = response at the probed index):\n");
        for (int b = 0; b < 4; ++b) {
            printf("     [%3d..%3d] ", 32 * b, 32 * b + 31);
            for (int j = 0; j < 32; ++j) printf("%c", lit_vals[32 * b + j] ? '1' : '0');
            printf("   values(milli):");
            for (int j = 0; j < 32; ++j) printf(" %d", lit_vals[32 * b + j]);
            printf("\n");
        }
        printf("   %d/128 indices respond\n", lit);
    }
    // ------------------------------------------------------------------ lboscan / sboscan
    // Which descriptor lbo/sbo (16B units) makes the K / M direction fully readable?
    else if (mode == "lboscan" || mode == "sboscan") {
        const bool is_lbo = (mode == "lboscan");
        const int vals[] = {1, 2, 4, 8, 16, 32, 64, 128};
        printf("--- %s: impulse sweep over %s with the descriptor %s varied ---\n",
               mode.c_str(), is_lbo ? "k (A side)" : "m (A side)", is_lbo ? "lbo" : "sbo");
        for (int v : vals) {
            const int lbo16 = is_lbo ? v : 8;
            const int sbo16 = is_lbo ? 16 : v;
            int lit = 0;
            std::string map(128, '.');
            for (int idx = 0; idx < 128; ++idx) {
                Stim s;
                all_ones_codes(s);
                std::fill(s.A.begin(), s.A.end(), 0);
                if (is_lbo)
                    s.A[(size_t)0 * BK + idx] = kE4M3One;
                else
                    s.A[(size_t)idx * BK + 0] = kE4M3One;
                if (!run(s, 1, 1, false, lbo16, sbo16)) continue;
                const int m = is_lbo ? 0 : idx;
                const float val = D[(size_t)m * BN + (is_lbo ? 0 : 0)];
                if (std::fabs(val) > 1e-6f) { map[idx] = '1'; ++lit; }
            }
            printf("   lbo16=%-4d sbo16=%-4d : %3d/128 respond   ", lbo16, sbo16, lit);
            printf("%s\n", map.c_str());
        }
    }
    // ------------------------------------------------------------------ offscan
    // Physical byte-offset response: write the impulse at RAW A/B smem byte offset.
    else if (mode == "offscan") {
        const std::string side = (argc >= 3) ? argv[2] : "A";
        const int n_off = (argc >= 4) ? atoi(argv[3]) : 512;
        printf("--- offscan %s: impulse at RAW smem byte offset (desc lbo=8 sbo=16) ---\n",
               side.c_str());
        for (int o = 0; o < n_off; ++o) {
            Stim s;
            all_ones_codes(s);
            std::fill(s.A.begin(), s.A.end(), 0);
            std::fill(s.B.begin(), s.B.end(), 0);
            if (side == "A") {
                std::fill(s.B.begin(), s.B.end(), kE2M1One);  // B all 1.0
            } else {
                std::fill(s.A.begin(), s.A.end(), kE4M3One);  // A all 1.0
            }
            if (!run(s, 1, 1, false, 8, 16, side == "A" ? o : -1, side == "B" ? o : -1))
                continue;
            // find the response
            int nm = 0, nn = 0, firstm = -1, firstn = -1;
            float fv = 0;
            for (int m = 0; m < BM; ++m)
                for (int n = 0; n < BN; ++n)
                    if (std::fabs(D[(size_t)m * BN + n]) > 1e-6f) {
                        if (!nm) { firstm = m; fv = D[(size_t)m * BN + n]; }
                        ++nm; ++nn;
                    }
            if (o % 16 == 0) printf("   off %% 256 == %3d:\n", o % 256);
            printf("     off=%-4d (rowgrp=%d,+%3d) -> rows=%3d first_row=%3d value=%.5f\n", o,
                   o / 256, o % 256, nm, firstm, fv);
        }
    }
    // ------------------------------------------------------------------ impulse
    else {
        std::vector<std::pair<int, int>> cases;
        const bool impulse_in_b = (mode == "impulseB");
        if (argc >= 4) {
            cases.emplace_back(atoi(argv[2]), atoi(argv[3]));
        } else {
            const int r0s[] = {0, 1, 7, 8, 31, 32, 63, 64, 127};
            const int k0s[] = {0, 32, 64, 96};
            for (int r0 : r0s)
                for (int k0 : k0s) cases.emplace_back(r0, k0);
        }
        printf("--- impulse-response sweep: %s ---\n",
               impulse_in_b ? "impulse in B at (n0,k0), A all 1.0"
                            : "impulse in A at (m0,k0), B all 1.0");
        printf("%-14s %-6s %-6s %-8s %-8s  %s\n", "CASE", "nz", "rows", "max|D|", "min_nz",
               "response");
        for (auto &c : cases) {
            const int r0 = c.first, k0 = c.second;
            Stim s;
            all_ones_codes(s);
            if (impulse_in_b) {
                std::fill(s.B.begin(), s.B.end(), 0);           // e2m1 0.0
                s.B[(size_t)r0 * BK + k0] = kE2M1One;
            } else {
                std::fill(s.A.begin(), s.A.end(), 0);           // e4m3 0.0
                s.A[(size_t)r0 * BK + k0] = kE4M3One;
            }
            if (!run(s, /*do_mma=*/1, /*do_poison=*/1, c == cases.front())) continue;
            double maxabs = 0, min_nz = 1e30, max_nz = -1e30;
            long nz = 0;
            std::vector<int> nz_rows, nz_cols(BN, 0), row_nz(BM, 0);
            std::vector<std::vector<int>> row_cols(BM);
            for (int m = 0; m < BM; ++m)
                for (int n = 0; n < BN; ++n) {
                    const float v = D[(size_t)m * BN + n];
                    if (std::fabs(v) > 1e-6f) {
                        ++nz;
                        ++row_nz[m];
                        ++nz_cols[n];
                        row_cols[m].push_back(n);
                        min_nz = std::fmin(min_nz, v);
                        max_nz = std::fmax(max_nz, v);
                    }
                    maxabs = std::fmax(maxabs, std::fabs((double)v));
                }
            for (int m = 0; m < BM; ++m)
                if (row_nz[m]) nz_rows.push_back(m);
            int nz_cols_cnt = 0, first_col = -1;
            for (int n = 0; n < BN; ++n)
                if (nz_cols[n]) { ++nz_cols_cnt; if (first_col < 0) first_col = n; }

            char resp[192];
            if (nz == 0)
                snprintf(resp, sizeof(resp), "NOTHING LIT (K atom read as 0 / wrong base?)");
            else if (!impulse_in_b && nz_rows.size() == 1 && nz_rows[0] == r0 && row_nz[r0] == BN)
                snprintf(resp, sizeof(resp),
                         std::fabs(min_nz - 1.0) < 1e-3 ? "EXPECTED row %d = 1.0 x128" : "row %d lit, VALUE %.4g..%.4g != 1.0", r0, min_nz, max_nz);
            else if (impulse_in_b && nz_cols_cnt == 1 && first_col == r0)
                snprintf(resp, sizeof(resp), "EXPECTED col %d = 1.0 x128", r0);
            else
                snprintf(resp, sizeof(resp), "rows=%zu cols=%d first_col=%d (want %s %d)",
                         nz_rows.size(), nz_cols_cnt, first_col, impulse_in_b ? "col" : "row", r0);
            printf("%s=%-3d k0=%-3d %-6ld %-6zu %-8.4f %-8.4f  %s\n", impulse_in_b ? "n0" : "m0",
                   r0, k0, nz, nz_rows.size(), maxabs, nz == 0 ? 0.0 : min_nz, resp);
            if (c == cases.front())
                printf("        dbg a_desc=0x%016llx b_desc=0x%016llx idesc0=0x%08x idesc3=0x%08x "
                       "mbar_wait_cycles=%llu\n",
                       (unsigned long long)dbg[0], (unsigned long long)dbg[1], (unsigned)dbg[2],
                       (unsigned)dbg[3], (unsigned long long)dbg[5]);
            if (nz_rows.size() == 1) {
                const int m = nz_rows[0];
                printf("        row %d samples: D[%d][0]=%.6f D[%d][1]=%.6f D[%d][63]=%.6f "
                       "D[%d][127]=%.6f\n",
                       m, m, D[(size_t)m * BN + 0], m, D[(size_t)m * BN + 1], m,
                       D[(size_t)m * BN + 63], m, D[(size_t)m * BN + 127]);
            }
        }
    }

    cudaFree(dD);
    cudaFree(dDbg);
    printf("\nverdict: %s\n", failures ? "FAILURES PRESENT" : "all checks passed");
    return failures ? 1 : 0;
}
