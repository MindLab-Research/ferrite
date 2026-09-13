// moe_bs_handwritten.cu — 手写版 block-scaled fp4 MoE gate/up MMA kernel
// 基于 tests_tcgen05_mxf8f6f4_1x.cu 的 VERIFIED 原语（sm_103a round-trip 验证）
// 结构：顺序执行（无流水线重叠）——load → sync → SF copy → MMA → sync → 循环
// 性能：比 TileLang 流水线慢（无 TMA overlap），但用验证过的指令序列
// 用途：TileLang kernel 的 m>1 crash 诊断替代 + 性能基线
//
// 与 TileLang kernel 的关键区别：
// 1. 直接 global load（不用 TMA）——消除 TMA descriptor 依赖
// 2. __syncthreads() 同步（不用 mbarrier pipeline）——消除流水线竞态
// 3. 验证过的 MMA 指令（带 memory clobber，不带 .scale_vec::1X——那个破坏 eager）
// 4. 顺序 k-loop（40 次迭代，每次 load + compute + sync）

#include <cuda.h>
#include <cstdint>
#include <cstdio>

// ============================================================
// VERIFIED tcgen05 原语（从 tests_tcgen05_mxf8f6f4_1x.cu 逐字复制）
// ============================================================

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

// VERIFIED MMA instruction — memory clobber included, .scale_vec::1X EXCLUDED
// (the suffix broke eager in previous experiment; clobber-only is the test)
__device__ __forceinline__ void hw_tc_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                          uint32_t idesc, uint32_t sfa_tmem,
                                          uint32_t sfb_tmem, uint32_t enable_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}"
        ::"r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
          "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d)
        : "memory");
}

__device__ __forceinline__ void hw_tc_commit(void *mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(mbar)) : "memory");
}

// TMEM read: 使用 TileLang 已验证的 tcgen05_ld（避免 128-operand 自定义 asm 的 ICE）
// tl::tcgen05_ld_32dp32bNx 在 moe_bs_up_tl.cu 的 include 链中已可用
// (replaces the custom hw_tc_ld<128> which caused Internal Compiler Error
//  "asm operand index larger than number of operands" with 128 constraints)

// SMEM descriptor for tcgen05 MMA（从 verified make_desc 适配）
// layout: start_addr[0:14) | lbo[16:30) | sbo[32:46) | version=1[46] | layout[61:64)
__device__ __forceinline__ uint64_t hw_make_desc(const void* smem_ptr,
                                                  uint32_t lbo_16B, uint32_t sbo_16B,
                                                  uint32_t layout) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);           // start_address (16B units)
    d |= (uint64_t)(lbo_16B & 0x3FFF) << 16;          // leading_byte_offset
    d |= (uint64_t)(sbo_16B & 0x3FFF) << 32;          // stride_byte_offset
    d |= (uint64_t)1 << 46;                            // version = SM100
    d |= (uint64_t)(layout & 0x7) << 61;               // layout_type
    return d;
}

// IDESC for mxf8f6f4 block-scaled MMA（从 TileLang 常量 144708608 适配）
// layout: b_sf[4:6) | a_fmt[7:10) | b_fmt[10:13) | n_dim[17:23) | sf_fmt[23] | m_dim[24:29) | a_sf[29:31)
__device__ __forceinline__ uint32_t hw_make_idesc(int m, int n,
                                                   int a_fmt, int b_fmt,
                                                   int sf_id) {
    uint32_t d = 0;
    d |= (uint32_t)(sf_id & 3);                        // b_sf_id [4,6)
    d |= (uint32_t)(a_fmt & 7) << 7;                   // a_format [7,10) — 0=E4M3
    d |= (uint32_t)(b_fmt & 7) << 10;                  // b_format [10,13) — 5=E2M1
    d |= (uint32_t)((n >> 3) & 63) << 17;              // n_dim [17,23) — N/8
    d |= (uint32_t)1 << 23;                             // scale_format = UE8M0
    d |= (uint32_t)((m >> 4) & 31) << 24;              // m_dim [24,29) — M/16
    d |= (uint32_t)(sf_id & 3) << 29;                  // a_sf_id [29,31)
    return d;
}

// SF 转置（从 TileLang tcgen05_sf_warp_transpose 复制——4×32 uint32 块内转置）
__device__ __forceinline__ void hw_sf_transpose(uint32_t *smem_ptr) {
    const uint32_t lane = threadIdx.x % 32;
    uint32_t values[4];
    for (uint32_t i = 0; i < 4; ++i)
        values[i] = smem_ptr[(i ^ (lane >> 3)) * 32 + lane];
    __syncwarp();
    for (uint32_t i = 0; i < 4; ++i)
        smem_ptr[lane * 4 + (i ^ (lane >> 3))] = values[i];
}

// SF copy to TMEM（从 TileLang tcgen05_cp 复制——32x128b.warpx4 shape）
__device__ __forceinline__ void hw_tc_cp(uint64_t smem_desc, uint32_t tmem_col) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_col), "l"(smem_desc));
}

// SF smem descriptor（从 TileLang make_sf_smem_desc 复制）
__device__ __forceinline__ uint64_t hw_make_sf_desc(void *smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= (uint64_t)(addr >> 4) & 0x3FFF;  // start_address
    desc |= (uint64_t)8u << 32;               // stride_byte_offset >> 4 = 8
    desc |= (uint64_t)1u << 46;               // version = 1
    return desc;
}

// ============================================================
// 手写 MoE BS gate/up kernel — 顺序执行
// ============================================================
// Grid: (kGridX=5, kSegCap=36), Block: 128 threads (4 warps)
// 每个 block 处理 C[seg*128..+127][n_tile*128..+127]（128×128 f32 输出块）
// K-loop: 40 iterations (5120 / 128)

constexpr int HW_BM = 128;    // M tile (rows per segment)
constexpr int HW_BN = 128;    // N tile (output columns)
constexpr int HW_BK = 128;    // K tile (per iteration)
constexpr int HW_K = 5120;    // total K
constexpr int HW_NUP = 640;   // total output columns (2 * 320)
constexpr int HW_NP = 320;    // weight rows per plane
constexpr int HW_NH = 64;     // rows per half-tile (W1 or W3)
constexpr int HW_SEGCAP = 36;
constexpr int HW_KITER = HW_K / HW_BK;  // 40

// smem 布局（总 98304 + SF 2048 = ~100KB，B300 227KB 内）
// A tile: [128, 128] e4m3 = 16384 B
// B tile: [128, 128] unpacked fp4 = 16384 B（前 64 行 W1，后 64 行 W3）
// SFA: [128] uint32 = 512 B
// SFB: [128] uint32 = 512 B
// C staging: [128, 128] f32 = 65536 B（与 A/B 重叠——所有 MMA 完成后写入）
// mbarrier: 8 B

extern "C" __global__ __launch_bounds__(128, 1) void moe_bs_handwritten_kernel(
    const uint8_t* __restrict__ A,      // [4608, 5120] e4m3 (gathered by shim)
    const uint8_t* __restrict__ W1,     // weight pool W1 base (expert 0)
    const uint8_t* __restrict__ W3,     // weight pool W3 base (expert 0)
    const uint32_t* __restrict__ SFA,   // [40*4608] packed activation SF
    const uint32_t* __restrict__ SFW1,  // [384, 40*320] packed weight SF1
    const uint32_t* __restrict__ SFW3,  // [384, 40*320] packed weight SF3
    const int* __restrict__ Eid,        // [36] expert IDs
    float* __restrict__ C,              // [4608, 640] f32 output
    int64_t w_stride                    // expert stride in weight pool
) {
    const int n_tile = blockIdx.x;   // 0-4
    const int seg = blockIdx.y;      // 0-35
    const int e = Eid[seg];
    const int tid = threadIdx.x;     // 0-127
    const int warp = tid >> 5;       // 0-3
    const int lane = tid & 31;       // 0-31

    // ---- smem layout ----
    // A: [0, 16384) e4m3 tile
    // B: [16384, 32768) unpacked fp4 tile
    // SFA: [32768, 33280) activation SF
    // SFB: [33280, 33792) weight SF
    // mbar: [33792, 33800) MMA completion barrier
    // C staging: [0, 65536) f32 output (overlaps A/B/SF/mbar — written after all MMAs,
    //             mbarrier is no longer needed at that point)
    // Total: 65536 bytes (dominated by C)
    extern __shared__ __align__(1024) uint8_t hw_smem[];
    uint8_t* A_sh = hw_smem;                                   // [128, 128] e4m3
    uint8_t* B_sh = hw_smem + 16384;                           // [128, 128] unpacked fp4
    uint32_t* SFA_sh = (uint32_t*)(hw_smem + 32768);           // [128] activation SF
    uint32_t* SFB_sh = (uint32_t*)(hw_smem + 33280);           // [128] weight SF
    float* C_sh = (float*)hw_smem;                             // [128, 128] f32 (overlaps A/B — safe: written after all MMAs, mbarrier no longer needed)
    // mbarrier for MMA completion — placed AFTER SF, within C's overlap range
    // (safe: mbarrier is only used during k-loop; C overwrites it in epilogue after all waits)
    uint64_t* mma_bar = (uint64_t*)(hw_smem + 33792);          // 8 bytes

    // ---- TMEM allocation (warp 0 only) ----
    __shared__ __align__(16) uint hw_C_tmem;
    __shared__ __align__(16) uint hw_SF_tmem;
    if (warp == 0) {
        hw_tc_alloc(&hw_C_tmem, 128);   // 128 columns for C
        hw_tc_alloc(&hw_SF_tmem, 32);    // 32 columns for SF
        hw_tc_relinquish();
    }
    __syncthreads();
    const uint32_t C_tmem = hw_C_tmem;
    const uint32_t SF_tmem = hw_SF_tmem;

    // ---- init mbarrier (1 arrival from tcgen05.commit) ----
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"((uint32_t)__cvta_generic_to_shared(mma_bar)));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");  // make mbarrier init visible

    // ---- K-loop (sequential, no pipeline) ----
    for (int k = 0; k < HW_KITER; ++k) {
        // (1) 所有线程协同加载 A tile — **CORE MATRIX 布局**（不是 row-major！）
        // UMMA smem descriptor 期望 core matrix 布局:
        //   addr(m, k) = (m/8)*1024 + (k/16)*128 + (m%8)*16 + (k%16)
        // 这是 8行×16B 的 core matrix 顺序排列（m_block 外层，k_block 内层）
        for (int i = tid; i < HW_BM * HW_BK; i += 128) {
            const int m = i >> 7;   // row [0,128)
            const int kk = i & 127; // col [0,128) — byte index for e4m3
            const uint8_t val = A[(int64_t)(seg * HW_BM + m) * HW_K + k * HW_BK + kk];
            // SW128 (CU_TENSOR_MAP_SWIZZLE_128B) — MUST match TileLang's TMA layout:
            //   addr(r,c) = (r/8)*1024 + (r%8)*128 + (((c/16) ^ (r%8))*16) + (c%16)
            A_sh[(m >> 3) * 1024 + (m & 7) * 128 + ((((kk >> 4) ^ (m & 7))) << 4) + (kk & 15)] = val;
        }

        // (2) 所有线程协同加载 B tile (W1 前 64 行 + W3 后 64 行)
        // W1[e][n_tile*64 + row][k*128..k*128+127) — packed fp4，需要 unpack
        // packed: [384, 320, 2560] — expert e 的 W1 面
        // row r 的第 k*128..k*128+127 列 = packed bytes [r*2560 + k*64 .. +64)
        // (2) 所有线程协同加载 B tile — **CORE MATRIX 布局**（同 A）
        // B = W1 前 64 行 + W3 后 64 行，unpacked fp4 (1 byte/element)
        // Core matrix: addr(m, k) = (m/8)*1024 + (k/16)*128 + (m%8)*16 + (k%16)
        for (int i = tid; i < HW_NH * 64; i += 128) {
            const int row = i >> 6;   // 0-63 (W1 row)
            const int col = i & 63;   // 0-63 (packed column)
            const uint8_t packed =
                W1[(int64_t)e * w_stride + (int64_t)(n_tile * HW_NH + row) * 2560 + k * 64 + col];
            // unpack to 2 elements, write in core matrix layout
            // NIBBLE ORDER: LOW nibble = FIRST element (even K index), HIGH = SECOND
            // This matches the old MoE path's GEMV convention:
            //   s_lut2[t] = (decode(t & 0xF), decode(t >> 4))
            //   gp0 = fmaf(sa[0], gt0.x, gp0)  // sa[0] (even K) × LOW nibble
            //   gp1 = fmaf(sa[1], gt0.y, gp1)  // sa[1] (odd K) × HIGH nibble
            // And fp4_pack_kernel: packed = (lo & 0xF) | (hi << 4), lo = element 2i
            const int k0 = col * 2;      // first element K index
            const int k1 = col * 2 + 1;  // second element K index
            // SW128 swizzled write (same layout as TileLang's TMA)
            B_sh[(row >> 3) * 1024 + (row & 7) * 128 + ((((k0 >> 4) ^ (row & 7))) << 4) + (k0 & 15)] = packed & 0xF;
            B_sh[(row >> 3) * 1024 + (row & 7) * 128 + ((((k1 >> 4) ^ (row & 7))) << 4) + (k1 & 15)] = packed >> 4;
        }
        for (int i = tid; i < HW_NH * 64; i += 128) {
            const int row = i >> 6;
            const int col = i & 63;
            const uint8_t packed =
                W3[(int64_t)e * w_stride + (int64_t)(n_tile * HW_NH + row) * 2560 + k * 64 + col];
            const int m = HW_NH + row;  // W3 rows are after W1
            const int k0 = col * 2;
            const int k1 = col * 2 + 1;
            B_sh[(m >> 3) * 1024 + (m & 7) * 128 + ((((k0 >> 4) ^ (m & 7))) << 4) + (k0 & 15)] = packed & 0xF;
            B_sh[(m >> 3) * 1024 + (m & 7) * 128 + ((((k1 >> 4) ^ (m & 7))) << 4) + (k1 & 15)] = packed >> 4;
        }

        // (3) 加载 SF
        // SFA: [40, 4608] — SFA[k][seg*128 + i] for i in [0, 128)
        for (int i = tid; i < HW_BM; i += 128) {
            SFA_sh[i] = SFA[(int64_t)k * (HW_SEGCAP * HW_BM) + seg * HW_BM + i];
        }
        // SFW1/SFW3: [384, 40*320] — SFW[e][k*320 + n_tile*64 + i] for i in [0, 64)
        for (int i = tid; i < HW_NH; i += 128) {
            SFB_sh[i] = SFW1[(int64_t)e * (40 * HW_NP) + k * HW_NP + n_tile * HW_NH + i];
            SFB_sh[HW_NH + i] = SFW3[(int64_t)e * (40 * HW_NP) + k * HW_NP + n_tile * HW_NH + i];
        }

        __syncthreads();

        // (4) warp 2: SF 转置（tcgen05_cp 需要的 smem 布局）
        if (warp == 2) {
            hw_sf_transpose(SFA_sh);
            hw_sf_transpose(SFB_sh);
        }
        __syncthreads();

        // (5) warp 1: SF copy to TMEM + MMA
        if (warp == 1) {
            // SF copy: SFA → SF_tmem+0, SFB → SF_tmem+4
            // (elect_one_sync: single thread issues the copy)
            if (lane == 0) {
                hw_tc_cp(hw_make_sf_desc(SFA_sh), SF_tmem + 0);
                hw_tc_cp(hw_make_sf_desc(SFB_sh), SF_tmem + 4);
            }

            // MMA: 4 sub-MMAs (ki=0-3, each covers 32 K elements)
            // smem descriptors for A and B tiles
            // layout=0 (no swizzle), LBO=1 (16B = core matrix row stride), SBO=64 (1024B = 8-row atom stride)
            // (数据已按 core matrix 布局写入，无需 swizzle)
            const uint64_t a_desc_base = hw_make_desc(A_sh, 1, 64, 2);  // TileLang exact: lbo=1, sbo=64, SW128(layout=2)
            const uint64_t b_desc_base = hw_make_desc(B_sh, 1, 64, 2);  // TileLang exact: lbo=1, sbo=64, SW128(layout=2)

            for (int ki = 0; ki < 4; ++ki) {
                // idesc: M=128, N=128, a_fmt=0 (E4M3), b_fmt=5 (E2M1), sf_id=ki
                const uint32_t idesc = hw_make_idesc(HW_BM, HW_BN, 0, 5, ki);
                // A/B descriptor advance: ki * 32 (raw descriptor units = 16B each → 512B offset)
                // Descriptor advance per K-block: 256 bytes = 16 units (NOT 32!)
                // Core matrix layout: each K-block (32 elements) spans 2 K-atoms
                // = 2 × 128 bytes = 256 bytes = 16 start_address units (16B each).
                // TileLang uses ki*32 for its TMA+swizzle layout — WRONG for our
                // core matrix + layout=0. With synthetic data (all same bytes),
                // this 2× error was invisible. With real data, it reads the
                // WRONG K-block's data → garbage output.
                const uint64_t a_desc = a_desc_base + (uint64_t)(ki * 2);
                const uint64_t b_desc = b_desc_base + (uint64_t)(ki * 2);
                // enable_d: 0 for first MMA (clear accumulator), 1 for rest
                const uint32_t enable_d = (k == 0 && ki == 0) ? 0 : 1;

                // elect_one_sync: single thread in warp issues the MMA
                if (lane == 0) {
                    hw_tc_mma(C_tmem, a_desc, b_desc, idesc,
                              SF_tmem + 0, SF_tmem + 4, enable_d);
                }
            }

            // Commit: signal mbarrier when all MMAs complete
            if (lane == 0) {
                hw_tc_commit(mma_bar);
            }
        }

        // (6) 所有线程等待 MMA 完成（mbarrier wait + syncthreads）
        if (tid == 0) {
            // mbarrier wait (phase 0 for first use, then alternating)
            const uint32_t phase = k & 1;
            asm volatile(
                "{\n\t.reg .pred P;\n\t"
                "WAIT:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra WAIT;\n\t}"
                :: "r"((uint32_t)__cvta_generic_to_shared(mma_bar)), "r"(phase));
        }
        __syncthreads();

        // (7) 下一轮 k-iteration 会覆盖 smem——MMA 已完成，安全
    }

    // ---- epilogue: read TMEM → registers → smem → global ----
    // TMEM read: each thread reads 128 f32 values (32x32b.x128)
    // C layout in TMEM: 128 lanes × 128 columns
    // Thread t reads from lane (t % 32) + warp offset, columns 0-127
    {
        float C_reg[128];
        // TileLang's verified TMEM read (replaces custom hw_tc_ld<128> that caused ICE)
        tl::tcgen05_ld_32dp32bNx<128, false>(C_tmem, 0, C_reg);

        __syncthreads();  // ensure all threads have read TMEM before writing C_sh

        // Write to C_sh (swizzled for coalesced global store)
        // Simple layout: C_sh[row * 128 + col] where row = warp * 32 + lane
        const int row = warp * 32 + lane;
        for (int col = 0; col < 128; ++col) {
            C_sh[row * 128 + col] = C_reg[col];
        }
        __syncthreads();

        // Store to global: C[seg*128 + row][n_tile*128 + col]
        for (int i = tid; i < HW_BM * HW_BN; i += 128) {
            const int r = i >> 7;
            const int c = i & 127;
            C[(int64_t)(seg * HW_BM + r) * HW_NUP + n_tile * HW_BN + c] = C_sh[r * HW_BN + c];
        }
    }

    // ---- TMEM dealloc ----
    __syncthreads();
    if (warp == 0) {
        hw_tc_dealloc(C_tmem, 128);
        hw_tc_dealloc(SF_tmem, 32);
    }
}
