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

// Runtime switch for the .scale_vec::1X suffix (set by the shim from
// DSV41_MOE_BS_SCALEVEC1X before launch; the asm must be compile-time so both
// spellings are compiled in and chosen at runtime).
__device__ int g_sv1x = 0;
// Runtime switch for the smem layout family:
//   0 = TileLang's SW128 (layout_type=2, lbo=1, sbo=64, advance ki*32B)
//   1 = repo-verified canonical interleave (layout_type=0, lbo=8, sbo=16, advance ki*4096B)
__device__ int g_canon = 0;
// SwapAB: A = WEIGHTS (e2m1), B = ACTIVATIONS (e4m3). Every GPU-verified in-tree
// mxf8f6f4 configuration uses this orientation (tests_tcgen05_mxf8f6f4_1x.cu's probe
// builds make_idesc_mxf8f6f4(.., a_fmt=5, b_fmt=0, ..) and tc5::e4 likewise), while the
// two implementations that use the opposite orientation (TileLang's generated kernel,
// err 700 at INIT) and this kernel (garbage) both fail.
__device__ int g_swapab = 0;
// SFREV: read the SF word MSB-first (byte j holds K-block 3-j) instead of LSB-first.
// A one-line test of the only hypothesis that explains a systematic error shared by
// every layout / orientation / SF-delivery combination: the sf_id byte order.
__device__ int g_sfrev = 0;
// g_packed: stage the fp4 (E2M1) operand the way the HARDWARE reads it — PACKED, two K
// elements per byte (low nibble = even k). Measured, not assumed: with every B byte set
// to 0x22 (both nibbles = 1.0) a dense parity matches the gold standard EXACTLY
// (relerr = 0), while 0x02 (high nibble 0) yields exactly half of it and only the even
// k indices respond to an impulse. One scale_vec::1X K-block (32 elements) is therefore
// 16 smem bytes; the descriptor that the vendor's own W operand uses is
// lbo=1 (16 B) / sbo=64 (1024 B) / layout_type=2 (SWIZZLE_128B). See
// docs/agent/moe-bs-crash-investigation.md §29-§31.
__device__ int g_sfst = 0;     // 1 = deliver the SF with tcgen05.st (isolated-instrument path)
// 1 = run ONLY K-stage 0 (DSV41_MOE_BS_STAGE1=1), collapsing production to the isolation
// instrument's single-stage regime so the K<128 partial can be judged against the oracle.
__device__ int g_stage1 = 0;
__device__ int g_ldw = 1;      // 1 (DEFAULT) = read D with the per-warp TMEM lane address, which
                               // PTX ISA 9.7.18.1.1 + CUTLASS + Triton all require (the address's
                               // lane field is an ABSOLUTE lane coordinate and a warp may only
                               // touch its own 32-lane partition). 0 = the old spelling with lane
                               // field 0 for every warp (`DSV41_MOE_BS_LDW=0`, escape hatch).
// 1 = one-shot semantic dump of the STAGED operands at k == 0 (see the dump block in the k-loop
// and `scripts/campaign/sfdump_check.py`). Tells "the staged content is wrong" apart from "the
// delivery/descriptor mapping is wrong" — the last two items no instrument has ever covered.
__device__ int g_sfdump_on = 0;
__device__ int g_sfdump_k = 0;   // which K-stage the snapshot captures (k == 0 is blind to advances)
__device__ uint8_t* g_sfdump = nullptr;
constexpr size_t kSfDumpA = 0;                                  // [36][128][128] u8
constexpr size_t kSfDumpB = kSfDumpA + 36 * 128 * 128;          // [36][5][128][64] u8
constexpr size_t kSfDumpSfa = kSfDumpB + 36 * 5 * 128 * 64;     // [36][128] u32
constexpr size_t kSfDumpSfb = kSfDumpSfa + 36 * 128 * 4;        // [36][5][128] u32
constexpr size_t kSfDumpBytes = kSfDumpSfb + 36 * 5 * 128 * 4;  // 2174976 B
__device__ int g_packed = 1;   // DEFAULT ON: the MEASURED-correct layout (hw_pack_sw128)
// Escape hatch for reference only: DSV41_MOE_BS_UNPACKED=1 restores the measured-WRONG
// unpacked staging (relerr 1.491) that this whole investigation started from.

// Address inside the PACKED fp4 operand tile (K packed into K/2 bytes = 64 B per row at
// K=128). PREDICTION (§30): the 128-byte swizzle span holds two 64-byte rows, so the row's
// four 16-byte chunks are XOR-permuted by the row index. The empirical calibration
// (subagent bs-packed-geometry, judged by relerr==0 on a dense parity) is authoritative —
// if its formula differs, replace THIS FUNCTION only.
// g_packgeom selects between the two candidate geometries (see §32):
//   0 = §30 guess   : lbo=1 (16 B), sbo=64 (1024 B), layout=2, 16 B chunks swizzled by row
//   1 = candidate B : lbo=8 (128 B), sbo=32 (512 B), layout=0, kb (= k/32) is the K-block
// The empirical calibration (relerr==0 on a dense parity) is authoritative.
__device__ int g_packgeom = 0;

// g_cpasync: cp.async DOUBLE BUFFERING (gate DSV41_MOE_BS_CPASYNC, DEFAULT OFF).
// ---------------------------------------------------------------------------
// The sequential kernel exposes the whole staging latency once per K-iteration:
// issue A/B/SF  -> LDG->STS dependency stalls the block -> __syncthreads ->
// SF transpose (warp 2 only; warps 0/1/3 wait) -> fence -> tcgen05.cp + 4x MMA ->
// mbarrier wait. Nothing overlaps.
//
// With g_cpasync=1 the A/B/SF buffers are split into TWO stages and the global
// -> smem staging is done with `cp.async` (no register round trip, no LDG->STS
// dependency), with the loads for K-iteration k+1 issued at the TOP of iteration
// k so that they run concurrently with the SF transpose + MMA of iteration k:
//
//   per-iteration timeline (gate ON)
//     [issue k+1 -> stage (k+1)&1] [wait k -> stage k&1] [sync] [transpose SF(k)]
//     [fence] [tcgen05.cp + MMA(k)] [mbar wait]   <- the MMA of k overlaps the
//                                                    in-flight loads of k+1
//
// INVARIANTS (must not change — see docs/agent/moe-bs-crash-investigation.md §47):
//   * global memory layout / read formulas are IDENTICAL to the sequential path;
//   * the fp4 operand is still staged with hw_pack_sw128 (packed bytes in 16 B
//     containers of which only the first 8 B are read);
//   * descriptors stay lbo=1/sbo=64/layout=2 with the ki*32 B K-advance, idesc
//     unchanged, nibble order unchanged -> the MMA consumes the same bytes in the
//     same order. Double buffering only changes WHEN the bytes are written.
//   * no TMA / no mbarrier-based multi-stage pipeline (that path crashed here).
__device__ int g_cpasync = 0;

// AUTHORITATIVE fp4 smem layout, measured to EXACT parity (relerr = 0) on B300/sm_103a:
// the hardware consumes two packed 4-bit elements per byte, but each byte pair lives in a
// 16-BYTE container of which ONLY THE FIRST 8 BYTES ARE READ (TMA dtype 16U4_ALIGN16B =
// 16 four-bit elements = 8 B of data in a 16 B container). So a 64-byte packed row (K=128)
// occupies 8 containers = 128 B of smem, and 128 rows = 16384 B — which is exactly why the
// vendor's per-stage fp4 smem is 16384 B with the SAME SW128 descriptor and the SAME ki*32
// advance as the e4m3 operand. Sweeping raw smem offsets showed both descriptor families
// read only bytes 0..7 of every 16-byte slot.
//   p = packed byte index inside the row, 0..63; c = 16-byte container index, 0..7
//   verified: const/random/sfprobe/impulse all PASS exactly; unpacked staging and every
//   dense-row variant FAIL (relerr 1.49 / 1.03 / 0.457).
__device__ __forceinline__ int hw_pack_sw128(int row, int p) {
    const int c = (p >> 3) & 7;
    return (row >> 3) * 1024 + (row & 7) * 128 + (((c ^ (row & 7)) & 7) << 4) + (p & 7);
}

// ⚠️ SUPERSEDED AND MEASURED WRONG — DO NOT USE (kept as a reference only, per the
// no-revert rule). This was the §30 guess; the hardware-calibrated layout is
// hw_pack_sw128 above (§47), which reaches EXACT parity (relerr = 0), whereas this
// one measures 1.031 and the GLOBAL env switch DSV41_MOE_BS_PACKGEOM is now moot for
// the packed path (it is pinned to the vendor descriptor + hw_pack_sw128).
__device__ __forceinline__ int hw_pack_idx(int row, int col /* packed byte index, 0..63 */) {
    if (g_packgeom == 3) {
        // candidate E (most likely): PLAIN row-major, no swizzle at all. Derived from the
        // official host-side TMA descriptor arguments (moe_bs_up_tl_host.cu:939-1000):
        // gdim=(5120,320,384), gstride=(1,2560,819200) BYTES so W1 is [n][K/2] with a
        // 2560 B row (K/2 packed), box=(128,64,1) -- i.e. 64 BYTES per row, giving the
        // observed 128*64 = 8192 B sub-tile -- and the trailing 1,1,1,0 suggest swizzle 0.
        // Plain rows then pair with lbo=1 (16 B = one packed K-block), sbo=32 (512 B = 8
        // rows x 64 B) and layout_type=0 in the descriptor table below.
        return row * 64 + col;
    }
    if (g_packgeom == 1) {
        // candidate B: byte(row, kb) = (row%8)*16 + kb*128 + (row/8)*512, kb = col/16
        return (row & 7) * 16 + (col >> 4) * 128 + (row >> 3) * 512 + (col & 15);
    }
    // listitem candidate D (g_packgeom == 2 in the descriptor/advance tables): same dense
    // 64-byte rows as geometry 0, i.e. the write formula below, but with the DOCUMENTED
    // SWIZZLE_64B layout_type (4) and sbo = 32 units (512 B = 8 rows x 64 B). Rationale:
    // common.h:751-757 says layout_type 2 = SWIZZLE_128B, 4 = SWIZZLE_64B, 6 = SWIZZLE_32B;
    // a 64-byte inner box with SWIZZLE_64B needs no padding, which is exactly why the
    // vendor's W sub-tile measures 128 rows x 64 B = 8192 B. The 16-byte chunks inside the
    // 64-byte span are then XOR-permuted by row%4 (a 2-bit swizzle).
    const int chunk = col >> 4;          // 16-byte chunk inside the row (0..3)
    const int within = col & 15;         // byte inside the chunk
    return (row >> 3) * 512 + (row & 7) * 64 +
           ((((chunk) ^ (row & 3))) << 4) + within;
}

// smem index for a (row, kk) element of a 128-row x 128-K(tiles) operand tile.
__device__ __forceinline__ int hw_smem_idx(int row, int kk, int canon) {
    if (canon) {
        // canonical UMMA K-major interleave (dsv41_experts_mxf4.cu:24-38):
        //   unit16(row,kb) = (row%8) + 8*kb + 16*(row/8); atom(K=32) = 4096 B block
        return (kk >> 5) * 4096 + (row >> 3) * 256 + (row & 7) * 16 +
               (((kk & 31) >> 4) * 128) + (kk & 15);
    }
    // SW128: addr(r,c) = (r/8)*1024 + (r%8)*128 + (((c/16) ^ (r%8))*16) + (c%16)
    return (row >> 3) * 1024 + (row & 7) * 128 + ((((kk >> 4) ^ (row & 7))) << 4) + (kk & 15);
}

// VERIFIED MMA instruction — memory clobber included
__device__ __forceinline__ void hw_tc_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                          uint32_t idesc, uint32_t sfa_tmem,
                                          uint32_t sfb_tmem, uint32_t enable_d) {
    if (g_sv1x) {
        // Repo-verified spelling (dsv41_experts_mxf4.cu:4344)
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
    d |= (uint32_t)(sf_id & 3) << 4;                   // b_sf_id [4,6)  ← FIX: was missing <<4
    d |= (uint32_t)(a_fmt & 7) << 7;                   // a_format [7,10) — 0=E4M3
    d |= (uint32_t)(b_fmt & 7) << 10;                  // b_format [10,13) — 5=E2M1
    d |= (uint32_t)((n >> 3) & 63) << 17;              // n_dim [17,23) — N/8
    d |= (uint32_t)1 << 23;                             // scale_format = UE8M0
    d |= (uint32_t)((m >> 4) & 31) << 24;              // m_dim [24,29) — M/16
    d |= (uint32_t)(sf_id & 3) << 29;                  // a_sf_id [29,31)
    return d;
}

// SF 转置（从 TileLang tcgen05_sf_warp_transpose 复制——4×32 uint32 块内转置）
//
// ⚠️ 这里的 `^ (lane >> 3)` 出现**两次**（读一次、写一次）⇒ 两次抵消：
//     post[lane*4 + (i^(l>>3))] = pre[(i^(l>>3))*32 + lane]
//   ⇒ 令 j = i^(l>>3)，即 post[lane*4 + j] = pre[j*32 + lane]
//   ⇒ **与 DeepGEMM 的 `utccp_required_smem_warp_transpose`（无 XOR）逐点等价** ✓
//     （DeepGEMM = SGLang 在 SM100/103 跑同款模型的生产实现，`sm100_fp8_fp4_gemm_1d1d.cuh`。）
//   ⇒ 曾经据"抄来的多了一个 XOR"判它是缺陷是**误读**：只删一处 XOR 才会破坏对称 ✗。本条已排除。
__device__ __forceinline__ void hw_sf_transpose(uint32_t *smem_ptr) {
    const uint32_t lane = threadIdx.x % 32;
    uint32_t values[4];
    for (uint32_t i = 0; i < 4; ++i)
        values[i] = smem_ptr[(i ^ (lane >> 3)) * 32 + lane];
    __syncwarp();
    for (uint32_t i = 0; i < 4; ++i)
        smem_ptr[lane * 4 + (i ^ (lane >> 3))] = values[i];
}

// SF 写 TMEM 的**寄存器路径**（`tcgen05.st`）—— 与隔离仪器 `tests_bs_impulse.cu` 用的同一条
// （那里 PASS 过 const/sfprobe/random_sf）。默认不使用：生产走"smesh→转置→tcgen05.cp"，
// 而**那条链从未被独立验证过**（两份审计独立指出）。`DSV41_MOE_BS_SFST=1` 切到本路径，用来
// 一次性判定"SF 投递链是不是缺陷"：仪器布局 = 第 (32j+l) 行的字放在 (lane l, column j)，
// 四个 partition 内容相同（`tcgen05.st` 每个 warp 只能写自己那 32 条 lane）。
__device__ __forceinline__ void hw_tc_st_x4(uint32_t taddr, uint32_t w0, uint32_t w1,
                                           uint32_t w2, uint32_t w3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr),
                 "r"(w0), "r"(w1), "r"(w2), "r"(w3)
                 : "memory");
}
__device__ __forceinline__ void hw_tc_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// TMEM 读的**仪器写法**（`tests_bs_impulse.cu` 用的同一条，`.32x32b.x8` + 显式 warp 偏移地址）。
// 默认不使用：生产用 TileLang 的 `tcgen05_ld_32dp32bNx<128>`（地址 lane 字段恒 0）。
// `DSV41_MOE_BS_LDW=1` 切到本路径，用来判定"lane 字段是否必须显式加 warp*32"。
__device__ __forceinline__ void hw_tc_ld_x8(uint32_t taddr, uint32_t *v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7])
        : "r"(taddr)
        : "memory");
}
__device__ __forceinline__ void hw_tc_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
// whp_* — BOUNDED WAITS (2026-09-13)
// ============================================================
// The incident this hardens against:
//
//   serve hung intermittently, the log flooding
//     `[ar5-hang] rank=2 site=0 peer=6 need=2 cur=1 spins>5000000 TIMEOUT -> PARK`
//   with ZERO `[dsv41] step pos=` lines in the whole round while a control round
//   (same env, same build) produced 48 step lines. The all-reduce wait is itself
//   BOUNDED — it PARKS after 5e6 spins — so that line is a SYMPTOM: some rank
//   never published its stamp. A rank that never RETURNS FROM HERE never reaches
//   its next all-reduce either, so every peer parks on it. That is exactly the
//   observed shape, and exactly intermittent: it needs the MMA completion to fail
//   to arrive ONE time.
//
//   The waiting point able to do that is the MMA-completion wait, which was an
//   UNBOUNDED spin
//       WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [bar], phase;
//       @!P bra WAIT;
//   executed by tid 0 alone, with the other 127 threads parked on the
//   __syncthreads() that follows. If the commit never arrived — MMA skipped,
//   issue/commit/wait count mismatch, phase desync, an early exit before the
//   issue — P stayed false forever: no thread ever passed that barrier, the
//   kernel never returned, the stream never advanced.
//
// Design contract (see docs/agent/moe-bs-crash-investigation.md §93):
//   * BOUNDEDNESS IS UNCONDITIONAL. No gate can restore an infinite spin; a
//     hang is never the thing anyone wants, and this is a safety property, not
//     an experiment.
//   * THE NORMAL PATH IS UNCHANGED. In the steady state the FIRST `try_wait`
//     returns true, so the loop costs one compare + one branch per K-iteration
//     and adds no memory traffic, no sleep and no synchronization.
//   * THE FAILURE IS NEVER SILENT. The one-line "bounded wait expired" report is
//     NOT gated (rate-limited to 8 prints, like `ar5_timeout`): this project's
//     red line is silent wrongness, and a silent early exit would hand the caller
//     garbage C. Only the VERBOSE state dump is behind `DSV41_MOE_BS_WAITDBG=1`
//     (default OFF).
__device__ int g_waitdbg = 0;                         // DSV41_MOE_BS_WAITDBG
// g_bounded_wait: DSV41_MOE_BS_BOUNDED_WAIT (DEFAULT OFF, 2026-09-14 regression fix).
// The bounded MMA wait was made ALWAYS-ON by the §92/§101 patch, and that is exactly when this
// campaign first saw the serve hang (the user's anchor: "之前从来没卡过"). The stripped-env
// control arm still hung, so the env is exonerated and the always-on wait change is the prime
// suspect. DEFAULT is therefore restored to the ORIGINAL unbounded wait — which this project
// used for its entire history without hanging — and the bounded variant is kept, gated, because
// the no-revert rule says a change is turned OFF, never deleted.
__device__ int g_bounded_wait = 0;                    // DSV41_MOE_BS_BOUNDED_WAIT
__device__ unsigned long long g_wait_abort_n = 0ull;  // expired waits, whole grid
// Spin cap for ONE MMA-completion wait. A worst-case MMA group is O(10 us) and a
// probe is O(10 ns), so this is ~4 orders of magnitude of headroom; the property
// that matters is that it is FINITE.
#define WHP_MMA_SPIN_CAP (1u << 22)

// One mbarrier parity probe: the SAME instruction as the original loop, but the
// result is returned to C instead of driving an asm-level back-branch. The bound
// therefore lives in readable C — a counter inside an asm template is the kind
// of thing that breaks silently.
__device__ __forceinline__ bool whp_mbar_probe(uint64_t* bar, uint32_t phase) {
    uint32_t done;
    asm volatile("{\n\t.reg .pred P;\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
                 "selp.b32 %0, 1, 0, P;\n\t}"
                 : "=r"(done)
                 : "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
    return done != 0u;
}

// Raw 64-bit read of the mbarrier word — DIAGNOSTIC ONLY. The mbarrier's internal
// layout is opaque ("implementation-specific"), so the value is dumped verbatim
// and NOT decoded into claims about pending counts.
__device__ __forceinline__ unsigned long long whp_ld_smem_u64(const void* p) {
    unsigned long long v;
    asm volatile("ld.volatile.shared.u64 %0, [%1];"
                 : "=l"(v) : "r"((uint32_t)__cvta_generic_to_shared(p)));
    return v;
}

// Report ONE expired MMA wait. `k`/`phase`/`seg`/`n_tile`/warp/lane pin the
// iteration that failed. The well-defined extra datum is the probe on the OTHER
// parity: if the barrier has completed parity `phase ^ 1` instead, the arrival
// happened but the parity arithmetic is off (a double arrival or a phase desync);
// if neither parity is ready, the arrival never happened at all.
__device__ __noinline__ void whp_mma_timeout_report(
    uint64_t* bar, uint32_t phase, int k, int seg, int n_tile, int warp, int lane) {
    const unsigned long long nth = atomicAdd(&g_wait_abort_n, 1ull) + 1ull;
    if (nth <= 8ull) {
        printf("[moe-bs-wait] TIMEOUT mma-arrive: block=(%d,%d) k=%d phase=%u warpid=%d "
               "lane=%d spins>%u -> ABORT (bounded wait: the kernel bails out instead "
               "of spinning forever)\n",
               seg, n_tile, k, phase, warp, lane, (unsigned)WHP_MMA_SPIN_CAP);
    }
    if (g_waitdbg) {
        const int other_ready = whp_mbar_probe(bar, phase ^ 1u) ? 1 : 0;
        printf("[moe-bs-wait-dbg] nth=%llu block=(%d,%d) k=%d expected_parity=%u "
               "other_parity_ready=%d mbar_raw=0x%016llx seg_idx=%d n_tile=%d warpid=%d "
               "lane=%d clock=%lld abort_n=%llu\n",
               nth, seg, n_tile, k, phase, other_ready, whp_ld_smem_u64(bar),
               seg, n_tile, warp, lane, clock64(), g_wait_abort_n);
    }
}

// BOUNDED MMA-completion wait. Returns true when the wait EXPIRED (the caller
// must stop and must NOT touch the TMEM accumulator).
__device__ __forceinline__ bool whp_mma_wait_bounded(
    uint64_t* bar, uint32_t phase, int k, int seg, int n_tile) {
    for (uint32_t spins = 0;; ++spins) {
        if (whp_mbar_probe(bar, phase)) return false;
        if (spins >= WHP_MMA_SPIN_CAP) {
            whp_mma_timeout_report(bar, phase, k, seg, n_tile, threadIdx.x >> 5,
                                   threadIdx.x & 31);
            return true;
        }
    }
}

// ============================================================
// cp.async primitives (gate DSV41_MOE_BS_CPASYNC — see g_cpasync)
// ============================================================
// Only the non-bulk `cp.async` family is used here (sm_80 style). It is NOT the
// same thing as the TMA/`cp.async.bulk` + mbarrier pipeline that crashed this
// kernel historically: there is no descriptor, no mbarrier, no multi-stage
// barrier object — completion is observed with the per-thread group counter
// (`cp.async.commit_group` / `cp.async.wait_group`) plus a plain __syncthreads().
//
// cp-size restrictions (PTX ISA "cp.async"):
//   * `.cg` (cache global, bypass L1) supports ONLY cp-size 16;
//   * `.ca` (cache all levels) supports cp-size 4, 8, 16.
// src and dst must both be aligned to cp-size — the per-operand alignment is
// checked at runtime before the copies are issued (see hw_issue_stage).
__device__ __forceinline__ void hw_cp_async16(void *smem_dst, const void *gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)), "l"(gmem_src));
}
__device__ __forceinline__ void hw_cp_async8(void *smem_dst, const void *gmem_src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)), "l"(gmem_src));
}
__device__ __forceinline__ void hw_cp_async4(void *smem_dst, const void *gmem_src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)), "l"(gmem_src));
}
// Close the current per-thread group (all cp.asyncs issued since the last commit).
__device__ __forceinline__ void hw_cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
// Wait until at most N of THIS thread's groups are still in flight. N must be a
// compile-time immediate, hence the template (one instantiation per call site).
template <int N>
__device__ __forceinline__ void hw_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

// 16 B-chunk base offset of an operand tile row, i.e. hw_smem_idx(m, c*16) — the
// byte-wise formula collapses to `base + j` for the 16 bytes j = 0..15 of chunk c.
//   SW128 (layout_type=2): addr(m,kk) = (m/8)*1024 + (m%8)*128 + (((kk/16)^(m%8))*16) + (kk%16)
__device__ __forceinline__ int hw_smem_idx16_sw128(int m, int c) {
    return (m >> 3) * 1024 + (m & 7) * 128 + (((c ^ (m & 7)) & 7) << 4);
}
//   canonical (layout_type=0): (kk/32)*4096 + (m/8)*256 + (m%8)*16 + ((kk%32)/16)*128 + (kk%16)
__device__ __forceinline__ int hw_smem_idx16_canon(int m, int c) {
    return (c >> 1) * 4096 + (m >> 3) * 256 + (m & 7) * 16 + (c & 1) * 128;
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
constexpr int HW_NT = HW_NUP / HW_BN;  // 5 — the n_tile count (the shim's `kGridX`)
constexpr int HW_SEGCAP = 36;
constexpr int HW_KITER = HW_K / HW_BK;  // 40

// ============================================================
// cp.async staging of ONE K-iteration into one stage (gate DSV41_MOE_BS_CPASYNC)
// ============================================================
// Byte-for-byte identical to the three sequential loader loops in the k-loop body
// (same source addresses, same smem addresses) — only the transport differs:
//   * A tile   : 128 rows x 128 B e4m3 = 1024 x 16 B chunks, 8 per thread.
//                Global src  = A + row*5120 + k*128 + c*16  (5120 = 320*16 ⇒ 16 B aligned)
//                Smem  dst   = (m/8)*1024 + (m%8)*128 + ((c ^ (m%8))*16)   [SW128]
//                            = (c/2)*4096 + (m/8)*256 + (m%8)*16 + (c%2)*128 [canon]
//                Both are exactly the 16 bytes hw_smem_idx(m, c*16..c*16+16).
//   * B tile   : packed fp4, 2 planes x 64 rows x 64 packed bytes = 1024 x 8 B
//                chunks. A packed row is 8 containers of 16 B of which only the
//                first 8 B are read (§47), so the natural copy granularity is
//                8 B: global src = ... + c*8 lands at the container's byte 0..7.
//                dst = (row/8)*1024 + (row%8)*128 + ((c ^ (row%8))*16), and W3
//                (rows 64..127) is the same formula + 8192 B.
//                ⇒ 16 B copies are NOT usable here: the 16 B container swizzle
//                would have to be undone (and would fetch the 8 dead bytes too).
//   * SF       : 1 x 4 B per thread for the activation SF (128 words) and 2 x 4 B
//                for threads 0..63 (SFW1 + SFW3, disjoint global locations).
//                A 4 B cp.async is the smallest size available.
__device__ __forceinline__ void hw_issue_stage(
    int k,
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ W1,
    const uint8_t* __restrict__ W3,
    const uint32_t* __restrict__ SFA, const uint32_t* __restrict__ SFW1,
    const uint32_t* __restrict__ SFW3,
    int seg, int n_tile, int e, int64_t w_stride, int tid,
    uint8_t* A_s, uint8_t* B_s, uint32_t* SFA_s, uint32_t* SFB_s,
    bool a_cp_ok, bool w_cp_ok, bool sf_cp_ok)
{
    // swapAB flips which smem tile holds which operand (A_s = the MMA's A operand)
    uint8_t* act_sh = g_swapab ? B_s : A_s;    // e4m3 activation tile (128 x 128 B)
    uint8_t* w_sh   = g_swapab ? A_s : B_s;    // packed fp4 weight tile (128 x 64 B data)
    uint32_t* sf_act = g_swapab ? SFB_s : SFA_s;
    uint32_t* sf_w   = g_swapab ? SFA_s : SFB_s;

    // (1) activation tile — 16 B chunks (cp.async.cg, the only size .cg supports)
    if (a_cp_ok) {
        for (int i = tid; i < HW_BM * 8; i += 128) {
            const int m = i >> 3;   // activation row [0,128)
            const int c = i & 7;    // 16 B chunk inside the 128 B K-tile (= tid&7)
            const uint8_t* src = A + (int64_t)(seg * HW_BM + m) * HW_K + (int64_t)k * HW_BK + c * 16;
            const int dst = g_canon ? hw_smem_idx16_canon(m, c) : hw_smem_idx16_sw128(m, c);
            hw_cp_async16(act_sh + dst, src);
        }
    } else {
        // unaligned base pointer (cannot happen for g_a which comes from cudaMalloc,
        // but cp.async on a misaligned address is UB) -> byte-wise LDG/STS
        for (int i = tid; i < HW_BM * HW_BK; i += 128) {
            const int m = i >> 7;
            const int kk = i & 127;
            act_sh[hw_smem_idx(m, kk, g_canon)] =
                A[(int64_t)(seg * HW_BM + m) * HW_K + k * HW_BK + kk];
        }
    }

    // (2) weight tile (packed fp4): 8 B chunks, W1 rows 0..63 and W3 rows 64..127
    if (g_packed) {
        if (w_cp_ok) {
            for (int i = tid; i < HW_NH * 8; i += 128) {
                const int row = i >> 3;  // in-plane row [0,64)
                const int c = i & 7;     // 8 B chunk inside the 64 B packed row
                const int64_t off = (int64_t)e * w_stride +
                                    (int64_t)(n_tile * HW_NH + row) * 2560 +
                                    (int64_t)k * 64 + c * 8;
                const int dst = (row >> 3) * 1024 + (row & 7) * 128 + (((c ^ (row & 7)) & 7) << 4);
                hw_cp_async8(w_sh + dst, W1 + off);
                hw_cp_async8(w_sh + dst + 8192, W3 + off);   // m = 64+row ⇒ +8 row-groups*1024
            }
        } else {
            // w_stride (caller-measured expert stride) is not a multiple of 8 ⇒ the
            // src would not satisfy cp.async's alignment rule. Same bytes, plain loads.
            for (int i = tid; i < HW_NH * 64; i += 128) {
                const int row = i >> 6;
                const int col = i & 63;
                const int64_t off = (int64_t)e * w_stride +
                                    (int64_t)(n_tile * HW_NH + row) * 2560 + (int64_t)k * 64 + col;
                w_sh[hw_pack_sw128(row, col)] = W1[off];
                w_sh[hw_pack_sw128(HW_NH + row, col)] = W3[off];
            }
        }
    } else {
        // unpacked escape hatch (measured WRONG, §47): the byte -> 2-nibble split is a
        // transform, not a copy, so cp.async cannot express it -> plain loads.
        for (int i = tid; i < HW_NH * 64; i += 128) {
            const int row = i >> 6;
            const int col = i & 63;
            const int64_t off = (int64_t)e * w_stride +
                                (int64_t)(n_tile * HW_NH + row) * 2560 + (int64_t)k * 64 + col;
            const uint8_t p1 = W1[off];
            const uint8_t p3 = W3[off];
            const int k0 = col * 2;
            const int k1 = col * 2 + 1;
            w_sh[hw_smem_idx(row, k0, g_canon)] = p1 & 0xF;
            w_sh[hw_smem_idx(row, k1, g_canon)] = p1 >> 4;
            w_sh[hw_smem_idx(HW_NH + row, k0, g_canon)] = p3 & 0xF;
            w_sh[hw_smem_idx(HW_NH + row, k1, g_canon)] = p3 >> 4;
        }
    }

    // (3) SF — 4 B words (the group-major packed scales, one u32 per K-block)
    if (sf_cp_ok) {
        if (tid < HW_BM) {
            hw_cp_async4(sf_act + tid,
                         SFA + (int64_t)k * (HW_SEGCAP * HW_BM) + seg * HW_BM + tid);
        }
        if (tid < HW_NH) {
            const int64_t off = (int64_t)e * (40 * HW_NP) + (int64_t)k * HW_NP +
                                n_tile * HW_NH + tid;
            hw_cp_async4(sf_w + tid, SFW1 + off);
            hw_cp_async4(sf_w + HW_NH + tid, SFW3 + off);
        }
    } else {
        if (tid < HW_BM) {
            sf_act[tid] = SFA[(int64_t)k * (HW_SEGCAP * HW_BM) + seg * HW_BM + tid];
        }
        if (tid < HW_NH) {
            const int64_t off = (int64_t)e * (40 * HW_NP) + (int64_t)k * HW_NP +
                                n_tile * HW_NH + tid;
            sf_w[tid] = SFW1[off];
            sf_w[HW_NH + tid] = SFW3[off];
        }
    }
}

// ============================================================
// Sequential staging of ONE K-iteration (the pre-2026-09-14 path, verbatim)
// ============================================================
// This is a pure refactor of the three loader loops that used to live inline in
// the k-loop body. Called with the single-buffer pointers (NS == 1, s == 0) it
// writes exactly the same bytes to exactly the same smem addresses in exactly the
// same order as before, so the gate-OFF path is unchanged bit for bit.
__device__ __forceinline__ void hw_load_stage_seq(
    int k,
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ W1,
    const uint8_t* __restrict__ W3,
    const uint32_t* __restrict__ SFA, const uint32_t* __restrict__ SFW1,
    const uint32_t* __restrict__ SFW3,
    int seg, int n_tile, int e, int64_t w_stride, int tid,
    uint8_t* A_s, uint8_t* B_s, uint32_t* SFA_s, uint32_t* SFB_s)
{
    uint8_t* act_sh = g_swapab ? B_s : A_s;
    uint8_t* w_sh   = g_swapab ? A_s : B_s;
    uint32_t* sf_act = g_swapab ? SFB_s : SFA_s;
    uint32_t* sf_w   = g_swapab ? SFA_s : SFB_s;

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
        act_sh[hw_smem_idx(m, kk, g_canon)] = val;
    }

    // (2) 所有线程协同加载 fp4 操作数（= W1 前 64 行 + W3 后 64 行）
    //     门控 OFF：unpacked 1 字节/元素（**实测是错的**——硬件按 packed 读，见 §29），
    //     门控 DSV41_MOE_BS_PACKED=1：整字节写入（2 元素/字节），几何见 hw_pack_idx
    // B = W1 前 64 行 + W3 后 64 行
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
        if (g_packed) {
            // the hardware reads two elements per byte, in 16 B containers of which
            // only the first 8 B are read -> write the source packed byte as-is at the
            // authoritative offset (MEASURED exact-parity layout, see hw_pack_sw128)
            w_sh[hw_pack_sw128(row, col)] = packed;
        } else {
            const int k0 = col * 2;      // first element K index
            const int k1 = col * 2 + 1;  // second element K index
            // SW128 swizzled write (same layout as TileLang's TMA)
            w_sh[hw_smem_idx(row, k0, g_canon)] = packed & 0xF;
            w_sh[hw_smem_idx(row, k1, g_canon)] = packed >> 4;
        }
    }
    for (int i = tid; i < HW_NH * 64; i += 128) {
        const int row = i >> 6;
        const int col = i & 63;
        const uint8_t packed =
            W3[(int64_t)e * w_stride + (int64_t)(n_tile * HW_NH + row) * 2560 + k * 64 + col];
        const int m = HW_NH + row;  // W3 rows are after W1
        if (g_packed) {
            w_sh[hw_pack_sw128(m, col)] = packed;
        } else {
            const int k0 = col * 2;
            const int k1 = col * 2 + 1;
            w_sh[hw_smem_idx(m, k0, g_canon)] = packed & 0xF;
            w_sh[hw_smem_idx(m, k1, g_canon)] = packed >> 4;
        }
    }

    // (3) 加载 SF
    // SFA: [40, 4608] — SFA[k][seg*128 + i] for i in [0, 128)
    for (int i = tid; i < HW_BM; i += 128) {
        sf_act[i] = SFA[(int64_t)k * (HW_SEGCAP * HW_BM) + seg * HW_BM + i];
    }
    // SFW1/SFW3: [384, 40*320] — SFW[e][k*320 + n_tile*64 + i] for i in [0, 64)
    for (int i = tid; i < HW_NH; i += 128) {
        sf_w[i] = SFW1[(int64_t)e * (40 * HW_NP) + k * HW_NP + n_tile * HW_NH + i];
        sf_w[HW_NH + i] = SFW3[(int64_t)e * (40 * HW_NP) + k * HW_NP + n_tile * HW_NH + i];
    }
}

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
    // NS = g_cpasync ? 2 : 1 stages. With NS == 1 (the default) every offset below
    // evaluates to the historical sequential value, byte for byte:
    //   A: [0, 16384) e4m3 tile            B: [16384, 32768) packed fp4 tile
    //   SFA: [32768, 33280) activation SF  SFB: [33280, 33792) weight SF
    //   mbar: [33792, 33800) MMA completion barrier
    //   C staging: [0, 65536) f32 output (overlaps A/B/SF/mbar — written after all MMAs)
    //   Total: 65536 bytes (dominated by C)
    // With NS == 2 only the OPERAND region doubles (stage stride = the NS==1 size):
    //   A: [0, 32768) | B: [32768, 65536) | SFA: [65536, 66560) | SFB: [66560, 67584)
    //   mbar: [67584, 67592)          ⇒ operand region = 67592 B
    //   C staging is STILL [0, 65536) and therefore still aliases the operands: the
    //   epilogue runs after the last mbarrier wait, all 40 MMA groups have completed
    //   and the final wait_group 0 has drained every cp.async ⇒ no reader is left.
    //   Footprint = max(65536, 67592) = 67592 B < 227 KiB/block ✓ (and ≤ the 65536 B
    //   C region + one stage, so 3 CTAs/SM still fit: 3*67592 = 202776 B of 227 KiB).
    extern __shared__ __align__(1024) uint8_t hw_smem[];
    const int NS = g_cpasync ? 2 : 1;
    uint8_t* A_sh = hw_smem;                                   // NS stages × [128, 128] e4m3
    uint8_t* B_sh = hw_smem + NS * 16384;                      // NS stages × [128, 64] packed fp4
    uint32_t* SFA_sh = (uint32_t*)(hw_smem + 2 * NS * 16384);  // NS × [128] activation SF
    uint32_t* SFB_sh = SFA_sh + NS * 128;                      // NS × [128] weight SF
    float* C_sh = (float*)hw_smem;                             // [128, 128] f32 (aliases the operand region — safe: written after all MMAs)
    // mbarrier for MMA completion — placed AFTER SF (one per kernel, NOT per stage:
    // there is exactly one tcgen05.commit per K-iteration, so the parity alternation
    // k&1 is unchanged from the sequential path)
    uint64_t* mma_bar = (uint64_t*)(SFB_sh + NS * 128);        // 8 bytes

    // ---- TMEM allocation (warp 0 only) ----
    __shared__ __align__(16) uint hw_C_tmem;
    __shared__ __align__(16) uint hw_SF_tmem;
    // Bounded-wait failure latch (see the whp_* block at the top of the file):
    // written by tid 0 BEFORE the per-iteration barrier and read by ALL 128 threads
    // right AFTER it, so the entire block leaves the k-loop together and the later
    // __syncthreads() calls stay non-divergent. `volatile` so the load cannot be
    // hoisted out of the k-loop.
    __shared__ volatile int hw_wait_fail;
    if (warp == 0) {
        hw_tc_alloc(&hw_C_tmem, 128);   // 128 columns for C
        hw_tc_alloc(&hw_SF_tmem, 32);    // 32 columns for SF
        hw_tc_relinquish();
    }
    // tcgen05.alloc writes the allocated TMEM base into smem; the generic read of
    // hw_C_tmem/hw_SF_tmem below must be ordered against it — TileLang wraps exactly
    // this barrier with the tcgen05 thread-sync fence pair (moe_bs_up_tl.cu:79/81).
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t C_tmem = hw_C_tmem;
    const uint32_t SF_tmem = hw_SF_tmem;

    // ---- init mbarrier (1 arrival from tcgen05.commit) ----
    if (tid == 0) {
        hw_wait_fail = 0;   // no wait has expired yet; published by the barrier below
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"((uint32_t)__cvta_generic_to_shared(mma_bar)));
        // mbarrier-init visibility uses the DEDICATED fence and must come BEFORE the
        // first barrier (TileLang: moe_bs_up_tl.cu:63 tl::fence_barrier_init() is
        // emitted ahead of tl:65's __syncthreads).
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");  // async-proxy view of the smem operands

    // ---- K-loop: cp.async double-buffered (gate ON) / sequential (gate OFF) ----
    // cp.async alignment predicates (uniform across the block, evaluated once):
    //   * A : src = A + row*5120 + k*128 + c*16 — every term is a multiple of 16
    //         (5120 = 320*16) and A comes from cudaMalloc ⇒ 16 B aligned. The
    //         destination 16 B chunks are 16 B aligned by construction.
    //   * B : src = W1/W3 + e*w_stride + row*2560 + k*64 + c*8 — 2560/64/8 are all
    //         multiples of 8, but w_stride is CALLER-MEASURED, so it is checked.
    //   * SF: 4 B words; base pointers are cudaMalloc'd (256 B aligned) and every
    //         offset is expressed in u32 units.
    // A failing predicate falls back to plain LDG/STS FOR THAT OPERAND ONLY — same
    // bytes, same smem addresses, just latency-exposed (no numerics difference).
    const bool a_cp_ok  = g_cpasync && (((uintptr_t)A & 15u) == 0u);
    const bool w_cp_ok  = g_cpasync && g_packed &&
                          ((((uintptr_t)W1 | (uintptr_t)W3) & 7u) == 0u) &&
                          (((uint64_t)w_stride & 7u) == 0u);
    const bool sf_cp_ok = g_cpasync &&
                          ((((uintptr_t)SFA | (uintptr_t)SFW1 | (uintptr_t)SFW3) & 3u) == 0u);

    if (g_cpasync) {
        // PROLOGUE: stage 0 holds k = 0 — the one load that cannot be overlapped
        hw_issue_stage(0, A, W1, W3, SFA, SFW1, SFW3, seg, n_tile, e, w_stride, tid,
                       A_sh, B_sh, SFA_sh, SFB_sh, a_cp_ok, w_cp_ok, sf_cp_ok);
        hw_cp_async_commit();
    }

    for (int k = 0; k < HW_KITER; ++k) {
        // DSV41_MOE_BS_STAGE1=1: collapse production to the isolation instrument's regime — ONE
        // 128-K stage. The instrument covers exactly one stage and PASSES; production runs 40, so
        // this splits "even a single stage is wrong in production" (host-side ingest / staging)
        // from "the stage-to-stage progression is wrong". `k` is block-uniform, so the branch and
        // the break are legal.
        if (g_stage1 && k > 0) break;
        const int s = g_cpasync ? (k & 1) : 0;
        uint8_t* A_s = A_sh + (size_t)s * HW_BM * HW_BK;   // this iteration's A stage
        uint8_t* B_s = B_sh + (size_t)s * HW_BM * HW_BK;
        uint32_t* SFA_s = SFA_sh + s * HW_BM;
        uint32_t* SFB_s = SFB_sh + s * HW_BM;

        if (g_cpasync) {
            // (0) PREFETCH k+1 into the OTHER stage, then wait for stage s. Buffer
            //     (1-s) was last read by iteration k-1, whose MMAs completed at the
            //     end of that iteration (mbarrier wait + __syncthreads) ⇒ it is free.
            if (k + 1 < HW_KITER) {
                hw_issue_stage(k + 1, A, W1, W3, SFA, SFW1, SFW3, seg, n_tile, e, w_stride, tid,
                               A_sh + (size_t)(1 - s) * HW_BM * HW_BK,
                               B_sh + (size_t)(1 - s) * HW_BM * HW_BK,
                               SFA_sh + (1 - s) * HW_BM, SFB_sh + (1 - s) * HW_BM,
                               a_cp_ok, w_cp_ok, sf_cp_ok);
                hw_cp_async_commit();
                hw_cp_async_wait<1>();   // the prefetch group stays in flight; stage s is done
            } else {
                hw_cp_async_wait<0>();   // last iteration: nothing prefetched, drain all
            }
        } else {
            // the pre-2026-09-14 sequential staging, verbatim (see hw_load_stage_seq)
            hw_load_stage_seq(k, A, W1, W3, SFA, SFW1, SFW3, seg, n_tile, e, w_stride, tid,
                              A_s, B_s, SFA_s, SFB_s);
        }

        // Block-wide visibility of stage s: every thread's own cp.async group has
        // completed (wait_group above) and this barrier orders those writes against
        // the SF transpose / fence / MMA that follow. In the gate-OFF path it is the
        // exact same single barrier that used to follow the LDG/STS loops.
        __syncthreads();

        // ---- OPT-IN (DSV41_MOE_BS_SFDUMP=1) one-shot semantic dump of the STAGED operands ----
        // Read back THROUGH the validated smem formulas (`hw_smem_idx` / `hw_pack_sw128`), i.e.
        // exactly the bytes the hardware reads, and write them in the semantic (row, K) form.
        // The point is NOT the layout (the formulas are instrument-validated) but the CONTENT:
        // if the loader fetched the wrong global byte, this shows it as a wrong value at the
        // right place. The shim copies this scratch back once. Diagnostic only, uniform branch
        // (both operands are `k` and block-uniform globals), so the barrier stays legal.
        if (g_sfdump_on && k == g_sfdump_k && g_sfdump != nullptr) {
            __syncthreads();
            uint8_t* dump_act = g_swapab ? B_s : A_s;
            uint8_t* dump_w = g_swapab ? A_s : B_s;
            uint32_t* dump_sfa = g_swapab ? SFB_s : SFA_s;
            uint32_t* dump_sfb = g_swapab ? SFA_s : SFB_s;
            if (n_tile == 0) {
                for (int i = tid; i < HW_BM * HW_BK; i += 128) {
                    const int m = i >> 7, kk = i & 127;
                    g_sfdump[kSfDumpA + (size_t)seg * HW_BM * HW_BK + (size_t)m * HW_BK + kk] =
                        dump_act[hw_smem_idx(m, kk, g_canon)];
                }
            }
            for (int i = tid; i < HW_BM * 64; i += 128) {
                const int r = i >> 6, col = i & 63;
                g_sfdump[kSfDumpB + ((size_t)seg * HW_NT + n_tile) * HW_BM * 64 +
                         (size_t)r * 64 + col] = dump_w[hw_pack_sw128(r, col)];
            }
            if (n_tile == 0) {
                for (int i = tid; i < HW_BM; i += 128) {
                    reinterpret_cast<uint32_t*>(g_sfdump + kSfDumpSfa)[(size_t)seg * HW_BM + i] =
                        dump_sfa[i];
                }
            }
            for (int i = tid; i < HW_BM; i += 128) {
                reinterpret_cast<uint32_t*>(g_sfdump + kSfDumpSfb)
                    [((size_t)seg * HW_NT + n_tile) * HW_BM + i] = dump_sfb[i];
            }
            __syncthreads();
        }

        // (4) SF → TMEM，两条路：
        //   默认（g_sfst=0）：warp 2 转置 staged 字，再由 warp 1 lane 0 用 tcgen05.cp 搬进 TMEM
        //                     —— 生产链，**从未被独立验证过**；
        //   g_sfst=1        ：每个 warp 用 tcgen05.st 把自己那 32 条 lane 的字直接写进 TMEM，
        //                     布局 = 隔离仪器 PASS 过的那一种（第 32j+l 行的字 → lane l, column j，
        //                     四个 partition 内容相同）。DSV41_MOE_BS_SFST=1，默认 OFF。
        if (g_sfst) {
            uint32_t wa[4], wb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wa[j] = SFA_s[32 * j + lane];
                wb[j] = SFB_s[32 * j + lane];
            }
            hw_tc_st_x4(((uint32_t)(warp * 32) << 16) | (SF_tmem + 0), wa[0], wa[1], wa[2], wa[3]);
            hw_tc_st_x4(((uint32_t)(warp * 32) << 16) | (SF_tmem + 4), wb[0], wb[1], wb[2], wb[3]);
            hw_tc_wait_st();
        } else if (warp == 2) {
            hw_sf_transpose(SFA_s);
            hw_sf_transpose(SFB_s);
        }
        __syncthreads();

        // (4.5) ★ ASYNC-PROXY FENCE (root cause fix 2026-09-14):
        // The MMA (tcgen05.mma) and the SF copy (tcgen05.cp) read shared memory through
        // the ASYNC proxy, while the operand stage (A_s / B_s / SFA_s / SFB_s — single
        // buffer A_sh/B_sh/SFA_sh/SFB_sh when the cp.async gate is OFF) was just
        // written with GENERIC accesses (LDG/STS stores OR cp.async copies). Per the
        // PTX memory model those generic-proxy writes must be made visible to the async
        // proxy with `fence.proxy.async` — TileLang's generated
        // code does exactly this (tl::fence_proxy_async() in its consumer path) but our
        // hand-written kernel only fenced once outside the k-loop. Without it the MMA
        // can read STALE smem from the previous k-iteration/launch. This is invisible
        // with uniform test data (stale == current) and produces garbage on real data.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        // Highest-suspicion gap: this is the ONLY tcgen05 fence TileLang places on the
        // "thread sync -> subsequent async tcgen05 op" direction (moe_bs_up_tl.cu:107,
        // right after the loaded/sf_full waits and before the cp/mma issue). Our kernel
        // went straight from the barrier to tcgen05.cp / tcgen05.mma.
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

        // (5) warp 1: SF copy to TMEM + MMA
        if (warp == 1) {
            // SF copy: SFA → SF_tmem+0, SFB → SF_tmem+4
            // (elect_one_sync: single thread issues the copy)
            if (lane == 0) {
                if (!g_sfst) {
                    hw_tc_cp(hw_make_sf_desc(SFA_s), SF_tmem + 0);
                    hw_tc_cp(hw_make_sf_desc(SFB_s), SF_tmem + 4);
                }
            }

            // MMA: 4 sub-MMAs (ki=0-3, each covers 32 K elements)
            // smem descriptors for A and B tiles — the layout family is a RUNTIME choice:
            //   g_canon=0 -> TileLang SW128  : lbo=1, sbo=64, layout_type=2 (SWIZZLE_128B),
            //                 writes use addr(r,c)=(r/8)*1024+(r%8)*128+(((c/16)^(r%8))*16)+(c%16)
            //   g_canon=1 -> repo canonical  : lbo=8, sbo=16, layout_type=0 (SWIZZLE_NONE),
            //                 writes use unit16(r,kb)=(r%8)+8*kb+16*(r/8), atom = 4096 B
            // The two parameter sets are NOT interchangeable — each must pair with its own
            // write formula (hw_smem_idx) and its own K-block advance (see below).
            // The fp4 operand (A under swapAB, else B) needs the PACKED descriptor when
            // g_packed: one scale_vec::1X K-block is 16 smem bytes there, so the vendor's
            // own W descriptor numbers apply — lbo=1 (16 B), sbo=64 (1024 B), layout=2.
            const bool a_packed = g_packed && g_swapab;
            const bool b_packed = g_packed && !g_swapab;
            // packed-fp4 descriptor: geometry 0 -> lbo=1/sbo=64/layout=2; candidate B ->
            // lbo=8/sbo=32/layout=0 (§32)
            // geometry 0 -> lbo=1/sbo=64/layout=2 (SWIZZLE_128B, likely WRONG for 64 B rows)
            // candidate B -> lbo=8/sbo=32/layout=0
            // candidate D -> lbo=1/sbo=32/layout=4 (SWIZZLE_64B — the documented match for a
            //                64-byte inner box; see common.h:751-757)
            // geom 3 (candidate E) -> lbo=1 / sbo=32 / layout=0 (plain rows, SWIZZLE_NONE)
            // The packed layout pairs with the VENDOR's unchanged descriptor (1,64,2) and
            // the standard ki*32 B advance — measured exact PARITY with relerr = 0 when
            // combined with hw_pack_sw128, and FAILING for every other combination.
            const uint64_t a_pk = hw_make_desc(A_s, 1, 64, 2);
            const uint64_t b_pk = hw_make_desc(B_s, 1, 64, 2);
            const uint64_t a_desc_base = a_packed ? a_pk
                                     : (g_canon ? hw_make_desc(A_s, 8, 16, 0)    // canonical SWIZZLE_NONE
                                                : hw_make_desc(A_s, 1, 64, 2));  // TileLang SW128
            const uint64_t b_desc_base = b_packed ? b_pk
                                     : (g_canon ? hw_make_desc(B_s, 8, 16, 0)
                                                : hw_make_desc(B_s, 1, 64, 2));

            for (int ki = 0; ki < 4; ++ki) {
                // idesc: M=128, N=128, a_fmt=0 (E4M3), b_fmt=5 (E2M1), sf_id=ki
                // swapAB: A operand = weights (E2M1=5), B operand = activations (E4M3=0)
                // SFREV: if the hardware reads the SF word MSB-first (i.e. byte j holds the
                // scale for K-block 3-j) then selecting byte (3-ki) for K-atom ki corrects
                // BOTH the weight side (pack_wsf) and the activation side (gather) at once,
                // because both pack byte j = K-block j. This is the cheapest test of the one
                // hypothesis that can explain a systematic error present in EVERY
                // layout/orientation/SF-delivery combination.
                const int sf_id = g_sfrev ? (3 - ki) : ki;
                const uint32_t idesc = g_swapab ? hw_make_idesc(HW_BM, HW_BN, 5, 0, sf_id)
                                                : hw_make_idesc(HW_BM, HW_BN, 0, 5, sf_id);
                // K-block descriptor advance: TileLang's `desc_a + (ki*32)` where
                // Tcgen05SMemDescriptor::operator+ does `reg32_[0] += offset >> 4`
                // (offset in BYTES). So the advance is ki*32 bytes = ki*2 units.
                // packed fp4: a K-block is 16 B = 1 unit, so the advance is ki*1
                // packed K-block advance: geometry 0 -> 16 B = 1 unit; candidate B -> 128 B = 8
                const uint64_t a_pkadv = 2;   // ki*32 B: one MMA = two 16 B containers
                const uint64_t b_pkadv = 2;
                const uint64_t a_desc = a_desc_base + (uint64_t)(ki * (a_packed ? a_pkadv : (g_canon ? 256 : 2)));
                const uint64_t b_desc = b_desc_base + (uint64_t)(ki * (b_packed ? b_pkadv : (g_canon ? 256 : 2)));
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
        // ⚠️ BOUNDED since 2026-09-13 (see the whp_* block at the top of the file).
        // This used to be `WAIT: mbarrier.try_wait.parity ... @!P bra WAIT` — an
        // UNBOUNDED spin by tid 0 alone, with the other 127 threads parked on the
        // __syncthreads() below. If the commit never arrived, nothing ever passed
        // that barrier: the kernel never returned, the rank never reached its next
        // all-reduce, and every peer PARKed on it (the `[ar5-hang]` flood with zero
        // steps in the round). Now the spin is capped and a timeout is REPORTED and
        // the whole block takes the abort path together.
        if (tid == 0) {
            // mbarrier wait (phase 0 for first use, then alternating)
            if (g_bounded_wait) {
                if (whp_mma_wait_bounded(mma_bar, k & 1, k, seg, n_tile)) {
                    hw_wait_fail = 1;   // published to all 128 threads by the barrier below
                }
            } else {
                // DEFAULT (restored): the ORIGINAL unbounded wait used for this kernel's whole
                // history. Kept as the default because the always-on bounded wait is the prime
                // suspect for the regression that made the serve hang on its first request.
                const uint32_t phase = k & 1;
                asm volatile(
                    "{\n\t.reg .pred P;\n\t"
                    "WAIT:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                    "@!P bra WAIT;\n\t}"
                    :: "r"((uint32_t)__cvta_generic_to_shared(mma_bar)), "r"(phase));
            }
        }
        // tcgen05 thread-sync fences (TileLang's pattern): the MMA is an async tcgen05
        // operation, so ordering its completion with the barrier that lets other threads
        // overwrite the operand smem requires both fences around the sync.
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        // Every thread reads the SAME latch value here (__syncthreads() is the
        // shared-memory visibility edge, and the latch is volatile so the load cannot
        // be hoisted out of the k-loop) ⇒ the block exits the loop as a block, which
        // is what keeps the following __syncthreads() calls non-divergent.
        if (hw_wait_fail) break;

        // (7) 下一轮 k-iteration 会覆盖 smem——MMA 已完成，安全
    }

    // ---- ABORT PATH: the MMA completion never arrived inside the bounded window ----
    // Do NOT read TMEM (the accumulator holds an unknown mix of K-iterations) and do
    // NOT write C_sh: the only correct behaviour is to release the TMEM columns and
    // return so the caller's next CUDA call reports the failure instead of the rank
    // hanging the whole serve. The loud report already happened in
    // whp_mma_timeout_report. With the cp.async gate ON there may still be in-flight
    // groups here; they are abandoned deliberately — nothing on this path reads the
    // staged smem, and adding a drain would put a second wait on the failure path.
    if (hw_wait_fail) {
        __syncthreads();
        if (warp == 0) {
            hw_tc_dealloc(C_tmem, 128);
            hw_tc_dealloc(SF_tmem, 32);
        }
        return;
    }

    // ---- epilogue: read TMEM → registers → smem → global ----
    // TMEM read: each thread reads 128 f32 values (32x32b.x128)
    // C layout in TMEM: 128 lanes × 128 columns
    // Thread t reads from lane (t % 32) + warp offset, columns 0-127
    {
        float C_reg[128];
        // TileLang brackets every TMEM operation with the tcgen05 thread-sync fences
        // (tcgen05_before_thread_sync / __syncthreads / tcgen05_after_thread_sync).
        // `tcgen05.fence::before_thread_sync` must precede a thread sync that orders
        // tcgen05 async results with other threads; our kernel was missing them.
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        // TileLang's verified TMEM read (replaces custom hw_tc_ld<128> that caused ICE)
        // g_ldw=1: the ISOLATED INSTRUMENT's spelling instead — per-warp address
        // `((warp*32)<<16) | (C_tmem + 8q)` with `.32x32b.x8` in 16 steps, so every warp
        // addresses its OWN TMEM lane partition explicitly. TMEM lanes are warp-private, so the
        // two spellings can only agree if the hardware ignores the address's lane field — which
        // is exactly what DSV41_MOE_BS_LDW=1 decides.
        if (g_ldw) {
            for (int q = 0; q < 16; ++q) {
                uint32_t v[8];
                hw_tc_ld_x8(((uint32_t)(warp * 32) << 16) | (C_tmem + 8 * q), v);
                hw_tc_wait_ld();
                for (int i = 0; i < 8; ++i) C_reg[8 * q + i] = __uint_as_float(v[i]);
            }
        } else {
            tl::tcgen05_ld_32dp32bNx<128, false>(C_tmem, 0, C_reg);
        }

        // ensure all threads have read TMEM before writing C_sh (fenced pattern)
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

        // Write to C_sh (swizzled for coalesced global store)
        // Simple layout: C_sh[row * 128 + col] where row = warp * 32 + lane
        const int row = warp * 32 + lane;
        for (int col = 0; col < 128; ++col) {
            C_sh[row * 128 + col] = C_reg[col];
        }
        __syncthreads();

        // Store to global: C[seg*128 + row][n_tile*128 + col]
        for (int i = tid; i < HW_BM * HW_BN; i += 128) {
            const int r = i >> 7;   // swapAB: r = weight row (output channel); else token row
            const int c = i & 127;  // swapAB: c = token; else output column
            if (g_swapab) {
                // C[m][n] with m = the WEIGHT row (gate rows [0,64) = W1, up rows
                // [64,128) = W3) and n = the token -> transpose the row/column roles.
                // The scratch layout the scatter READS must be kept exactly as in the
                // non-swapAB branch: 128 columns per n_tile, gate in the first 64 and up
                // in the second 64 (see the scatter's
                //   bx = col/128, n = (j<64) ? bx*64+j : 320 + bx*64 + (j-64)
                // in moe_bs_shim.cu). The previous formula packed all gates into [0,320)
                // and all ups into [320,640), which the scatter mis-reads — a real
                // wiring defect found by re-deriving the mapping (see docs §61).
                C[(int64_t)(seg * HW_BM + c) * HW_NUP + n_tile * HW_BN + r] = C_sh[r * HW_BN + c];
            } else {
                C[(int64_t)(seg * HW_BM + r) * HW_NUP + n_tile * HW_BN + c] = C_sh[r * HW_BN + c];
            }
        }
    }

    // ---- TMEM dealloc ----
    __syncthreads();
    if (warp == 0) {
        hw_tc_dealloc(C_tmem, 128);
        hw_tc_dealloc(SF_tmem, 32);
    }
}
