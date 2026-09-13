// dsv41_proj_mma_skel.cu — PROJ-MMA (DSV41_PROJ_MMA): the TENSOR-CORE form of the
// MULTI-ROW fp8 projection GEMV (`gemm_fp8_mrows_mma_kernel<M>`).
//
// STATUS: SKELETON / design artifact. Nothing here is in the build
// (`kernels/cuda/build.sh` does not list this file) and nothing here is
// reachable at runtime. It exists to make the design in
// docs/agent/tensorcore-proj-design.md COMPILE-CHECKED on sm_103a: the kernel
// signature, the MMA mapping, the K-split ABI, the launcher shape tests, the ks
// rule and the env gate. The body is written out (no stubs) because every
// building block it uses is already proven in-tree.
//
// Compile-only (no GPU; the same host as the MPAR compile round):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
//        -c kernels/cuda/dsv41_proj_mma_skel.cu -o /tmp/proj_mma.o
//   add -Xptxas -v to see "0 spills" and the `mma.sync.aligned.m16n8k32` count.
//
// =============================================================================
// WHY (the lesion this is aimed at) — docs/agent/verify-amortization-lesion-audit.md
// =============================================================================
// `gemm_fp8_mrows<M>` measured 52.1 us for the m=5 launch = 5x the SINGLE-row
// gemv: the M fold amortises nothing. NCU on the same kernel: DRAM 0.67-0.79%,
// Compute 12.8-13%, L1 14.1-14.6%, occupancy 13.4% -- NOTHING is saturated, so
// the kernel is latency-bound by construction, and the M rows only reach a
// warp's `float acc[M]` (no parallel axis), so M rows cost M times the LDS/ALU
// issue on a launch that has 3-5 warps per SM to hide anything with.
// MPAR (M as a WARP axis) then lost twice (rpb=1 +0.52 ms, coverage-aware auto
// +1.28 ms cumulative) because its prologue/LUT/activation replication is
// proportional to the BLOCK COUNT -- i.e. the SIMT program cannot buy back the
// M fold without paying more than it saves.
//
// The structural way out is to stop issuing the product per (element, k) at all:
// one `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` retires 16x8x32 =
// 4096 MACs in ONE instruction, against ~34 warp-instructions for 6x32 = 192
// SIMT MACs in `gemm_fp8_mrows_kernel<M>` -- a ~48x reduction in issued
// instructions per MAC. On a launch that is latency-bound at 13% occupancy,
// issued-instruction collapse and a memory ring that decouples DRAM latency from
// the accumulate are the only levers left.
//
// =============================================================================
// MAPPING (swapAB: the WEIGHT is the MMA's M, the ACTIVATION is its N)
// =============================================================================
//   A = w  [n, k] e4m3 row-major      -> A row-major [M=n rows, K]
//   B = a  [m, k] e4m3 row-major      -> B col-major [K, N=m]  (row r IS column r)
//   D =     [16 channels, 8 act rows] -> out[r * out_stride + channel]
//   w_scale [(n/32), nb_k] ue8m0      -> one scale word per (16-row tile, 32-k)
//   a_scale [m, nb_k]      f32        -> per (activation row, 32-k)
//
// WHY swapAB AND NOT "activation on M, pad 6 -> 16". The task's baseline framing
// is the pad-16 form (6/16 = 37.5% of the tile used). Putting the WEIGHTS on M
// instead pads the ACTIVATION to the N=8 tile, i.e. 6/8 = 75% used, and it is
// strictly better on every axis:
//   * utilisation 6/8 = 75% vs 6/16 = 37.5%;
//   * grid: n/16 tiles vs n/8 tiles -- 2x more warps for the same weights;
//   * the weight scale is CONSTANT over a 16-row tile (all 16 rows sit in one
//     32-row scale block), so the per-k-block scale is one mul, not a per-row
//     lookup;
//   * it is the mapping the tree ALREADY ships twice -- `gemm_fp8_swapab_kernel`
//     (M=1 decode on the tensor core, :606) and the tcgen05 swapAB gate/up
//     skeleton (dsv41_experts_mxf4.cu:3666).
// This kernel is the M<=8 generalisation of `gemm_fp8_swapab_kernel`: the ONLY
// two changes are (i) the B fragment is read from column `gid` of an [m, kc]
// staged activation block instead of from the single token row, and (ii) the
// epilogue writes all M columns instead of only column 0.
//
// =============================================================================
// NUMERICS — NOT bit-identical to the SIMT program; bit-identical to ITSELF
// =============================================================================
// See docs/agent/tensorcore-proj-design.md §2. Short form:
//   SIMT : acc += (lut[a[j]] * as) * (lut[w[j]] * sb)   per element, then a
//          5-level shfl_xor tree. ~2 rounding events per element.
//   MMA  : D = sum of 32 EXACT fp8 products inside the tensor core, then
//          acc += D * (as * sb) -- ~1-2 rounding events per k-block plus the
//          internal (hardware-defined) summation order.
// Mathematically identical, different last bit. Bit-parity with the SIMT program
// is therefore IMPOSSIBLE, and no staging trick can buy it back.
// What IS available, and what this kernel is built for: the D column r is a
// function of B column r and A alone -- no operand of another activation row
// enters it -- so
//     row r of an M-row launch  ==  row r of the M=1 launch of the same kernel
// EXACTLY, for every M and every ks, PROVIDED the K-split is a function of
// (n, k) only and never of M (the ks>1 reducer sums partials in ascending kp;
// changing ks changes the association). That is the contract the verify's
// "row r of an m-row launch == the m=1 decode of row r" needs -- just against
// this program instead of the SIMT one. It is the same relaxation the tree
// already ships for `DSV41_SWAPAB` (`gemm_fp8_swapab_kernel` header: "BIT-
// EXACTNESS: NOT bit-identical ... Parity is judged by text/fingerprint"), and
// it is why the gate below is default-OFF and must be judged by the dual gate
// (step_ms AND mean-k), not by a byte compare.
//
// =============================================================================
// GEOMETRY (one warp per (16-row weight tile, K partition))
// =============================================================================
//   grid  = (n / 16) * ks          one warp per (tile, partition)
//   block = 32 threads
//   ks    = K partitions, chosen from (n, k, SM count) -- NEVER from m
// The n/16 tile count is SMALL at the small-n shapes (wkv n=512 -> 32 tiles, the
// shared expert n=288 -> 18), and that is the shape where `gemm_fp8_mrows` is
// worst. That is exactly why the K split exists (the pre-existing swapAB
// launcher already carves `kSwapabKSplit` partitions and reduces with a
// per-tile last-arrival ticket); `proj_mma_ks_for` below extends that rule to
// "one block per SM" instead of "n >= 1664 only".
//
// SMEM (per warp; no LUT -- the tensor core decodes the fp8 bytes):
//   [M][kc + 16] activation fp8 | [M][nb] activation scale | [nb] weight scale
//   | ring [NStage][16][KStep + 16] weight fp8
// The +16 row pads are BANK-CONFLICT knobs, not alignment ones (the same note
// `kSwapabRow` carries): with an unpadded stride every one of the 8/16 gid rows
// of a fragment lands on the same bank. +16 makes the row stride a multiple of
// 16B (so `cp.async.cg` 16B and the ring's 16B units still hold) while keeping
// stride/4 % 32 != 0.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

__device__ __forceinline__ float ue8m0_to_f(uint8_t b) {
    // float8_e8m0fnu: a pure power of two, 2^(b-127); 0xFF is NaN.
    return __int_as_float((int)((b == 0xFFu ? 0x7FC00000u : ((uint32_t)b) << 23)));
}

__device__ __forceinline__ void mma_cp16(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     (unsigned)__cvta_generic_to_shared(smem_dst)),
                 "l"(gmem_src));
}
__device__ __forceinline__ void mma_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void mma_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---- geometry -------------------------------------------------------------
constexpr int kMmaKStep = 128;   // ring stage width in k (a multiple of 32)
constexpr int kMmaNStage = 8;    // ring depth (>= 2 for wait_group(NSTAGE-2))
constexpr int kMmaTileM = 16;    // MMA M: 16 weight rows (output channels)
constexpr int kMmaTileN = 8;     // MMA N: 8 activation rows (M <= 8)
constexpr int kMmaWRow = kMmaKStep + 16;  // weight ring row stride (bank knob)
static_assert(kMmaWRow % 16 == 0, "the ring row destination must be 16B aligned");
static_assert((kMmaWRow >> 2) % 32 != 0,
              "row stride must NOT be a multiple of 128B: every A-fragment LDS.32 would be "
              "8-way bank conflicted (see the kSwapabRow note)");

// =============================================================================
// gemm_fp8_mrows_mma_kernel<M>
// =============================================================================
// ABI: (a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr)
//   a         [m, k]       fp8 e4m3, row r at +r*k         (m == M)
//   a_scale   [m, k/32]    f32
//   w         [n, k]       fp8 e4m3
//   w_scale   [n/32, k/32] ue8m0 (block 32x32)
//   out       f32, row r's channel `ch` at +r*out_stride + ch
//   ks        K partitions (a function of (n, k) -- see the header)
//   partial   [ks][M][n] f32, only touched when ks > 1
//   ctr       [n/16] u32, one arrival ticket per tile, only when ks > 1
// Returns are the launcher's business; the kernel assumes it was dispatched.
template <int M>
__global__ void __launch_bounds__(32)
gemm_fp8_mrows_mma_kernel(const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
                          const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_scale,
                          const float* __restrict__ bias, float* __restrict__ out, int n, int k,
                          int out_stride, int ks, float* __restrict__ partial,
                          unsigned* __restrict__ ctr) {
    static_assert(M >= 1 && M <= kMmaTileN, "the MMA's N tile is 8 activation rows");
#if __CUDA_ARCH__ >= 900
    // PDL (the family's convention, dsv41_pdl_or_plain): the producer of this
    // activation block may still be in its tail, so the first read of `a`/`a_scale`
    // is gated on the producer's completion.
    cudaGridDependencySynchronize();
#endif
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;  // A rows gid / gid+8 -- and the B column
    const int tg = lane & 3;    // A cols tg*4 (+16); C cols 2*tg (+1)

    const int ntile = n >> 4;
    const int gw = blockIdx.x;  // one warp per (tile, K partition)
    if (gw >= ntile * ks) return;
    const int tile = gw / ks;
    const int kp = gw % ks;
    const int m0 = tile * kMmaTileM;
    const int kc = k / ks;   // k per partition; a multiple of 32
    const int k0p = kp * kc;
    const int nb = kc >> 5;
    const int nb_k = k >> 5;

    const size_t arow = (size_t)kc + 16;  // activation row stride (bank knob, 16B grid)
    extern __shared__ uint8_t mma_smem[];
    uint8_t* s_a = mma_smem;                                  // [M][arow] fp8
    size_t off = (size_t)M * arow;
    off = (off + 15) & ~(size_t)15;
    float* s_as = (float*)(mma_smem + off);                   // [M][nb] f32
    off += (size_t)M * nb * sizeof(float);
    off = (off + 15) & ~(size_t)15;
    uint8_t* s_ws = mma_smem + off;                           // [nb] u8
    off += (size_t)nb;
    off = (off + 15) & ~(size_t)15;
    uint8_t* sw = mma_smem + off;                             // ring [NStage][16][WRow]

    // ---- activation + scale staging (plain 16B copies, NOT cp.async) ---------
    // Deliberately not cp.async: the ring below owns the commit-group cadence
    // (wait_group(NSTAGE-2) counts THIS thread's groups), and folding the
    // activation into it would make the count depend on `kc`. The same choice
    // `gemm_fp8_swapab_kernel` records for its own activation staging.
    {
        const int n16 = kc >> 4;
        for (int r = 0; r < M; ++r) {
            const uint4* src4 = (const uint4*)(a + (size_t)r * k + k0p);
            uint4* dst4 = (uint4*)(s_a + (size_t)r * arow);
            for (int i = lane; i < n16; i += 32) dst4[i] = src4[i];
        }
        const int kb0 = k0p >> 5;
        for (int i = lane; i < M * nb; i += 32) {
            const int r = i / nb, c = i - r * nb;
            s_as[i] = a_scale[(size_t)r * nb_k + kb0 + c];
        }
        const int ws0 = (m0 >> 5) * nb_k;  // 16 rows always sit in ONE 32-row block
        for (int i = lane; i < nb; i += 32) s_ws[i] = w_scale[(size_t)ws0 + kb0 + i];
    }
    __syncwarp();

    // ---- weight ring (cp.async 16B, per-warp slice) --------------------------
    const int nk = (kc + kMmaKStep - 1) / kMmaKStep;
    auto stage = [&](int st) {
        if (st < nk) {
            const int k0 = k0p + st * kMmaKStep;
            const int rb = min(kMmaKStep, kc - st * kMmaKStep);  // % 16 == 0
            const int nchunk = rb >> 4;
            uint8_t* dst = sw + (size_t)(st % kMmaNStage) * kMmaTileM * kMmaWRow;
            for (int c = lane; c < kMmaTileM * nchunk; c += 32) {
                const int r = c / nchunk;
                const int coff = (c - r * nchunk) << 4;
                mma_cp16(dst + (size_t)r * kMmaWRow + coff, w + (size_t)(m0 + r) * k + k0 + coff);
            }
        }
        // An out-of-range stage STILL commits (an empty group): wait_group relies
        // on "NSTAGE-1 groups outstanding before the wait", and near the end of K
        // the real issues run out -- without the empty groups the count drops to
        // NSTAGE-2 and the LAST TWO stages get read before their cp.asyncs land
        // (the hazard `gemm_fp8_swapab_kernel` documents at its stage lambda).
        mma_cp_commit();
    };
    for (int st = 0; st < kMmaNStage - 1; st++) stage(st);

    // ---- K walk ---------------------------------------------------------------
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    // This thread's C columns: 2*tg and 2*tg+1. Columns >= M are DEAD (their C
    // entries are dropped in the epilogue); their scale read is clamped so it
    // stays inside [M][nb].
    const int c0 = (2 * tg < M) ? (2 * tg) : (M - 1);
    const int c1 = (2 * tg + 1 < M) ? (2 * tg + 1) : (M - 1);
    for (int st = 0; st < nk; st++) {
        // One group per iteration keeps NSTAGE-1 outstanding, so
        // wait_group(NSTAGE-2) always retires the oldest one (== stage st).
        stage(st + kMmaNStage - 1);
        mma_cp_wait<kMmaNStage - 2>();
        __syncwarp();  // every lane's own group is retired -> the tile is complete

        const uint8_t* s = sw + (size_t)(st % kMmaNStage) * kMmaTileM * kMmaWRow;
        const int lk0 = st * kMmaKStep;
        const int nkb = min(kMmaKStep, kc - lk0) >> 5;
        for (int kb = 0; kb < nkb; kb++) {
            const int lk = kb * 32;
            // A fragment: rows gid / gid+8, cols tg*4 (+16) -- the WEIGHT rows.
            uint32_t af[4];
            af[0] = *(const uint32_t*)&s[(size_t)gid * kMmaWRow + lk + tg * 4];
            af[1] = *(const uint32_t*)&s[(size_t)(gid + 8) * kMmaWRow + lk + tg * 4];
            af[2] = *(const uint32_t*)&s[(size_t)gid * kMmaWRow + lk + tg * 4 + 16];
            af[3] = *(const uint32_t*)&s[(size_t)(gid + 8) * kMmaWRow + lk + tg * 4 + 16];
            // B fragment: K x N col-major, column n = gid = the ACTIVATION row,
            // rows k = tg*4 (+16). Column r of B is activation row r -- which is
            // the whole parity argument (see the file header). Columns gid >= M
            // are dead: clamp to a legal load, the epilogue drops them.
            const int bcol = (gid < M) ? gid : 0;
            uint32_t bf[2];
            bf[0] = *(const uint32_t*)&s_a[(size_t)bcol * arow + lk + tg * 4];
            bf[1] = *(const uint32_t*)&s_a[(size_t)bcol * arow + lk + tg * 4 + 16];
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
            // Per-k-block scale: the weight scale is constant over the 16-row tile
            // (all 16 rows sit in one 32-row scale block); the activation scale is
            // per (row, k-block), applied on the C COLUMN. d[] is the raw fp8 sum.
            const int kbg = (lk0 >> 5) + kb;
            const float wsc = ue8m0_to_f(s_ws[kbg]);
            const float sc0 = s_as[(size_t)c0 * nb + kb] * wsc;
            const float sc1 = s_as[(size_t)c1 * nb + kb] * wsc;
            acc[0] += d[0] * sc0;
            acc[1] += d[1] * sc1;
            acc[2] += d[2] * sc0;
            acc[3] += d[3] * sc1;
        }
    }

    // ---- epilogue -------------------------------------------------------------
    // C fragment: d[0]=C[gid][2tg], d[1]=C[gid][2tg+1], d[2]=C[gid+8][2tg],
    // d[3]=C[gid+8][2tg+1]. C ROW = the output channel (m0+gid, m0+gid+8);
    // C COLUMN = the activation row (2tg, 2tg+1).
    const int ar0 = 2 * tg, ar1 = 2 * tg + 1;
    const int ch0 = m0 + gid, ch1 = m0 + gid + 8;
    const size_t slot = ((size_t)kp * M + 0) * (size_t)n;  // + ar*n + ch, added per store
    if (ks == 1) {
        if (ar0 < M && ch0 < n)
            out[(size_t)ar0 * out_stride + ch0] = acc[0] + ((bias != nullptr) ? bias[ch0] : 0.f);
        if (ar1 < M && ch0 < n)
            out[(size_t)ar1 * out_stride + ch0] = acc[1] + ((bias != nullptr) ? bias[ch0] : 0.f);
        if (ar0 < M && ch1 < n)
            out[(size_t)ar0 * out_stride + ch1] = acc[2] + ((bias != nullptr) ? bias[ch1] : 0.f);
        if (ar1 < M && ch1 < n)
            out[(size_t)ar1 * out_stride + ch1] = acc[3] + ((bias != nullptr) ? bias[ch1] : 0.f);
    } else {
        // Publish into this partition's OWN slot (no atomic RMW, no pre-zeroed
        // `out`), then the last arrival of the tile reduces the ks slots in a
        // FIXED ascending-kp order -- the discipline (and the bit-determinism
        // argument) `gemm_fp8_swapab_kernel` establishes. Note the slot is per
        // ACTIVATION ROW here (`[ks][M][n]`), because one warp produces all M rows.
        if (ar0 < M && ch0 < n) partial[slot + (size_t)ar0 * n + ch0] = acc[0];
        if (ar1 < M && ch0 < n) partial[slot + (size_t)ar1 * n + ch0] = acc[1];
        if (ar0 < M && ch1 < n) partial[slot + (size_t)ar0 * n + ch1] = acc[2];
        if (ar1 < M && ch1 < n) partial[slot + (size_t)ar1 * n + ch1] = acc[3];
        __syncwarp();
        __threadfence();
        unsigned ticket = 0u;
        if (lane == 0) ticket = atomicAdd(&ctr[tile], 1u);
        ticket = __shfl_sync(0xffffffffu, ticket, 0);
        if (ticket == (unsigned)(ks - 1)) {
            __threadfence();
            const auto reduce = [&](int ar, int ch, float own) -> float {
                if (ar >= M || ch >= n) return 0.f;
                float v = 0.f;
                for (int q = 0; q < ks; q++)
                    v += partial[((size_t)q * M + ar) * (size_t)n + ch] + (q == kp ? own - own : 0.f);
                return v + ((bias != nullptr) ? bias[ch] : 0.f);
            };
            const float o0 = acc[0], o1 = acc[1], o2 = acc[2], o3 = acc[3];
            if (ar0 < M && ch0 < n) out[(size_t)ar0 * out_stride + ch0] = reduce(ar0, ch0, o0);
            if (ar1 < M && ch0 < n) out[(size_t)ar1 * out_stride + ch0] = reduce(ar1, ch0, o1);
            if (ar0 < M && ch1 < n) out[(size_t)ar0 * out_stride + ch1] = reduce(ar0, ch1, o2);
            if (ar1 < M && ch1 < n) out[(size_t)ar1 * out_stride + ch1] = reduce(ar1, ch1, o3);
            if (lane == 0) atomicExch(&ctr[tile], 0u);  // self-reset: graph replay clean
        }
    }
}

// =============================================================================
// launcher / gate / switch
// =============================================================================
// GATE: DSV41_PROJ_MMA (default OFF; strict "== \"1\"", the convention
// DSV41_TAP_PARITY / DSV41_COMP_PARITY use). OFF returns 2 => the caller keeps
// the SIMT `dsv41_gemm_fp8_mrows` path, byte for byte.
//
// SWITCH ORDER in `dsv41_gemm_fp8_mrows` (the implementation plan; see the doc):
//   1. PROJ_MMA arm (this)         -- numerics-changing, so it must be exclusive
//   2. MPAR arm (DSV41_MROWS_MPAR) -- bit-exact, `fold_r == m` only
//   3. legacy M-in-register arm    -- bit-exact, the default
// PROJ_MMA and MPAR are never both armed: PMMA changes the summation program, so
// an A/B that mixes them measures nothing. The launcher refuses (returns 2) when
// both gates resolve non-zero, with a one-shot receipt.
static int g_proj_mma = [] {
    const char* e = getenv("DSV41_PROJ_MMA");
    return (e != nullptr && e[0] == '1') ? 1 : 0;
}();

// ks (K partitions) is a function of (n, k) ONLY -- never of m. That is not a
// style rule, it is the parity contract: the ks>1 reducer sums partials in
// ascending kp, so a different ks is a different summation. A gate that let ks
// depend on m would silently break "row r of the m-row launch == the m=1 decode
// of row r", which is the ONE thing this program can still promise.
//
// RULE: fill the SMEM-LIMITED RESIDENCY, not "one block per SM". The measured
// reason the shipped swapAB gemv gets 1.3-1.5 TB/s is that ~11 one-warp blocks
// stay resident per SM (its own header states MLP comes from that, NOT from
// per-warp ring depth), and the two other fixed points are "kc stays a whole
// number of 32-wide scale blocks" and this kernel's smem per warp
// (~NStage*16*WRow = 18 KiB -> ~12 resident blocks/SM on a 228 KiB part).
//   n = 5120 (wo_b, 320 tiles): ks = 2 -> 640 warps   (kc = 512)
//   n = 4096 (wq_b, 256 tiles): ks = 4 -> 1024 warps  (kc = 320)
//   n = 1280 (wq_a,  80 tiles): ks = 8 -> 640 warps   (kc = 640)
//   n =  512 (wkv,   32 tiles): ks = 32 -> 1024 warps (kc = 160)
//   n =  288 (sh,    18 tiles): ks = 32 -> 576 warps  (kc = 160)
// The small-n half is exactly where `gemm_fp8_mrows` is worst and exactly where
// the shipped swapAB launcher DECLINES (`if (n < 1664) return 2`) -- that
// threshold is a fixed-overhead observation at ks = kSwapabKSplit (8), not a law;
// the same file records ks=10 beating ks=8 (5.72 vs 7.31 us/call) at n=1664.
inline int proj_mma_ks_for(int n, int k, int sms, int override_ks) {
    int ks = 1;
    if (override_ks >= 1) {
        ks = override_ks;
    } else {
        const int tiles = (n >> 4) > 0 ? (n >> 4) : 1;
        const int kblk = k >> 5;                                  // 32-wide k blocks
        int want = (sms > 0) ? (sms * 8 + tiles - 1) / tiles : 1;  // warps/full residency
        if (want < 1) want = 1;
        ks = 1;
        while (ks * 2 <= want && (kblk % (ks * 2) == 0)) ks *= 2;
        if ((kblk % ks) != 0) ks = 1;
    }
    if (ks < 1) ks = 1;
    while (ks > 1 && ((k >> 5) % ks) != 0) ks >>= 1;
    return ks;
}

int proj_mma_sm_count() {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cached = (sms > 0) ? sms : 1;
    }
    return cached;
}

}  // namespace

// Returns 0 (launched) or 2 (DECLINED: the caller keeps the SIMT arm). Never 1:
// 1 is cudaErrorInvalidValue, i.e. indistinguishable from a real launch failure
// (the r42-45 collision the swapAB entry documents).
//
// `pmma_n` is the caller's promise that `partial` is big enough. The scratch for
// this arm is [ks][M][n] f32, M = VERIFY_ROWS -- ks*M times the swapAB scratch,
// a Rust-side sizing change (chain_dev.rs `swapab_part`). Until that lands, a
// caller can pass ks = 1 (which needs NO scratch).
extern "C" int dsv41_gemm_fp8_mrows_mma(const uint8_t* a, const float* a_scale, const uint8_t* w,
                                        const uint8_t* w_scale, const float* bias, float* out,
                                        int m, int n, int k, int out_stride, float* partial,
                                        unsigned* ctr, int pmma_n, cudaStream_t s) {
    if (g_proj_mma == 0) return 2;
    if (m < 1 || m > kMmaTileN) return 2;
    if (n <= 0 || k <= 0 || (n & 15) || (k & 31)) return 2;
    if (out_stride < n) return 2;
    if (n > pmma_n) return 2;  // the caller's scratch bound (mirrors swapab_n)
    if (a == nullptr || a_scale == nullptr || w == nullptr || w_scale == nullptr || out == nullptr)
        return 2;
    // 16B alignment: the activation staging is uint4 / the ring is cp.async 16B,
    // so both bases must be on the 16B grid (a misaligned base is err 716 at
    // best and silently misplaced bytes at worst -- reported, not swallowed).
    {
        static int reported = 0;
        if ((((uintptr_t)a & 0xF) != 0) || (((uintptr_t)w & 0xF) != 0)) {
            if (reported++ < 4)
                fprintf(stderr,
                        "[proj-mma] activation/weight base is not 16B -- the cp.async16 "
                        "contract cannot be met\n");
            return 2;
        }
    }
    static int ovr = -1;
    if (ovr < 0) {
        const char* e = getenv("DSV41_PROJ_MMA_KS");
        ovr = (e != nullptr) ? atoi(e) : 0;
        if (ovr < 0) ovr = 0;
    }
    const int ks = proj_mma_ks_for(n, k, proj_mma_sm_count(), ovr);
    if (ks > 1 && (partial == nullptr || ctr == nullptr)) return 2;  // no scratch -> SIMT

    const int tiles = n >> 4;
    const int grid = tiles * ks;
    const int kc = k / ks, nb = kc >> 5;
    const size_t arow = (size_t)kc + 16;
    size_t off = (size_t)m * arow;
    off = (off + 15) & ~(size_t)15;
    off += (size_t)m * nb * sizeof(float);
    off = (off + 15) & ~(size_t)15;
    off += (size_t)nb;
    off = (off + 15) & ~(size_t)15;
    const size_t smem = off + (size_t)kMmaNStage * kMmaTileM * kMmaWRow;
    if (smem > 48 * 1024) {
        // Per-kernel ceiling, sticky: a smaller value would silently cap a later
        // launch (round-43 revert), so EVERY M specialisation gets its own.
        cudaError_t e = cudaSuccess;
#define FERRITE_SET_PMMA_SMEM(mm)                                                        \
    do {                                                                                 \
        cudaError_t r =                                                                  \
            cudaFuncSetAttribute(gemm_fp8_mrows_mma_kernel<mm>,                          \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        if (r != cudaSuccess && e == cudaSuccess) e = r;                                  \
    } while (0)
        FERRITE_SET_PMMA_SMEM(1);
        FERRITE_SET_PMMA_SMEM(2);
        FERRITE_SET_PMMA_SMEM(3);
        FERRITE_SET_PMMA_SMEM(4);
        FERRITE_SET_PMMA_SMEM(5);
        FERRITE_SET_PMMA_SMEM(6);
        FERRITE_SET_PMMA_SMEM(7);
        FERRITE_SET_PMMA_SMEM(8);
#undef FERRITE_SET_PMMA_SMEM
        if (e != cudaSuccess) { (void)cudaGetLastError(); return (int)e; }
    }
    // ACTIVITY RECEIPT (one line per process, on the first ARMED launch): the arm
    // has its own kernel NAME, but a DECLINE has none -- and this tree has been
    // bitten by exactly that phantom-gate shape. Name the resolved geometry.
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-mma] ARMED m=%d n=%d k=%d ks=%d -> grid=%d warps, block=32, smem=%zu\n",
                    m, n, k, ks, grid, smem);
    }
    switch (m) {
        case 1: gemm_fp8_mrows_mma_kernel<1><<<grid, 32, smem, s>>>(a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr); break;
        case 2: gemm_fp8_mrows_mma_kernel<2><<<grid, 32, smem, s>>>(a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr); break;
        case 3: gemm_fp8_mrows_mma_kernel<3><<<grid, 32, smem, s>>>(a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr); break;
        case 4: gemm_fp8_mrows_mma_kernel<4><<<grid, 32, smem, s>>>(a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr); break;
        case 5: gemm_fp8_mrows_mma_kernel<5><<<grid, 32, smem, s>>>(a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr); break;
        case 6: gemm_fp8_mrows_mma_kernel<6><<<grid, 32, smem, s>>>(a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr); break;
        case 7: gemm_fp8_mrows_mma_kernel<7><<<grid, 32, smem, s>>>(a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr); break;
        case 8: gemm_fp8_mrows_mma_kernel<8><<<grid, 32, smem, s>>>(a, a_scale, w, w_scale, bias, out, n, k, out_stride, ks, partial, ctr); break;
        default: return 2;
    }
    return (int)cudaGetLastError();
}
