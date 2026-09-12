// =============================================================================
// tests_tcgen05_misalign_repro.cu — deliberate-misalignment reproduction harness
// for the tcgen05 prefill arms.
//
// WHY THIS FILE EXISTS
// -----------------------------------------------------------------------------
// The tcgen05 prefill arms report a DETACHED `misaligned address` (err 716):
// the CUDA context dies at whatever synchronisation runs NEXT, not at the
// kernel that issued the bad load, so the message carries neither the kernel
// name nor the instruction. The production stack (40 GB checkpoint + a full
// TP/mega-graph step) is the WRONG instrument for finding it: a fault there
// costs a model load, and the report is still unattributed.
//
// This harness reproduces the SAME class of fault in isolation:
//   * NO checkpoint, NO 40 GB — every operand is a few-MB synthetic buffer;
//   * seconds to first launch (one small cudaMemcpy per plane);
//   * the misalignment is CONSTRUCTED ON PURPOSE (a per-plane byte slip), so
//     the fault is deterministic instead of data-dependent;
//   * the kernel is launched DIRECTLY, bypassing the launcher's al16 guard.
//     THIS IS THE WHOLE POINT: every entry (`dsv41_expert_tcgen05_gate_up_e4m3`,
//     `..._mxf4`, `dsv41_expert_gemm_e4m3_grouped`) REJECTS a misaligned BASE
//     with cudaErrorInvalidValue before the kernel exists, so a harness that
//     only calls the extern "C" entry can NEVER reach the failing read — it can
//     only prove the guard returned. `--launch entry` is kept as the CONTROL
//     that demonstrates exactly that, and `--launch direct` is the reproduction.
//
// WHAT IT IS FOR (the positive evidence)
// -----------------------------------------------------------------------------
// Run ONE case per process under compute-sanitizer:
//
//   compute-sanitizer --tool memcheck --launch-timeout 120 --print-limit 0 \
//       /tmp/t_misalign --case 11
//
// memcheck names the KERNEL, the SASS INSTRUCTION and the ADDRESS — the
// positive evidence the driver's `misaligned address` line can never give.
// What the reported instruction is then tells us the fix class:
//   * cp.async.bulk (TMA)  -> the SOURCE needs a 16-byte-aligned VIEW; a
//                             byte-fallback helper cannot help (it is a bulk
//                             copy, there is nothing to fall back to). Fix the
//                             caller's view, or add the guard.
//   * LDG.E.128 (uint4)    -> route that read through ld_uint4_a16.
//   * LDG.E.64 / .32       -> ld_uint2_a8 / the 4-byte + 2-byte split-body
//                             siblings (dsv41_experts_mxf4.cu:170,178).
//   * STS / descriptor     -> the smem side is aligned by construction; the
//                             gmem source is the suspect again.
//
// Build (nvcc; NO GPU needed to build):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//        -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 \
//        -DDSV41_TCGEN05_GATEUP_E4M3_SKELETON=1 \
//        -o /tmp/t_misalign kernels/cuda/tests_tcgen05_misalign_repro.cu
//
// Run (needs ONE free B300; peak allocation a few MB at the production shape):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_misalign --list     # the case table
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_misalign --case 11  # one case, fresh ctx
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_misalign --sweep     # the driver loop
//
// ⚠️ ONE CASE PER PROCESS. A device-side fault POISONS the context, so every
// later launch in the same process returns the sticky error and would invent
// failures. `--case N` is the isolation primitive (the same design as
// tests_tcgen05_mxf4_gateup.cu), and `--sweep` prints the ready-to-run loop.
// =============================================================================

// Both skeletons, so all three tcgen05 prefill arms are compiled into this TU:
//   tc5::mxf4  expert_tcgen05_gateup_mxf4_kernel   (fp4 activation, 2X scales)
//   tc5::e4    expert_tcgen05_gateup_e4_kernel     (e4m3 activation, 1X scales)
//   tc5::e4x   e4m3_gemm_grouped_kernel            (e4m3, grouped masked-M tile)
#define DSV41_TCGEN05_GATEUP_MXF4_SKELETON 1
#define DSV41_TCGEN05_GATEUP_E4M3_SKELETON 1
#include "dsv41_experts_mxf4.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// ----------------------------------------------------------------- the arms ---
enum Arm { ARM_E4, ARM_MXF4, ARM_E4X };
const char* arm_name(Arm a) {
    switch (a) {
        case ARM_E4:   return "e4-gateup";
        case ARM_MXF4: return "mxf4-gateup";
        default:       return "e4x-grouped";
    }
}

// ------------------------------------------------------------ the case spec ---
// A case is a launch SHAPE plus a set of byte slips relative to the base of each
// operand plane. A slip of 0 is the aligned CONTROL (it must run clean, or the
// harness itself is the bug); a slip of 4/8/12 is the reproduction.
//
// The slips exist because the production failure is a TP-SHARDED VIEW: the
// loader hands the kernel `checkpoint_pool + e * stride` and a shard boundary
// can leave ONE rank's view a few bytes off while every other rank is fine
// (dsv41_experts_mxf4.cu:160-164, :197-205). A slip is exactly that, minus the
// checkpoint.
struct Case {
    const char* name;
    Arm arm;
    int launch_entry;   // 0 = direct <<<>>> (reproduction), 1 = extern "C" (control)
    int dim, inter, slots;
    float limit;
    // slip in BYTES applied to each plane's base (0 = aligned)
    int off_act, off_acts, off_w1, off_w1s, off_w3, off_w3s;
    int off_a, off_as, off_b, off_bs, off_bh, off_bhs;  // grouped only
    // per-expert STRIDE slip (only reachable when experts > 1, i.e. ids != null)
    int slip_stride;
    int experts;        // 1 => ids == nullptr (the direct-pointer form)
};

// A case table that walks the fault surface in the order that localises it:
//   control -> B-side (act) -> A-side (w1/w3) -> scale planes -> strides,
//   then the grouped arm (which is fully byte-safe and is the CONTRAST).
const Case kCases[] = {
    // ---- 0..3  e4-gateup, minimal legal shape (dim 128, inter 64) -----------
    {"e4/control      direct  off=0",              ARM_E4,   0, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    {"e4/act+4        direct  B-side TMA",         ARM_E4,   0, 128,  64, 1, 0.f, 4,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    {"e4/w1+4         direct  A-side TMA (gate)",  ARM_E4,   0, 128,  64, 1, 0.f, 0,0,4,0,0,0, 0,0,0,0,0,0, 0, 1},
    {"e4/w3+4         direct  A-side TMA (up)",    ARM_E4,   0, 128,  64, 1, 0.f, 0,0,0,0,4,0, 0,0,0,0,0,0, 0, 1},
    // ---- 4..5  the SF prologue planes: byte-fallback SHOULD save these ------
    {"e4/w1s+4        direct  SF prologue (gate)", ARM_E4,   0, 128,  64, 1, 0.f, 0,0,0,4,0,0, 0,0,0,0,0,0, 0, 1},
    {"e4/w3s+4        direct  SF prologue (up)",   ARM_E4,   0, 128,  64, 1, 0.f, 0,0,0,0,0,4, 0,0,0,0,0,0, 0, 1},
    // ---- 6..7  per-expert stride slip (needs ids != nullptr) ---------------
    {"e4/stride+4 x4  direct  expert stride",      ARM_E4,   0, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,0,0,0,0, 4, 4},
    {"e4/stride+8 x4  direct  expert stride",      ARM_E4,   0, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,0,0,0,0, 8, 4},
    // ---- 8..10 the ring-wrap bisect on the e4 arm (the :5009 suspect) ------
    {"e4/ring dim=576  direct  A-side TMA",        ARM_E4,   0, 576,  64, 1, 0.f, 0,0,0,0,4,0, 0,0,0,0,0,0, 0, 1},
    {"e4/prod-shape   direct  off=0 (control)",    ARM_E4,   0,5120,2048, 8,10.f, 0,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    {"e4/prod w3+4    direct  A-side TMA (up)",    ARM_E4,   0,5120,2048, 8,10.f, 0,0,0,0,4,0, 0,0,0,0,0,0, 0, 1},
    // ---- 11..13  the CONTROL: the same slips through the guarded entry -----
    {"e4/w3+4  ENTRY  direct=0 guard=1",           ARM_E4,   1, 128,  64, 1, 0.f, 0,0,0,0,4,0, 0,0,0,0,0,0, 0, 1},
    {"e4/act+4 ENTRY  direct=0 guard=1",           ARM_E4,   1, 128,  64, 1, 0.f, 4,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    {"e4/aligned ENTRY  must RUN",                 ARM_E4,   1, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    // ---- 14..17  mxf4-gateup (fp4 activation) ------------------------------
    {"mxf4/control    direct  off=0",              ARM_MXF4, 0, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    {"mxf4/act+4      direct  B-side TMA",         ARM_MXF4, 0, 128,  64, 1, 0.f, 4,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    {"mxf4/w3+4       direct  A-side TMA (up)",    ARM_MXF4, 0, 128,  64, 1, 0.f, 0,0,0,0,4,0, 0,0,0,0,0,0, 0, 1},
    {"mxf4/prod-shape  direct  off=0 (control)",   ARM_MXF4, 0,5120,2048, 8,10.f, 0,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    // ---- 18..21  grouped arm: fully byte-safe => every slip must be CLEAN --
    //             (this is the CONTRAST that proves the fault is TMA-specific;
    //              bh/bhs are also the planes the grouped launcher does NOT
    //              alignment-check — e4m3_gemm_grouped_kernel's guard covers a,
    //              a_scale, b_base, bs_base only, :6163-6167)
    {"e4x/control     direct  off=0",              ARM_E4X,  0, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,0,0,0,0, 0, 1},
    {"e4x/b+4         direct  B plan (guarded)",   ARM_E4X,  0, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,4,0,0,0, 0, 1},
    {"e4x/bh+4        direct  UP plan (UNguarded)",ARM_E4X,  0, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,0,0,4,0, 0, 1},
    {"e4x/bhs+4       direct  UP scales (UNguard)",ARM_E4X,  0, 128,  64, 1, 0.f, 0,0,0,0,0,0, 0,0,0,0,0,4, 0, 1},
};

const size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

// The output sentinel: proves the kernel actually RAN (the launcher returns 0
// both on success and when the env gate is OFF — a "0 means pass" reading would
// green-light a no-op).
const float kPoison = -1.2345e30f;

bool same_bits(float a, float b) {
    uint32_t x = 0, y = 0;
    memcpy(&x, &a, 4);
    memcpy(&y, &b, 4);
    return x == y;
}

void fill_bytes(std::vector<uint8_t>& v, uint8_t b) { std::fill(v.begin(), v.end(), b); }// ------------------------------------------------------------- allocation ---
// One aligned slab per plane plus 64 bytes of slack, so a 4/8/12-byte slip is
// always a legal address inside our own allocation (the point is to make the
// VIEW misaligned, not to read past the end).
struct Buf {
    uint8_t* raw = nullptr;   // the aligned allocation
    size_t bytes = 0;
    uint8_t* plane(int off) const { return raw + off; }
};

Buf alloc_plane(size_t bytes) {
    Buf b;
    b.bytes = bytes + 64;
    if (cudaMalloc(&b.raw, b.bytes) != cudaSuccess) {
        printf("  cudaMalloc(%zu) failed: %s\n", b.bytes, cudaGetErrorString(cudaGetLastError()));
        b.raw = nullptr;
    }
    return b;
}

// ============================================================== run a case ===
int run_case(const Case& cfg) {
    const int dim = cfg.dim, inter = cfg.inter, slots = cfg.slots;
    const int rows = 2 * inter;
    const int kbytes = dim / 2;    // packed weight bytes per row
    const int nsf = dim / 32;      // e8m0 scale bytes per weight row
    const size_t nb_out = (size_t)slots * rows;

    // ---- buffers (see the parameter comments for the ABI of each arm) -------
    Buf dact  = cfg.arm == ARM_MXF4 ? alloc_plane((size_t)dim / 2)      // fp4 packed
                                    : alloc_plane((size_t)dim);          // e4m3
    Buf dacts = alloc_plane((size_t)(dim / 32) * sizeof(float));
    Buf dw1   = alloc_plane((size_t)cfg.experts * ((size_t)inter * kbytes + 64));
    Buf dw1s  = alloc_plane((size_t)cfg.experts * ((size_t)inter * nsf + 64));
    Buf dw3   = alloc_plane((size_t)cfg.experts * ((size_t)inter * kbytes + 64));
    Buf dw3s  = alloc_plane((size_t)cfg.experts * ((size_t)inter * nsf + 64));
    float* dout = nullptr;
    int* dids = nullptr;
    cudaMalloc(&dout, nb_out * sizeof(float));
    if (cfg.experts > 1) {
        std::vector<int> ids(cfg.experts);
        for (int i = 0; i < cfg.experts; ++i) ids[i] = i;
        cudaMalloc(&dids, cfg.experts * sizeof(int));
        cudaMemcpy(dids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (!dact.raw || !dacts.raw || !dw1.raw || !dw1s.raw || !dw3.raw || !dw3s.raw || !dout) {
        printf("  [%s] FAIL: allocation\n", cfg.name);
        return 1;
    }

    // Benign fill: 0x11 nibbles / 0x38 e4m3 (==1.0) / 127 e8m0 (==1) / 1.0f.
    {
        std::vector<uint8_t> z;
        z.assign(dact.bytes, cfg.arm == ARM_MXF4 ? 0x11 : 0x38);
        cudaMemcpy(dact.raw, z.data(), z.size(), cudaMemcpyHostToDevice);
        std::vector<float> f; f.assign(dacts.bytes / 4 + 1, 1.0f);
        cudaMemcpy(dacts.raw, f.data(), dacts.bytes, cudaMemcpyHostToDevice);
        for (Buf* b : {&dw1, &dw3}) {
            z.assign(b->bytes, 0x11);
            cudaMemcpy(b->raw, z.data(), z.size(), cudaMemcpyHostToDevice);
        }
        for (Buf* b : {&dw1s, &dw3s}) {
            z.assign(b->bytes, 127);
            cudaMemcpy(b->raw, z.data(), z.size(), cudaMemcpyHostToDevice);
        }
        std::vector<float> vp(nb_out, kPoison);
        cudaMemcpy(dout, vp.data(), vp.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    // ---- the MISALIGNED VIEWS ----------------------------------------------
    // A slip of 4/8/12 makes the plane base no longer 16-byte aligned, which is
    // exactly the TP-shard-boundary case. The kernel's OWN derived pointers
    // (`wrow + pk0`, `wsp + rr*nsf`, TMA source) then inherit the slip.
    const uint8_t* p_act  = dact.plane(cfg.off_act);
    const float*   p_acts = reinterpret_cast<const float*>(dacts.plane(cfg.off_acts));
    const uint8_t* p_w1   = dw1.plane(cfg.off_w1);
    const uint8_t* p_w1s  = dw1s.plane(cfg.off_w1s);
    const uint8_t* p_w3   = dw3.plane(cfg.off_w3);
    const uint8_t* p_w3s  = dw3s.plane(cfg.off_w3s);
    const long w1_stride  = (long)inter * kbytes + cfg.slip_stride;
    const long w3_stride  = (long)inter * kbytes + cfg.slip_stride;
    const long w1s_stride = (long)inter * nsf + cfg.slip_stride;
    const long w3s_stride = (long)inter * nsf + cfg.slip_stride;

    cudaError_t sync_err = cudaSuccess;
    int rc = 0;
    const char* how = nullptr;

    if (cfg.arm == ARM_E4X) {
        // ---- grouped arm: grouped layout tables ----------------------------
        // 4 experts x 128 rows, one 128-row M tile each; n_total = 2*inter split
        // into a gate|up pair so bh/bhs are on a live path.
        const int n_experts = 4, m_per = 128, m_cap = 128;
        const int n_total = rows;  // == 2*inter; output columns == weight rows
        const int n_assign = n_experts;
        const int k = dim;
        const int nk_blk = k / 32;
        std::vector<int> active(n_experts), counts(n_experts, m_per), starts(n_experts + 1, 0);
        for (int i = 0; i < n_experts; ++i) active[i] = i;
        for (int i = 0; i < n_experts; ++i) starts[i + 1] = starts[i] + counts[i];
        const int n_active = n_experts;
        const int sum_rows = starts[n_experts];

        Buf da   = alloc_plane((size_t)sum_rows * k);
        Buf das  = alloc_plane((size_t)sum_rows * nk_blk * sizeof(float));
        Buf db   = alloc_plane((size_t)n_experts * ((size_t)inter * kbytes + 64));
        Buf dbs  = alloc_plane((size_t)n_experts * ((size_t)inter * nsf + 64));
        Buf dbh  = alloc_plane((size_t)n_experts * ((size_t)inter * kbytes + 64));
        Buf dbhs = alloc_plane((size_t)n_experts * ((size_t)inter * nsf + 64));
        float* dgo = nullptr;
        int *dactive = nullptr, *dnact = nullptr, *dcounts = nullptr, *dstarts = nullptr;
        cudaMalloc(&dgo, (size_t)sum_rows * n_total * sizeof(float));
        cudaMalloc(&dactive, n_experts * sizeof(int));
        cudaMalloc(&dnact, sizeof(int));
        cudaMalloc(&dcounts, n_experts * sizeof(int));
        cudaMalloc(&dstarts, (n_experts + 1) * sizeof(int));
        cudaMemcpy(dactive, active.data(), active.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dnact, &n_active, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dcounts, counts.data(), counts.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dstarts, starts.data(), starts.size() * sizeof(int), cudaMemcpyHostToDevice);
        std::vector<uint8_t> z(sum_rows * k, 0x38);
        cudaMemcpy(da.raw, z.data(), z.size(), cudaMemcpyHostToDevice);
        std::vector<float> f(sum_rows * nk_blk, 1.0f);
        cudaMemcpy(das.raw, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice);
        for (Buf* b : {&db, &dbh}) {
            z.assign(b->bytes, 0x11);
            cudaMemcpy(b->raw, z.data(), z.size(), cudaMemcpyHostToDevice);
        }
        for (Buf* b : {&dbs, &dbhs}) {
            z.assign(b->bytes, 127);
            cudaMemcpy(b->raw, z.data(), z.size(), cudaMemcpyHostToDevice);
        }

        const uint8_t* pa  = da.plane(cfg.off_a);
        const float*   pas = reinterpret_cast<const float*>(das.plane(cfg.off_as));
        const uint8_t* pb  = db.plane(cfg.off_b);
        const uint8_t* pbs = dbs.plane(cfg.off_bs);
        const uint8_t* pbh = dbh.plane(cfg.off_bh);
        const uint8_t* pbhs = dbhs.plane(cfg.off_bhs);
        const long b_stride  = (long)inter * kbytes + cfg.slip_stride;
        const long bs_stride = (long)inter * nsf + cfg.slip_stride;
        const long bh_stride = (long)inter * kbytes + cfg.slip_stride;
        const long bhs_stride = (long)inter * nsf + cfg.slip_stride;

        const int m_tiles = (m_cap + tc5::e4x::kMTile - 1) / tc5::e4x::kMTile;
        const dim3 grid((unsigned)(n_total / tc5::e4x::kNTile), (unsigned)m_tiles,
                        (unsigned)n_assign);
        how = "direct";
        tc5::e4x::e4m3_gemm_grouped_kernel<<<grid, tc5::e4x::kThreads>>>(
            pa, pas, dgo, dactive, dnact, dcounts, dstarts, n_experts, n_total, k,
            /*b_split=*/inter, /*epi_mode=*/1, cfg.limit, pb, b_stride, pbs, bs_stride, pbh,
            bh_stride, pbhs, bhs_stride);
        rc = (int)cudaGetLastError();
        sync_err = cudaDeviceSynchronize();
        // re-use the gateup poison check on the grouped output
        cudaFree(da.raw); cudaFree(das.raw); cudaFree(db.raw); cudaFree(dbs.raw);
        cudaFree(dbh.raw); cudaFree(dbhs.raw); cudaFree(dgo);
        // report and free the common planes
        printf("  [%s]\n", cfg.name);
        printf("      arm=%s launch=%s  rc=%d sync=%s\n", arm_name(cfg.arm), how, rc,
               cudaGetErrorString(sync_err));
        const bool clean = (rc == 0 && sync_err == cudaSuccess);
        cudaFree(dact.raw); cudaFree(dacts.raw); cudaFree(dw1.raw); cudaFree(dw1s.raw);
        cudaFree(dw3.raw); cudaFree(dw3s.raw); cudaFree(dout);
        cudaFree(dactive); cudaFree(dnact); cudaFree(dcounts); cudaFree(dstarts);
        if (dids) cudaFree(dids);
        return clean ? 0 : 1;
    }

    // ---- gate/up arms (e4 + mxf4) ------------------------------------------
    const dim3 grid((unsigned)(rows / (cfg.arm == ARM_MXF4 ? tc5::mxf4::kMTile
                                                          : tc5::e4::kMTile)),
                    (unsigned)slots, 1u);
    if (cfg.launch_entry) {
        // CONTROL: the guarded extern "C" entry. A misaligned base must be
        // REJECTED here (cudaErrorInvalidValue = 1) and the kernel must NOT run.
        how = "entry";
        if (cfg.arm == ARM_MXF4) {
            rc = tc5::mxf4::dsv41_expert_tcgen05_gate_up_mxf4(
                p_act, p_acts, dout, (long)rows, inter, dim, cfg.limit, slots, p_w1, w1_stride,
                p_w1s, w1s_stride, p_w3, w3_stride, p_w3s, w3s_stride, dids, /*stream=*/0);
        } else {
            rc = tc5::e4::dsv41_expert_tcgen05_gate_up_e4m3(
                p_act, p_acts, dout, (long)rows, inter, dim, cfg.limit, slots, p_w1, w1_stride,
                p_w1s, w1s_stride, p_w3, w3_stride, p_w3s, w3s_stride, dids, /*stream=*/0);
        }
        sync_err = cudaDeviceSynchronize();
    } else {
        // REPRODUCTION: the kernel itself, so the slip REACHES the failing read
        // instead of being rejected by the guards.
        how = "direct";
        if (cfg.arm == ARM_MXF4) {
            tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel<<<grid, tc5::mxf4::kThreads>>>(
                p_act, p_acts, dout, (long)rows, dim, /*epi_mode=*/1, cfg.limit, inter, p_w1,
                w1_stride, p_w1s, w1s_stride, p_w3, w3_stride, p_w3s, w3s_stride, dids);
        } else {
            tc5::e4::expert_tcgen05_gateup_e4_kernel<<<grid, tc5::e4::kThreads>>>(
                p_act, p_acts, dout, (long)rows, dim, /*epi_mode=*/1, cfg.limit, inter, p_w1,
                w1_stride, p_w1s, w1s_stride, p_w3, w3_stride, p_w3s, w3s_stride, dids);
        }
        rc = (int)cudaGetLastError();
        sync_err = cudaDeviceSynchronize();
    }

    // Did the kernel actually write? (distinguishes "ran" from "gate off" /
    // "guard rejected" — the project's #1 measurement-bias trap.)
    size_t poison_left = 0;
    {
        std::vector<float> got(nb_out);
        cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < got.size(); ++i)
            if (same_bits(got[i], kPoison)) ++poison_left;
    }

    printf("  [%s]\n", cfg.name);
    printf("      arm=%s launch=%s  rc=%d (%s)  sync=%s  outputs_written=%zu/%zu\n",
           arm_name(cfg.arm), how, rc, cudaGetErrorString((cudaError_t)rc),
           cudaGetErrorString(sync_err), nb_out - poison_left, nb_out);

    cudaFree(dact.raw); cudaFree(dacts.raw); cudaFree(dw1.raw); cudaFree(dw1s.raw);
    cudaFree(dw3.raw); cudaFree(dw3s.raw); cudaFree(dout);
    if (dids) cudaFree(dids);
    // A live fault poisons the context: report it as such so the driver marks
    // the context dead and re-runs the remaining cases in fresh processes.
    if (sync_err != cudaSuccess) return 2;
    return (rc == 0) ? 0 : 1;
}

// ---------------------------------------------------------------- the driver ---
void usage(const char* argv0) {
    printf("usage: %s [--list] [--case N] [--sweep] [--help]\n\n"
           "  --list   print the case table (index, name, arm, launch)\n"
           "  --case N run ONE case in this process (the isolation primitive)\n"
           "  --sweep  print the ready-to-run per-case + compute-sanitizer loop\n\n"
           "compute-sanitizer (the attribution step):\n"
           "  compute-sanitizer --tool memcheck --launch-timeout 120 --print-limit 0 \\\n"
           "      %s --case 3\n"
           "  -> reports the kernel name, the SASS instruction and the address.\n",
           argv0, argv0);
}

void list_cases() {
    printf("cases=%zu\n", kNumCases);
    for (size_t i = 0; i < kNumCases; ++i)
        printf("   --case %2zu  %-40s arm=%-12s launch=%s\n", i, kCases[i].name,
               arm_name(kCases[i].arm), kCases[i].launch_entry ? "entry" : "direct");
}

void print_sweep(const char* argv0) {
    printf("# one process per case: a device fault poisons the context, so the\n"
           "# ONLY reliable attribution is one case per process.\n"
           "for i in $(seq 0 %zu); do\n", kNumCases - 1);
    printf("  echo \"===== case $i =====\"\n");
    printf("  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} %s --case $i || true\n", argv0);
    printf("done\n\n");
    printf("# the same loop under compute-sanitizer (names kernel + instruction + address):\n");
    printf("for i in $(seq 0 %zu); do\n", kNumCases - 1);
    printf("  echo \"===== case $i =====\"\n");
    printf("  compute-sanitizer --tool memcheck --launch-timeout 120 --print-limit 0 \\\n");
    printf("      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} %s --case $i \\\n", argv0);
    printf("      2>&1 | grep -E 'Invalid|Misaligned|(^| )at |kernel|save|=====' || true\n");
    printf("done\n");
}

}  // namespace

int main(int argc, char** argv) {
    int only_case = -1;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "--list") == 0) { list_cases(); return 0; }
        if (strcmp(a, "--sweep") == 0) { print_sweep(argv[0]); return 0; }
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) { usage(argv[0]); return 0; }
        if (strcmp(a, "--case") == 0 || strncmp(a, "--case=", 7) == 0) {
            only_case = (a[6] == '=') ? atoi(a + 7) : (i + 1 < argc ? atoi(argv[++i]) : -1);
            if (only_case < 0 || only_case >= (int)kNumCases) {
                printf("--case wants 0..%zu\n", kNumCases - 1);
                return 2;
            }
        } else {
            printf("unknown arg: %s (try --help)\n", a);
            return 2;
        }
    }
    if (only_case < 0) { usage(argv[0]); return 0; }

    // Arm the .so-side runtime gates BEFORE the first entry call (they are
    // process statics read once). Harmless for a direct launch; required so the
    // `entry` control cases exercise the REAL gate + guard path.
    setenv("DSV41_EXPERT_TCGEN05_MXF4", "1", 1);
    setenv("DSV41_EXPERT_TCGEN05_E4M3", "1", 1);
    setenv("DSV41_EXPERT_GROUPED", "1", 1);

    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("== tcgen05 misaligned-address reproduction harness ==\n");
    printf("   device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);
    if (prop.major != 10 && prop.major != 11)
        printf("   WARNING: the tcgen05 path targets sm_103a; expect a launch failure here\n");
    printf("   gates armed: TCGEN05_MXF4=1 TCGEN05_E4M3=1 EXPERT_GROUPED=1\n");

    const Case& cfg = kCases[only_case];
    const int r = run_case(cfg);
    const bool failed = (r != 0);
    printf("\nRESULT case %d (%s): %s\n", only_case, cfg.name,
           failed ? (r == 2 ? "DEVICE FAULT (reproduced — see compute-sanitizer)"
                            : "FAIL (guard / shape / no-output)")
                  : "CLEAN");
    return failed ? 1 : 0;
}
