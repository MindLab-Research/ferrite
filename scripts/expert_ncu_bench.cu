// =============================================================================
// expert_ncu.cu — ISOLATED ncu repro for the DSV4.1-Flash expert family
//                 (fused batched gate/up fp4 GEMV  +  fused down+reduce GEMV)
// =============================================================================
//
// WHY THIS FILE EXISTS (post-swapab-landscape, STATUS.md 2026-09-12 13:00)
// -------------------------------------------------------------------------
// After swapAB the step is ~4.0 ms (250 tok/s) and the ranking is:
//     #1 expert gateup+down        2.00 ms   50%   <- this repro measures it
//     #2 hc_dots_late (side stream) 1.43 ms  36%
//     #3 hc_mixes_tail + gaps + AR  1.98 ms
//
// The two floor verdicts in flight were both produced by ncu on a KERNEL THAT
// NO LONGER EXISTS:
//     gateup = "LUT smem random gather latency"
//     down   = "L1TEX pipe 71%"
// Since those runs the kernel changed three times over:
//     1. gate_up + swiglu fusion (epilogue writes `inter`, not `2*inter`)
//     2. DSV41_EXPERT_ILV — gate/up interleaved in ONE region, 8-byte granule,
//        row pitch doubled, one LDG.128 per k-group instead of two LDG.64
//     3. DSV41_GATEUP_KSPLIT — now pinned to 2 in production (fused body only:
//        cross-half merge via CTA smem + __fadd_rn, blockDim 8*2 warps)
//     4. DSV41_DOWN_VEC4 -> g_down_fp4_mode == 3 (4 values/lane, k < 512 path)
// ⇒ Any further expert optimization MUST start from fresh counters. This file
// is the instrument. It is deliberately shaped like the two accepted precedents
// (kernels/cuda/ncu_miniprof.cu for the profiler window,
//  kernels/cuda/ncu_moe_bench.cu for the link-the-production-.so + real-shape
//  pattern) and like /tmp/hc_repro.cu (the repro whose numbers actually matched
//  serve: 58.4 us/call flat across rows).
//
// WHAT IT DOES
// ------------
//   * links the PRODUCTION .so (kernels/cuda/libferrite_kernels.so) — the real
//     launchers `dsv41_expert_gate_up_fp4_batched` /
//     `dsv41_expert_down_reduce_fp4_batched`, not a copy of the kernel body;
//   * allocates the PRODUCTION-SIZE expert pool (~960 MiB over 384 experts) so
//     the weight streams are L2-cold, not 15 MB of L2-resident toy data. This is
//     THE point that invalidated earlier microbenches on this kernel ("a
//     cache-warm run is a different machine" — dsv41_experts_mxf4.cu:2508);
//   * fills it with PSEUDO-RANDOM bytes: the LUT index is DATA-dependent
//     (`s_lut2[weight_byte]`), so a memset-filled pool would broadcast on one
//     bank and hide the entire scatter cost;
//   * reproduces the ILV per-expert block layout byte-for-byte (offsets mirror
//     load.rs::load_expert_pool), including the degenerate-stride foot-gun;
//   * warms up, then opens a cudaProfilerStart/Stop window over exactly
//     `PROF_ITERS` iterations for `ncu --profile-from-start off`;
//   * prints the real geometry (grid/block/smem/stride) + wall-clock us/call so
//     the ncu numbers can be sanity-checked against the nsys table.
//
// BUILD (on the B300 node; the .so must already exist in the same tree)
//   K=$PWD/kernels/cuda
//   nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a \
//        -o /tmp/expert_ncu kernels/cuda/../scripts/expert_ncu_bench.cu \
//        -L$K -l:libferrite_kernels.so
//   (nvcc does NOT accept -Wl,-rpath here — use LD_LIBRARY_PATH, see
//    scripts/dsv41_microbench.sh.)
//
// RUN (env MUST match production; the .so caches every one of these at first call)
//   export LD_LIBRARY_PATH=$PWD/kernels/cuda
//   DSV41_GATEUP_KSPLIT=2 DSV41_GATEUP_ROWS=8 DSV41_EXPERT_ILV=1 \
//   DSV41_EXPERT_FP4_MODE=2 DSV41_DOWN_VEC4=1 DSV41_GATEUP_FUSE=1 \
//   DSV41_DOWN_FUSE=1 DSV41_PDL=0 \
//     ./expert_ncu pair 20          # pair = gateup then down, production order
//   modes: gateup | down | pair (default) | both (pair + timings)
//   2nd arg = warmup iterations (default 20)
//
// NCU
// ---
//   # (a) one kernel at a time, full SOL set, 3 launches:
//   sudo /usr/local/cuda-13.2/bin/ncu --profile-from-start off \
//        --launch-count 3 --launch-skip 0 --set full \
//        -k "regex:expert_gemv_fp4_batched" -f -o /tmp/ncu_expert_gateup \
//        ./expert_ncu gateup 20
//   sudo /usr/local/cuda-13.2/bin/ncu --profile-from-start off \
//        --launch-count 3 --set full \
//        -k "regex:expert_gemv_fp4_down_reduce" -f -o /tmp/ncu_expert_down \
//        ./expert_ncu down 20
//   # (b) the stall decomposition that ANSWERS THE FLOOR QUESTION (see below):
//   sudo /usr/local/cuda-13.2/bin/ncu --profile-from-start off --launch-count 3 \
//        --metrics \
//   smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
//   smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
//   smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
//   smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
//   smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
//   smsp__warp_issue_stalled_not_selected_per_warp_active.pct,\
//   smsp__warp_issue_stalled_wait_per_warp_active.pct,\
//   smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
//   smsp__warp_issue_stalled_drain_per_warp_active.pct,\
//   smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,\
//   smsp__issue_active_per_warp_active.pct,\
//   sm__throughput.avg.pct_of_peak_sustained_elapsed,\
//   gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
//   dram__throughput.avg.pct_of_peak_sustained_elapsed,\
//   l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
//   lts__throughput.avg.pct_of_peak_sustained_elapsed,\
//   l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
//   l1tex__data_pipe_lsu_wavefronts_mem_global.sum,\
//   sm__warps_active.avg.pct_of_peak_sustained_active,\
//   sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
//   sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,\
//   launch__registers_per_thread,launch__shared_mem_per_block_dynamic,\
//   launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
//   launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem \
//        -k "regex:expert_gemv_fp4" ./expert_ncu pair 20
//   # (c) per-SASS-instruction attribution (which PC stalls):
//   ... --set full --page source --print-source sass ...
//       (the .so is built WITHOUT -lineinfo by build.sh, so correlate the
//        top-stalling PC against `cuobjdump -sass` / ncu's SASS page, not source)
//
// JUDGMENT CRITERIA  (decide, before touching the kernel again)
// ------------------------------------------------------------
//   Def.:  S_lut = short_scoreboard + mio_throttle   (LDS: LUT + s_act)
//          S_w   = long_scoreboard  + lg_throttle    (LDG: weight stream)
//          S_fma = math_pipe_throttle                (FFMA issue)
//          S_ns  = not_selected                      (warps ARE available)
//   * S_lut < 30% of stall cycles  -> the "LUT smem random gather" floor verdict
//                                     is WRONG for the current kernel; re-target.
//   * S_lut > 60%                  -> LUT gather floor CONFIRMED (keep the
//                                     verdict; only an MMA/hw-decode design helps).
//   * S_w dominant                 -> it is the GLOBAL weight stream / LSU queue,
//                                     i.e. cp.async/MLP territory — note this
//                                     contradicts the 2026-09-12 08:00 note
//                                     ("80% issue stall is NOT weight latency").
//   * issue_active > 60% AND S_fma/S_lsu high
//                                  -> the kernel is ISSUE-bound (instruction
//                                     stream per value). Fix = fewer instructions
//                                     per value (tcgen05 kind::f8f6f4), NOT ILP.
//   * dram__throughput < 10%       -> NOT DRAM-bound (confirms the 30-40% / 5%
//                                     observations); ignore any bandwidth story.
//   * l1tex__throughput > 60%      -> L1TEX pipe saturated (the down verdict).
//                                     Then split it: shared-wavefronts vs
//                                     global-wavefronts says whether the LUT/act
//                                     side or the weight side owns the pipe.
//   * waves_per_multiprocessor < 1.2 AND occupancy_limit_registers is the cap
//                                  -> grid/registers, not latency: K-split / MLP
//                                     is the lever (this is the K-split decision
//                                     the 2026-09-11 doc left open).
//
// CAVEATS THAT MUST BE QUOTED WITH ANY RESULT
// -------------------------------------------
//   * ncu's default `--cache-control all` FLUSHES L2 before each replay pass, so
//     under ncu every launch starts L2-cold. That matches production HERE only
//     because the pool is 960 MiB >> 126 MiB L2 and production reads each layer's
//     pool once. If you ever run against a small pool, DRAM% is a fiction.
//   * DSV41_W2_PREWARM is DEFAULT OFF in the .cu (dsv41_experts_mxf4.cu:2661 —
//     "default OFF ... +0.04ms regression"), so production's down GEMV really
//     does stream w2 from DRAM. The chain_dev.rs:4215 comment claiming
//     "default ON" is stale; do not let it change the experiment. `--prewarm`
//     below exists only to price the prewarm alternative.
//   * Isolation -> production has failed FIVE times on this kernel. These
//     numbers are for ELIMINATION and for choosing a direction; a positive
//     result still has to be confirmed by a single-round serve A/B.
//   * ncu serializes and re-runs the kernel; OVERLAP (PDL, side stream, grid
//     contention with the warmer) is invisible here by construction.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// ---- production launchers (kernels/cuda/libferrite_kernels.so) -------------
extern "C" {
// kernels/cuda/dsv41_experts_mxf4.cu:2333
int dsv41_expert_gate_up_fp4_batched(const uint8_t* a, const float* a_scale, float* out,
                                     long out_slot_stride, int rows, int dim, int inter, float limit,
                                     int slots, const uint8_t* w1_base, long w1_stride,
                                     const uint8_t* w1s_base, long w1s_stride,
                                     const uint8_t* w3_base, long w3_stride,
                                     const uint8_t* w3s_base, long w3s_stride, const int* ids,
                                     int ilv, cudaStream_t stream);
// kernels/cuda/dsv41_experts_mxf4.cu:2486
int dsv41_expert_down_reduce_fp4_batched(const float* act_base, long act_stride, float* out,
                                         int rows, int dim, int inter, const float* row_weight,
                                         long rw_stride, int slots, const uint8_t* w2_base,
                                         long w2_stride, const uint8_t* w2s_base, long w2s_stride,
                                         const int* ids, cudaStream_t stream);
// kernels/cuda/dsv41_experts_mxf4.cu:2660 (default OFF — see caveats)
int dsv41_w2_l2_prewarm(const uint8_t* w2_base, long w2_stride, const uint8_t* w2s_base,
                        long w2s_stride, const int* ids, int slots, long sel_bytes, long sc_bytes,
                        cudaStream_t stream);
// kernels/cuda/ferrite_kernels.cu:33 — proves WHICH .so got loaded (AGENTS.md
// "加载错防线"). Both must be non-null; a mismatch with .build_id means stop.
const char* ferrite_kernel_build_id(void);
unsigned ferrite_kernel_abi_version(void);
}

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t e_ = (x);                                                        \
        if (e_ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA err @%d %s: %s\n", __LINE__, #x,                   \
                    cudaGetErrorString(e_));                                         \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

// ============================================================================
// PRODUCTION SHAPE  (verified against the code, not from memory)
// ============================================================================
//   cfg: dim = 5120, moe_inter_dim = 2304, n_routed_experts = 384,
//        n_activated_experts = 6 (config.rs:496-506; configs/dsv41_flash.json)
//   world = 8 (--tp 8, single node)
//   inter_local = padded_inter(2304/8) = padded_inter(288) = 320
//        (weights.rs:442 padded_inter = ceil(n/64)*64, K_ATOM = 64)
//   chain_dev.rs:3896-3902, 4189-4214 (gate/up), 4288-4304 (down fused)
//
//   gate/up  (fused, ILV, ksplit=2):  rows=1, dim=5120 (=K), inter=320 (=n_total)
//        grid  = (ceil(320/8), 6) = (40, 6) = 240 CTA       [launcher :2399]
//        block = rows*ksplit*32 = 512 threads (16 warps)
//        smem  = dim*4 + 256*8 + nwarps*8 [+ nwarps*512 cp.async] = 30848 B
//                (launcher :2390-2392; the 30848 figure is the one the doc quotes)
//   down     (fused down+reduce, vec4): rows=1, dim=5120 (=n), inter=320 (=K)
//        grid  = (ceil(5120/8), 1) = 640 CTA, block = 8*32 = 256 threads
//        smem  = slots*inter*4 + 256*8 = 9728 B  => STAGED path (fits opt-in cap)
//
//   NOTE the two directions are TRANSPOSED (this is why one shared dispatcher
//   does not fit, and why they must be profiled separately):
//        gateup: n = 320 (one row per warp), k = 5120
//        down  : n = 5120 (one row per warp), k = 320
// ============================================================================
static const int DIM = 5120;        // cfg.dim
static const int WORLD = 8;         // --tp 8
static const int MOE_INTER = 2304;  // cfg.moe_inter_dim
static const int IL = 320;          // padded_inter(2304/8)
static const int SLOTS = 6;         // n_activated_experts (topk)
static const int NEXP = 384;        // n_routed_experts — FULL production pool
static const int ROWS = 1;
static const float LIMIT = 10.0f;   // cfg.swiglu_limit

// Per-expert plane sizes and the interleaved block layout.
// Mirrors load.rs::load_expert_pool (ILV branch: o[1]=2*w1b; w3.weight aliases
// offset 0; scales stay in their own blocks).
static const long W1B = (long)IL * (DIM / 2);     // 819200  hp = inter*dim/2
static const long W1SB = (long)IL * (DIM / 32);   // 51200   scale row = dim/32
static const long POFF_W1 = 0;                    // w1||w3 interleaved (aliased)
static const long POFF_W1S = 2 * W1B;             // 1638400
static const long POFF_W3S = POFF_W1S + W1SB;     // 1689600
static const long W2B = (long)DIM * (IL / 2);     // 819200  hp = dim*inter/2
static const long POFF_W2 = POFF_W3S + W1SB;      // 1740800
static const long W2SB = (long)DIM * (IL / 32);   // 51200
static const long POFF_W2S = POFF_W2 + W2B;       // 2560000
static const long BLOCK = POFF_W2S + W2SB;        // 2611200 B per expert
static const long POOL_BYTES = BLOCK * NEXP;      // ~960 MiB (>> 126 MiB L2)

// --- device-side pseudorandom fill (data-dependent LUT indices!) -----------
__global__ void fill_rand_kernel(uint8_t* p, size_t n, unsigned seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned x = (unsigned)i * 2654435761u + seed * 40503u;
        x ^= x >> 13;
        x *= 0x5bd1e995u;
        x ^= x >> 15;
        p[i] = (uint8_t)(x >> 11);
    }
}
static void fill_rand(uint8_t* d, size_t n, unsigned seed, cudaStream_t s) {
    const int threads = 256;
    long blocks = (long)((n + threads - 1) / threads);
    if (blocks > 1 << 20) blocks = 1 << 20;
    fill_rand_kernel<<<(unsigned)blocks, threads, 0, s>>>(d, n, seed);
    CK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "pair";
    const int warm = (argc > 2) ? atoi(argv[2]) : 20;
    const bool do_gateup = strcmp(mode, "gateup") == 0 || strcmp(mode, "pair") == 0 ||
                           strcmp(mode, "both") == 0;
    const bool do_down = strcmp(mode, "down") == 0 || strcmp(mode, "pair") == 0 ||
                         strcmp(mode, "both") == 0;
    const bool do_time = strcmp(mode, "both") == 0;

    CK(cudaSetDevice(0));
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    printf("== expert_ncu ==  build_id=%s  abi=%u\n",
           ferrite_kernel_build_id ? ferrite_kernel_build_id() : "(null)",
           ferrite_kernel_abi_version ? ferrite_kernel_abi_version() : 0u);
    printf("   shape: dim=%d inter_local=%d slots=%d experts=%d world=%d rows=%d\n", DIM, IL,
           SLOTS, NEXP, WORLD, ROWS);
    printf("   pool : %.1f MiB (%ld B/expert, %ld experts)\n",
           (double)POOL_BYTES / (1024.0 * 1024.0), BLOCK, (long)NEXP);
    printf("   mode : %s  warmup=%d\n", mode, warm);

    // ---------------- weights: ONE production-size pool ----------------
    uint8_t* pool = nullptr;
    CK(cudaMalloc(&pool, POOL_BYTES));
    fill_rand(pool, (size_t)POOL_BYTES, 12345u, stream);
    CK(cudaStreamSynchronize(stream));

    // per-expert bases are all derived from ONE base + a uniform stride, exactly
    // as chain_dev.rs does with pointer differences between expert 0 and 1.
    const uint8_t* w1_base = pool + POFF_W1;
    const uint8_t* w1s_base = pool + POFF_W1S;
    const uint8_t* w3_base = pool + POFF_W1;  // ALIASED under ILV (never read)
    const uint8_t* w3s_base = pool + POFF_W3S;
    const uint8_t* w2_base = pool + POFF_W2;
    const uint8_t* w2s_base = pool + POFF_W2S;

    // ---------------- activations / outputs ----------------
    uint8_t* d_a = nullptr;      // packed fp4 activation for gate/up: k/2 bytes
    float* d_asc = nullptr;      // e8m0-as-float scales: k/32 entries
    float* d_act = nullptr;      // [slots][act_stride] f32 (gate/up fused output)
    float* d_out = nullptr;      // down output: dim floats
    float* d_rw = nullptr;       // [slots] routing weights, rw_stride=1
    int* d_ids = nullptr;        // [slots] expert ids (router output)
    CK(cudaMalloc(&d_a, (size_t)(DIM / 2)));
    CK(cudaMalloc(&d_asc, (size_t)(DIM / 32) * sizeof(float)));
    CK(cudaMalloc(&d_act, (size_t)SLOTS * (size_t)IL * sizeof(float)));
    CK(cudaMalloc(&d_out, (size_t)DIM * sizeof(float)));
    CK(cudaMalloc(&d_rw, (size_t)SLOTS * sizeof(float)));
    CK(cudaMalloc(&d_ids, (size_t)SLOTS * sizeof(int)));
    fill_rand(d_a, (size_t)(DIM / 2), 777u, stream);
    fill_rand(reinterpret_cast<uint8_t*>(d_act), (size_t)SLOTS * (size_t)IL * sizeof(float), 778u,
              stream);
    {
        // activation scales inside a valid e8m0 exponent range (values are
        // irrelevant to the counters, but keep the arithmetic from going inf).
        std::vector<float> h((size_t)(DIM / 32));
        for (size_t i = 0; i < h.size(); i++) h[i] = ldexpf(1.0f, (int)(i % 5) - 2);
        CK(cudaMemcpy(d_asc, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> rw(SLOTS, 1.0f);
        for (int i = 0; i < SLOTS; i++) rw[i] = 0.5f + 0.1f * (float)i;
        CK(cudaMemcpy(d_rw, rw.data(), SLOTS * sizeof(float), cudaMemcpyHostToDevice));
    }

    // ---------------- rotating expert ids ----------------
    // Production reads a DIFFERENT layer's 960 MiB pool on each launch, so the
    // 6 slots' experts are cold. Holding ids fixed here would make the same
    // 6 x 2.5 MiB L2-resident after the first iteration and the DRAM column of
    // the report would be meaningless. Rotating the ids reproduces the cold
    // read while keeping every access inside a real per-expert stride.
    std::vector<int> h_ids(SLOTS);
    auto set_ids = [&](long iter) {
        for (int s = 0; s < SLOTS; s++)
            h_ids[s] = (int)(((long)s * 61 + iter * SLOTS) % NEXP);
        CK(cudaMemcpyAsync(d_ids, h_ids.data(), SLOTS * sizeof(int), cudaMemcpyHostToDevice,
                           stream));
    };

    // ---------------- launch closures ----------------
    const long act_slot = IL;  // gate/up fused writes `inter` floats per slot
    auto gateup = [&]() {
        return dsv41_expert_gate_up_fp4_batched(d_a, d_asc, d_act, act_slot, ROWS, DIM, IL, LIMIT,
                                                SLOTS, w1_base, BLOCK, w1s_base, BLOCK, w3_base,
                                                BLOCK, w3s_base, BLOCK, d_ids, /*ilv=*/1, stream);
    };
    auto down = [&]() {
        return dsv41_expert_down_reduce_fp4_batched(d_act, act_slot, d_out, ROWS, DIM, IL, d_rw,
                                                    /*rw_stride=*/1, SLOTS, w2_base, BLOCK,
                                                    w2s_base, BLOCK, d_ids, stream);
    };
    auto prewarm = [&]() {
        return dsv41_w2_l2_prewarm(w2_base, BLOCK, w2s_base, BLOCK, d_ids, SLOTS,
                                   (long)DIM * (IL / 2), (long)DIM * (IL / 32), stream);
    };

    // ---------------- warmup (LUT/L2/plan-cache settle) ----------------
    for (int i = 0; i < warm; i++) {
        set_ids(i);
        if (do_gateup) CK(gateup());
        if (do_down) CK(down());
    }
    CK(cudaStreamSynchronize(stream));
    CK(cudaGetLastError());

    // ---------------- PROFILE WINDOW ----------------
    // ncu --profile-from-start off profiles only inside this window. Keep the
    // window tight: exactly PROF_ITERS iterations, no fill kernels inside.
    int prof_iters = 6;
    {
        const char* pe = getenv("EXPERT_NCU_ITERS");
        if (pe != nullptr) prof_iters = atoi(pe);
        if (prof_iters < 1) prof_iters = 1;
        CK(cudaProfilerStart());
        for (int i = 0; i < prof_iters; i++) {
            set_ids(warm + i);
            if (do_gateup) CK(gateup());
            if (do_down) {
                if (getenv("DSV41_W2_PREWARM")) CK(prewarm());  // price the alternative
                CK(down());
            }
        }
        CK(cudaStreamSynchronize(stream));
        CK(cudaProfilerStop());
    }
    printf("   profiler window closed (%d iterations in window)\n", prof_iters);

    // ---------------- wall-clock (only meaningful with =both) ----------------
    // us/call must reproduce the nsys medians (gateup ~24 us, down ~17 us) at
    // production shape; if they do not, this repro is measuring a different
    // configuration and the ncu numbers are void.
    if (do_time) {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        auto bench = [&](const char* name, auto&& fn) {
            for (int i = 0; i < 5; i++) {
                set_ids(1000 + i);
                CK(fn());
            }
            CK(cudaStreamSynchronize(stream));
            CK(cudaEventRecord(e0, stream));
            const int N = 200;
            for (int i = 0; i < N; i++) {
                set_ids(2000 + i);
                CK(fn());
            }
            CK(cudaEventRecord(e1, stream));
            CK(cudaEventSynchronize(e1));
            float ms = 0.f;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            printf("   %-22s %8.2f us/call (cold ids, %d iters)\n", name, ms * 1000.f / N, N);
        };
        if (do_gateup) bench("gateup", gateup);
        if (do_down) bench("down", down);
    }

    CK(cudaFree(d_a));
    CK(cudaFree(d_asc));
    CK(cudaFree(d_act));
    CK(cudaFree(d_out));
    CK(cudaFree(d_rw));
    CK(cudaFree(d_ids));
    CK(cudaFree(pool));
    CK(cudaStreamDestroy(stream));
    printf("done\n");
    return 0;
}
