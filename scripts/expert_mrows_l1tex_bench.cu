// =============================================================================
// expert_mrows_l1tex_bench.cu — rows-sweep L1TEX repro for the DSV4.1-Flash
//                               routed-expert gate/up kernel
//                 (expert_gemv_fp4_batched_kernel<ILV,1> via the PRODUCTION .so)
// =============================================================================
//
// WHY THIS FILE EXISTS (moe-batch-fwd, 第五路 A, 2026-09-13)
// ---------------------------------------------------------
// The verdict is: the SAME kernel through the SAME launcher measures
//   eager  (chain_dev.rs:18584, rows = 1) : ~700 GB/s  ~ 0.88 L1TEX-op/value
//   verify (chain_dev.rs:15218, rows = m) : ~377 GB/s  ~ 1.46 L1TEX-op/value
// i.e. a 1.66x per-value L1TEX regression purely as a function of `rows`.
//
// The call-site audit (docs/agent/moe-batch-fwd-rows-l1tex-verdict.md, 交付①)
// shows the two call sites pass BYTE-IDENTICAL launcher arguments except `rows`
// itself, and `rows` only becomes `gridDim.z` (dsv41_experts_mxf4.cu:1385
// `const int arow = (int)blockIdx.z;`). So the regression is NOT a config
// difference — it must be produced by the launch GEOMETRY. Two candidates remain:
//
//   (C2) FOOTPRINT / L2: rows = m multiplies the number of CONCURRENT expert
//        regions in flight (every (slot, arow) pair has its own `ids` entry, so
//        a resident window of K CTAs spans K/ctas_x distinct expert regions
//        instead of 6). Predicts: L1TEX wavefronts/value rise because of L1TEX
//        miss handling, and DRAM efficiency drops.
//   (C3) OCCUPANCY / smem carveout under grid.z. Predicts: the rows=1 arm is the
//        SLOWER one (it runs 240 CTAs over 148 SMs = 1.6 CTA/SM, i.e. LOWER
//        occupancy). This arm is the falsification test for C3.
//
// THE DECISIVE EXPERIMENT: run rows = 1 / 3 / 6 with the SAME kernel and TWO
// ids patterns, so `rows` and `footprint` stop being confounded:
//
//   DSV41_BENCH_IDS=distinct : every (slot, arow) routes to its own expert
//                              => footprint scales with rows  (production-like)
//   DSV41_BENCH_IDS=same     : all rows of a slot route to the SAME expert
//                              => footprint is the rows=1 footprint, but the
//                                 grid is still (ctas_x, slots, rows)
//
//   * rows=6/same  == rows=1  -> the cost is PURELY footprint (C2). Fix = locality
//                                (see the design doc §4), and the launch geometry
//                                itself is innocent.
//   * rows=6/same  >> rows=1  -> the cost survives with a constant footprint, so
//                                it is the grid.z geometry / occupancy (C3) and
//                                the smem-carveout / CTA-shape lever applies.
//
// BUILD (on the B300 node; kernels/cuda/libferrite_kernels.so must exist)
//   K=$PWD/kernels/cuda
//   /usr/local/cuda-13.2/bin/nvcc -O3 -std=c++17 \
//        -gencode arch=compute_103a,code=sm_103a \
//        -o /tmp/expert_mrows scripts/expert_mrows_l1tex_bench.cu \
//        -L$K -l:libferrite_kernels.so
//
// RUN (env MUST match production — the .so caches every one of these at 1st call)
//   export LD_LIBRARY_PATH=$PWD/kernels/cuda
//   DSV41_GATEUP_KSPLIT=2 DSV41_GATEUP_ROWS=8 DSV41_GATEUP_CPASYNC=1 \
//   DSV41_GATEUP_PIPELINE=1 DSV41_EXPERT_ILV=1 DSV41_EXPERT_FP4_MODE=2 \
//   DSV41_GATEUP_FUSE=1 DSV41_MOE_BATCH=1 DSV41_PDL=0 \
//     /tmp/expert_mrows            # sweep: rows 1/3/6 x {same,distinct}
//   /tmp/expert_mrows 6 distinct     # single point
//
// NCU (the three columns the ticket asks for, plus the dividers)
// -------------------------------------------------------------
//   sudo /usr/local/cuda-13.2/bin/ncu --profile-from-start off --launch-count 3 \
//     --metrics \
//   smsp__issue_active.avg.pct_of_peak_sustained_elapsed,\
//   smsp__issue_active_per_warp_active.pct,\
//   l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
//   l1tex__data_pipe_lsu_wavefronts_mem_global.sum,\
//   l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
//   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
//   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
//   l1tex__t_sector_hit_rate.pct,\
//   lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
//   gpu__time_duration.sum,\
//   smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
//   smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
//   smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
//   smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
//   smsp__warps_active.avg.pct_of_peak_sustained_active,\
//   launch__registers_per_thread,launch__shared_mem_per_block_dynamic,\
//   launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,\
//   launch__grid_size,launch__block_size,launch__waves_per_multiprocessor \
//     -k "regex:expert_gemv_fp4_batched" /tmp/expert_mrows 6 distinct
//
//   ONE ROW OF THE TABLE = one (rows, ids-mode) point. `条/value` =
//       (l1tex__data_pipe_lsu_wavefronts_mem_global.sum +
//        l1tex__data_pipe_lsu_wavefronts_mem_shared.sum)
//       / (rows * SLOTS * IL * DIM)            // fp4 weight values consumed
//   and GB/s = gpu__time_duration.sum^-1 * rows*SLOTS*IL*DIM/2 * 2 planes.
//   The wall-clock column this program prints must land on the nsys medians
//   (gateup ~24 us/call at rows=1) or the repro is measuring a different config.
//
// CAVEATS
// -------
//   * ncu's default `--cache-control all` flushes L2 per pass, so each profiled
//     launch is L2-cold. THAT IS WHAT WE WANT here (production streams a 960 MiB
//     pool per step), but it means a small-pool variant would be a fiction.
//   * ncu SERIALIZES; production overlaps this kernel with the shared-expert
//     half on a side stream. Use the wall-clock column for the overlap story.
//   * ncu's replay may re-run the kernel several times; the ids rotation below
//     keeps every pass cold, matching production's per-layer cold pool.
//   * the pool layout here mirrors load.rs::load_expert_pool (ILV: w1||w3
//     aliased at offset 0, scales in their own blocks). A layout drift would
//     silently measure a different access pattern, so the constants are printed.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// ---- production launcher (kernels/cuda/libferrite_kernels.so) --------------
// NOTE the TRAILING `int act_e4m3` — this is the current 21-argument ABI
// (dsv41_experts_mxf4.cu:3523). scripts/expert_ncu_bench.cu still declares the
// 19-argument pre-e4m3 form and would pass `stream` as `act_e4m3`; do not copy
// that declaration.
extern "C" {
int dsv41_expert_gate_up_fp4_batched(const uint8_t* a, const float* a_scale, float* out,
                                    long out_slot_stride, int rows, int dim, int inter, float limit,
                                    int slots, const uint8_t* w1_base, long w1_stride,
                                    const uint8_t* w1s_base, long w1s_stride,
                                    const uint8_t* w3_base, long w3_stride,
                                    const uint8_t* w3s_base, long w3s_stride, const int* ids,
                                    int ilv, int act_e4m3, cudaStream_t stream);
// THE CANDIDATE-1 PROBE. `launch_mxf4_indirect` (dsv41_experts_mxf4.cu:3412)
// dispatches `rows == 1 && getenv("DSV41_NO_GEMV_FP4") == nullptr` to
// `expert_gemv_fp4_kernel` — a DIFFERENT kernel from the batched one (48 regs vs
// the batched ILV body's 64, its own vec-mode table at .cu:900). chain_dev.rs's
// `moe()` reaches this entry only when DSV41_MOE_BATCH=0, while `moe_rows()`
// NEVER consults moe_batch() and therefore always takes the batched launcher
// (chain_dev.rs:15216-15218). So `DSV41_MOE_BATCH=0` makes the two arms run
// different kernels — arm 3 below measures exactly that, and ncu's kernel-name
// column names it.
//   ⚠ its `out` is DENSE [rows][2*inter] (pitch n_total=2*inter), NOT the batched
//     [rows][slots][inter] — it needs its own buffer (d_out_dense).
//   ⚠ it has NO row dimension at the slot level: it computes ONE slot per call
//     and `slot` selects ids[slot]. rows>1 only tiles the M dim.
//   ⚠ its B side is the PLAIN (non-interleaved) w1/w3 layout, so on an ILV pool
//     the BYTES it reads are not the numbers the model means. That is fine for a
//     COUNTER probe (the LUT indices are data-dependent either way) and is the
//     point: we are asking which INSTRUCTION STREAM costs 1.46 条/value.
int dsv41_expert_gate_up_fp4_indirect(const uint8_t* a, const float* a_scale, float* out,
                                      int rows, int dim, int inter, float limit,
                                      const uint8_t* w1_base, long w1_stride,
                                      const uint8_t* w1s_base, long w1s_stride,
                                      const uint8_t* w3_base, long w3_stride,
                                      const uint8_t* w3s_base, long w3s_stride, const int* ids,
                                      int slot, int act_e4m3, cudaStream_t stream);
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
// PRODUCTION SHAPE (configs/dsv41_flash.json + weights.rs::padded_inter)
//   dim = 5120, moe_inter_dim = 2304, n_routed_experts = 384, topk = 6, world = 8
//   inter_local = padded_inter(2304/8) = padded_inter(288) = 320
//   gate/up pair body: n_total = inter = 320, k = dim = 5120
//   grid = (ceil(320/DSV41_GATEUP_ROWS), 6, rows), block = ROWS*KSPLIT*32
// ============================================================================
static const int DIM = 5120;
static const int IL = 320;
static const int SLOTS = 6;
static const int NEXP = 384;
static const float LIMIT = 10.0f;
static const int MAX_ROWS = 8;

static const long W1B = (long)IL * (DIM / 2);     // 819200  hp = inter*dim/2
static const long W1SB = (long)IL * (DIM / 32);   // 51200
static const long POFF_W1 = 0;                    // w1||w3 interleaved (aliased)
static const long POFF_W1S = 2 * W1B;
static const long POFF_W3S = POFF_W1S + W1SB;
static const long BLOCK = POFF_W3S + W1SB;        // 1740800 B per expert (gate/up half)
static const long POOL_BYTES = BLOCK * NEXP;      // ~638 MiB (>> L2 -> cold streams)

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
    long blocks = (long)((n + 255) / 256);
    if (blocks > 1 << 20) blocks = 1 << 20;
    fill_rand_kernel<<<(unsigned)blocks, 256, 0, s>>>(d, n, seed);
    CK(cudaGetLastError());
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "sweep";   // sweep | <rows> | all
    const char* idmode = (argc > 2) ? argv[2] : "distinct";  // distinct | same | prod
    const int warm = 20;
    const int iters = 100;

    CK(cudaSetDevice(0));
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    printf("== expert_mrows ==  build_id=%s  abi=%u\n",
           ferrite_kernel_build_id ? ferrite_kernel_build_id() : "(null)",
           ferrite_kernel_abi_version ? ferrite_kernel_abi_version() : 0u);
    printf("   shape: dim=%d inter=%d slots=%d experts=%d rows<=%d\n", DIM, IL, SLOTS, NEXP,
           MAX_ROWS);
    printf("   pool : %.1f MiB (%ld B/expert, %d experts)\n",
           (double)POOL_BYTES / (1024.0 * 1024.0), BLOCK, NEXP);

    uint8_t* pool = nullptr;
    CK(cudaMalloc(&pool, POOL_BYTES));
    fill_rand(pool, (size_t)POOL_BYTES, 12345u, stream);
    CK(cudaStreamSynchronize(stream));

    const uint8_t* w1_base = pool + POFF_W1;
    const uint8_t* w1s_base = pool + POFF_W1S;
    const uint8_t* w3_base = pool + POFF_W1;   // ALIASED under ILV (never read)
    const uint8_t* w3s_base = pool + POFF_W3S;

    // ---- activations / outputs, sized for the WIDEST row count ---------------
    uint8_t* d_a = nullptr;    // [rows][dim/2] packed fp4
    float* d_asc = nullptr;    // [rows][dim/32] f32 scales
    float* d_act = nullptr;    // [rows][slots][IL] f32 (fused epilogue)
    float* d_out_dense = nullptr;  // [rows][2*IL] f32 -- the INDIRECT entry's pitch
    int* d_ids = nullptr;      // [rows][slots] i32
    CK(cudaMalloc(&d_a, (size_t)MAX_ROWS * (DIM / 2)));
    CK(cudaMalloc(&d_asc, (size_t)MAX_ROWS * (DIM / 32) * sizeof(float)));
    CK(cudaMalloc(&d_act, (size_t)MAX_ROWS * SLOTS * IL * sizeof(float)));
    CK(cudaMalloc(&d_out_dense, (size_t)MAX_ROWS * 2 * IL * sizeof(float)));
    CK(cudaMalloc(&d_ids, (size_t)MAX_ROWS * SLOTS * sizeof(int)));
    fill_rand(d_a, (size_t)MAX_ROWS * (DIM / 2), 777u, stream);
    fill_rand(reinterpret_cast<uint8_t*>(d_act), (size_t)MAX_ROWS * SLOTS * IL * sizeof(float),
              778u, stream);
    {
        std::vector<float> h((size_t)MAX_ROWS * (DIM / 32));
        for (size_t i = 0; i < h.size(); i++) h[i] = ldexpf(1.0f, (int)(i % 5) - 2);
        CK(cudaMemcpy(d_asc, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // ---- ids patterns -------------------------------------------------------
    // `distinct`: expert = (arow*SLOTS + slot)*K + rot  -> footprint scales with rows
    // `same`    : expert = slot*K + rot (arow independent) -> footprint == rows=1
    // `prod`    : every (slot,arow) independent but drawn from a shared pool, so
    //             duplicates are possible exactly as production's router produces
    //             them (36 assignments over 384 experts).
    std::vector<int> h_ids((size_t)MAX_ROWS * SLOTS);
    auto set_ids = [&](int rows, long iter) {
        for (int r = 0; r < rows; r++)
            for (int s = 0; s < SLOTS; s++) {
                int e;
                if (!strcmp(idmode, "same"))
                    e = (int)(((long)s * 61 + iter * SLOTS) % NEXP);
                else if (!strcmp(idmode, "prod"))
                    e = (int)(((long)(r * (SLOTS - 1) + s) * 97 + iter * SLOTS) % NEXP);
                else
                    e = (int)(((long)(r * SLOTS + s) * 61 + iter * SLOTS * MAX_ROWS) % NEXP);
                h_ids[(size_t)r * SLOTS + s] = e;
            }
        CK(cudaMemcpyAsync(d_ids, h_ids.data(), (size_t)rows * SLOTS * sizeof(int),
                           cudaMemcpyHostToDevice, stream));
    };

    const long act_slot = IL;   // fused epilogue writes `inter` floats per slot
    auto gateup = [&](int rows) {
        return dsv41_expert_gate_up_fp4_batched(d_a, d_asc, d_act, act_slot, rows, DIM, IL, LIMIT,
                                                SLOTS, w1_base, BLOCK, w1s_base, BLOCK, w3_base,
                                                BLOCK, w3s_base, BLOCK, d_ids, /*ilv=*/1,
                                                /*act_e4m3=*/0, stream);
    };

    auto run_point = [&](int rows) {
        for (int i = 0; i < warm; i++) { set_ids(rows, i); CK(gateup(rows)); }
        CK(cudaStreamSynchronize(stream));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, stream));
        for (int i = 0; i < iters; i++) { set_ids(rows, 1000 + i); CK(gateup(rows)); }
        CK(cudaEventRecord(e1, stream));
        CK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = ms * 1000.0 / iters;
        // weight bytes actually streamed: rows*slots*(n_total inter rows)*(kbytes)*2 planes
        // (kbytes = dim/2; gate+up alias ONE interleaved region of 2*that per row)
        const double bytes = (double)rows * SLOTS * IL * (double)(DIM / 2) * 2.0;
        const double gbs = bytes / (us * 1e-6) / 1e9;
        const double values = (double)rows * SLOTS * IL * DIM;   // fp4 values consumed
        printf("   rows=%-2d ids=%-8s %8.2f us/call  %7.1f GB/s  %.2f value/us/call-scale\n",
               rows, idmode, us, gbs, values / 1e6);
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
    };

    // ---- profiler window (for `ncu --profile-from-start off`) ---------------
    auto profile_point = [&](int rows) {
        for (int i = 0; i < 3; i++) { set_ids(rows, i); CK(gateup(rows)); }
        CK(cudaStreamSynchronize(stream));
        CK(cudaProfilerStart());
        for (int i = 0; i < 3; i++) { set_ids(rows, 100 + i); CK(gateup(rows)); }
        CK(cudaStreamSynchronize(stream));
        CK(cudaProfilerStop());
    };

    if (!strcmp(mode, "sweep")) {
        for (int rows = 1; rows <= 6; rows += (rows == 1 ? 2 : 3))   // 1, 3, 6
            run_point(rows);
    } else if (!strcmp(mode, "all")) {
        for (int rows = 1; rows <= MAX_ROWS; rows++) run_point(rows);
    } else {
        const int rows = atoi(mode);
        run_point(rows);
        profile_point(rows);
    }

    CK(cudaFree(d_a));
    CK(cudaFree(d_asc));
    CK(cudaFree(d_act));
    CK(cudaFree(d_ids));
    CK(cudaFree(pool));
    CK(cudaStreamDestroy(stream));
    printf("done\n");
    return 0;
}
