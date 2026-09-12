// =============================================================================
// tcgen05_mxf4_bench.cu — T1 ISOLATED MICROBENCH (go/no-go) for the tcgen05
//                         mxf4 routed-expert gate/up arm.
// =============================================================================
//
// THE ONE QUESTION
//   At the production shape, does the tcgen05 mxf4 gate/up kernel
//   (`dsv41_expert_tcgen05_gate_up_mxf4`, dsv41_experts_mxf4.cu:4228) beat the
//   SIMT path it would replace, and does it clear the T1 gate (22.2 us)?
//
//   docs/agent/final-400-config.md §P0-3 / §5-T1: "先过单层微基准门
//   (22.2µs / 17.2µs)。达标才继续；不达标 ⇒ tcgen05 关闭".
//   docs/agent/routed-expert-residual.md §3(b): "前置门：先跑单层 microbench
//   打 22.2µs（gateup）/17.2µs（down），打不过不进集成。"
//
// WHY A SEPARATE BINARY (and not the parity harness)
//   kernels/cuda/tests_tcgen05_mxf4_gateup.cu answers CORRECTNESS. This file
//   answers TIME. It links the PRODUCTION .so (the same launchers the serve step
//   calls, never a copy of the kernel body — the discipline of
//   scripts/expert_ncu_bench.cu) and reports a MEDIAN over ITERS individually
//   event-timed launches per arm. 100 iterations is the T1 contract.
//
// THE FIVE ARM-LEVEL TRAPS THIS FILE ENCODES (each one has already produced a
// wrong number in this repo's history):
//   (1) L2-WARM WEIGHTS ARE A DIFFERENT MACHINE. dsv41_experts_mxf4.cu:2508 and
//       :2681 both say so. The pool is therefore the FULL production size
//       (NEXP=384 experts x 2.6112 MB = 960 MiB >> 126 MiB L2) and the 6 slot
//       ids ROTATE every iteration, so each launch streams DRAM-cold expert
//       weights exactly like a real layer.
//   (2) THE POOL LAYOUT IS AN ARM. The production SIMT gate/up runs ILV=1
//       (gate/up interleaved in ONE plane, one LDG.128 per k-group); the tcgen05
//       kernel needs the loader's DIRECT layout (w1 and w3 as SEPARATE planes,
//       `load.rs:644`). Feeding the tcgen05 arm an ILV pool would make its six
//       slots read gate-as-up bytes (wrong, and a different access pattern), and
//       feeding the SIMT arm a direct pool would understate the baseline. BOTH
//       pools are therefore allocated at full size and each arm is documented
//       with the pool it reads.
//   (3) THE GATE IS A PROCESS STATIC. `DSV41_EXPERT_TCGEN05[_MXF4]` is read once
//       per process (dsv41_experts_mxf4.cu:4243) and the entry returns 0 WITHOUT
//       WRITING when it is off — so "rc == 0" cannot distinguish "ran" from
//       "silently measured the old path" (the project's #1 measurement-bias
//       trap). A sentinel write-check is printed as `TCGEN05_RAN` and the script
//       must assert it before believing any tcgen05 number.
//   (4) PER-ITERATION EVENTS, NOT ONE WINDOW. A batched window reports the MEAN
//       (and hides a bimodal distribution); T1 is a MEDIAN gate. Every
//       iteration gets its own event pair, and the ids copy sits OUTSIDE the
//       window (set_ids is enqueued before e0).
//   (5) SM-COUNT DISCLOSURE. The tcgen05 gate/up grid is (2*inter/128, slots)
//       = (5, 6) = 30 CTAs at the production shape, i.e. 30 of 148 SMs. That is
//       the documented occupancy limit (plan §1d-4: 256 TMEM columns/CTA => 2
//       CTA/SM), and at slots=1 it collapses to 5 CTAs. The printed grid is part
//       of the result: a number taken at a smaller slot count is a LATENCY
//       measurement, not a bandwidth one.
//
// THE DOWN HALF
//   There is NO tcgen05 down kernel in the tree (routed-expert-residual.md
//   §3(b)-①: "只有 gateup 骨架，down 的 swapAB + fused asc-slot reduce 未写").
//   The 17.2 us gate is still measured — on the CURRENT down path
//   (`dsv41_expert_down_reduce_fp4_batched`, the production fused down+reduce)
//   — so the script can state the reference and mark the tcgen05 side
//   NOT-IMPLEMENTED instead of inventing an ABI.
//
// BUILD (on the GPU node; the .so must already exist in the same tree)
//   nvcc -O3 --use_fast_math -std=c++17 \
//        -gencode arch=compute_103a,code=sm_103a \
//        -o /tmp/tcgen05_mxf4_bench scripts/tcgen05_mxf4_bench.cu \
//        -L kernels/cuda -l:libferrite_kernels.so
//   (nvcc does NOT accept -Wl,-rpath here — use LD_LIBRARY_PATH, see
//    scripts/dsv41_microbench.sh.)
//   The .so MUST have been built with -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1
//   (build.sh: default ON since 2026-09-12) or the link fails on the missing
//   symbol. scripts/tcgen05_bench.sh checks that with `nm -D` BEFORE compiling.
//
// RUN
//   export LD_LIBRARY_PATH=$PWD/kernels/cuda
//   # gate OFF — proves the sentinel path (#3) and gives the SIMT reference:
//   DSV41_EXPERT_TCGEN05=0 /tmp/tcgen05_mxf4_bench
//   # gate ON — the T1 measurement:
//   DSV41_EXPERT_TCGEN05=1 DSV41_EXPERT_ILV=0 /tmp/tcgen05_mxf4_bench
//   flags: --iters N (default 100)  --warmup N (20)  --rows N (5)
//          --slots N (6)  --nexp N (384)  --dim N (5120)  --inter N (320)
//
//   drivers: scripts/tcgen05_bench.sh (build + both passes + verdict).
//
// OUTPUT (machine-readable, parsed by the driver)
//   MEDIAN_US <tag> <us>          one line per arm that completed
//   TCGEN05_RAN yes|no            the sentinel verdict for the tcgen05 arm
// =============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

// ---- production launchers (kernels/cuda/libferrite_kernels.so) --------------
extern "C" {
// dsv41_experts_mxf4.cu:2454 — ABI 2: `ilv` is the trailing switch. `ilv == 1`
// requires the fused body + the interleaved pool; `ilv == 0` with
// out_slot_stride == 2*inter is the unfused [2*inter] form (the layout the
// tcgen05 arm also writes, which is why the two are comparable).
int dsv41_expert_gate_up_fp4_batched(const uint8_t* a, const float* a_scale, float* out,
                                     long out_slot_stride, int rows, int dim, int inter, float limit,
                                     int slots, const uint8_t* w1_base, long w1_stride,
                                     const uint8_t* w1s_base, long w1s_stride,
                                     const uint8_t* w3_base, long w3_stride,
                                     const uint8_t* w3s_base, long w3s_stride, const int* ids,
                                     int ilv, cudaStream_t stream);
// dsv41_experts_mxf4.cu:2625 — the PRODUCTION down (DSV41_DOWN_FUSE default ON,
// nsys sees this kernel 40x/step): one launch = the [slots] down GEMV + the
// ascending-slot sum.
int dsv41_expert_down_reduce_fp4_batched(const float* act_base, long act_stride, float* out,
                                         int rows, int dim, int inter, const float* row_weight,
                                         long rw_stride, int slots, const uint8_t* w2_base,
                                         long w2_stride, const uint8_t* w2s_base, long w2s_stride,
                                         const int* ids, cudaStream_t stream);
// dsv41_experts_mxf4.cu:4228 — the T1 subject. 18 parameters; `ids != nullptr`
// selects the per-slot expert indirection (the production form) and the four
// `*_base`/`*_stride` pairs are the loader's DIRECT planes.
int dsv41_expert_tcgen05_gate_up_mxf4(const uint8_t* act, const float* act_scale, float* out,
                                      long out_slot_stride, int inter, int dim, float limit,
                                      int slots, const uint8_t* w1_base, long w1_stride,
                                      const uint8_t* w1s_base, long w1s_stride,
                                      const uint8_t* w3_base, long w3_stride,
                                      const uint8_t* w3s_base, long w3s_stride, const int* ids,
                                      cudaStream_t stream);
// ferrite_kernels.cu:33 — proves WHICH .so got loaded (AGENTS.md 加载错防线).
const char* ferrite_kernel_build_id(void);
unsigned ferrite_kernel_abi_version(void);
}

#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA err @%d %s: %s\n", __LINE__, #x,                     \
                    cudaGetErrorString(e_));                                           \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

// ============================================================================
// PRODUCTION SHAPE (config.rs:496-506 / configs/dsv41_flash.json, --tp 8)
//   dim = 5120 (K of gate/up, N of down) · moe_inter_dim 2304 · topk 6 ·
//   n_routed 384 · world 8 · inter_local = padded_inter(2304/8) = 320
//   (weights.rs:442 padded_inter = ceil(n/64)*64, K_ATOM = 64)
// `rows` is the T1 "m": the number of ACTIVATION rows per launch (grid.z of the
// SIMT kernels). The tcgen05 gate/up kernel has NO rows dimension — its grid is
// (2*inter/128, slots) and one launch serves ONE activation row — so the
// dedicated rows==1 SIMT arms exist to give the same-work comparison.
// ============================================================================
static int DIM = 5120;      // cfg.dim
static int IL = 320;        // inter_local
static int SLOTS = 6;       // n_activated_experts (topk)
static int NEXP = 384;      // n_routed_experts — the FULL production pool
static int ROWS = 5;        // T1 m
static float LIMIT = 10.0f; // cfg.swiglu_limit
static int ITERS = 100;     // T1: median over 100 launches
static int WARMUP = 20;

// ---------------------------------------------------------------------------
// Per-expert plane sizes. IDENTICAL for both pool layouts, which is why the
// two per-expert BLOCKs come out the same size (2,611,200 B).
// ---------------------------------------------------------------------------
static long W1B, W1SB, W2B, W2SB, BLOCK;
// ILV (production, expert_ncu_bench.cu:238-248): [w1||w3][w1s][w3s][w2][w2s]
static long I_W1, I_W1S, I_W3, I_W3S, I_W2, I_W2S;
// DIR (loader order, load.rs:644): [w1][w1s][w3][w3s][w2][w2s]
static long D_W1, D_W1S, D_W3, D_W3S, D_W2, D_W2S;

static void layout_init() {
    W1B = (long)IL * (DIM / 2);
    W1SB = (long)IL * (DIM / 32);
    W2B = (long)DIM * (IL / 2);
    W2SB = (long)DIM * (IL / 32);
    // ILV: w1 and w3 live interleaved in ONE plane; w3 aliases offset 0 because
    // the ilv=1 body never reads the separate w3 plane (expert_ncu_bench.cu:303).
    I_W1 = 0;
    I_W1S = 2 * W1B;
    I_W3 = 0;
    I_W3S = I_W1S + W1SB;
    I_W2 = I_W3S + W1SB;
    I_W2S = I_W2 + W2B;
    BLOCK = I_W2S + W2SB;
    // DIR: the loader's own order (this is what the tcgen05 arm and the parity
    // harness feed; `split` = IL is the row boundary between the two planes).
    D_W1 = 0;
    D_W1S = W1B;
    D_W3 = W1B + W1SB;
    D_W3S = D_W3 + W1B;
    D_W2 = D_W3S + W1SB;
    D_W2S = D_W2 + W2B;
}

// ---------------------------------------------------------------------------
// Data-dependent LUT indices: a memset-filled pool would make every lane gather
// the same bank and hide the real access pattern (expert_ncu_bench.cu:250).
// ---------------------------------------------------------------------------
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

// The output sentinel (same value as the parity harness). It is an extreme fp32
// pattern that no accumulation over fp4 operands produces, so "still poisoned"
// is a reliable "the kernel did not write".
static const float kPoison = -1.2345e30f;

struct Stat {
    double med = 0.0, mn = 0.0, mean = 0.0;
    bool ok = false;
};

static cudaEvent_t g_e0 = nullptr, g_e1 = nullptr;
static int* d_ids = nullptr;          // [ROWS * SLOTS]
static std::vector<int> h_ids;        // host mirror
static long g_ids_calls = 0;

// Rotate the 6 (x ROWS) expert ids so every iteration streams different, DRAM-
// cold weights (trap #1). Enqueued BEFORE the event window, so the 480-byte H2D
// is never billed to the kernel.
static void set_ids(long iter, cudaStream_t s) {
    for (int r = 0; r < ROWS; ++r)
        for (int k = 0; k < SLOTS; ++k)
            h_ids[(size_t)r * SLOTS + k] =
                (int)(((long)k * 61 + iter * SLOTS + r) % (long)NEXP);
    CK(cudaMemcpyAsync(d_ids, h_ids.data(), (size_t)ROWS * SLOTS * sizeof(int),
                       cudaMemcpyHostToDevice, s));
    ++g_ids_calls;
}

// Per-iteration events (trap #4). Reports the MEDIAN, the min and the mean so a
// bimodal distribution is visible.
template <class F>
static Stat bench(const char* tag, F&& fn, cudaStream_t s) {
    Stat st;
    for (int i = 0; i < WARMUP; ++i) {
        set_ids(i, s);
        fn();
    }
    const cudaError_t we = cudaStreamSynchronize(s);
    if (we != cudaSuccess) {
        printf("  %-24s WARMUP CUDA ERROR: %s\n", tag, cudaGetErrorString(we));
        (void)cudaGetLastError();
        return st;
    }
    (void)cudaGetLastError();
    std::vector<double> t((size_t)ITERS);
    for (int i = 0; i < ITERS; ++i) {
        set_ids(WARMUP + i, s);
        CK(cudaEventRecord(g_e0, s));
        fn();
        CK(cudaEventRecord(g_e1, s));
        const cudaError_t se = cudaEventSynchronize(g_e1);
        if (se != cudaSuccess) {
            printf("  %-24s CUDA ERROR: %s\n", tag, cudaGetErrorString(se));
            (void)cudaGetLastError();
            return st;
        }
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, g_e0, g_e1));
        t[(size_t)i] = (double)ms * 1000.0;
    }
    std::sort(t.begin(), t.end());
    st.med = t[t.size() / 2];
    st.mn = t.front();
    st.mean = std::accumulate(t.begin(), t.end(), 0.0) / (double)t.size();
    st.ok = true;
    printf("  %-24s median %9.3f us   min %9.3f   mean %9.3f   n=%d\n", tag, st.med, st.mn,
           st.mean, ITERS);
    printf("MEDIAN_US %s %.3f\n", tag, st.med);
    return st;
}

static const char* env_or(const char* k, const char* dflt) {
    const char* v = getenv(k);
    return (v == nullptr || v[0] == '\0') ? dflt : v;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (!strcmp(a, "--iters")) { const char* v = next(); if (v) ITERS = atoi(v); }
        else if (!strcmp(a, "--warmup")) { const char* v = next(); if (v) WARMUP = atoi(v); }
        else if (!strcmp(a, "--rows")) { const char* v = next(); if (v) ROWS = atoi(v); }
        else if (!strcmp(a, "--slots")) { const char* v = next(); if (v) SLOTS = atoi(v); }
        else if (!strcmp(a, "--nexp")) { const char* v = next(); if (v) NEXP = atoi(v); }
        else if (!strcmp(a, "--dim")) { const char* v = next(); if (v) DIM = atoi(v); }
        else if (!strcmp(a, "--inter")) { const char* v = next(); if (v) IL = atoi(v); }
        else if (!strcmp(a, "--limit")) { const char* v = next(); if (v) LIMIT = (float)atof(v); }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            printf("usage: %s [--iters N] [--warmup N] [--rows N] [--slots N] [--nexp N]\n"
                   "          [--dim N] [--inter N] [--limit F]\n", argv[0]);
            return 0;
        } else {
            printf("unknown arg: %s (try --help)\n", a);
            return 2;
        }
    }
    if (ITERS < 1) ITERS = 1;
    if (ROWS < 1) ROWS = 1;
    if (SLOTS < 1) SLOTS = 1;
    if (NEXP < SLOTS) NEXP = SLOTS;
    layout_init();
    h_ids.assign((size_t)ROWS * SLOTS, 0);

    CK(cudaSetDevice(0));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    CK(cudaEventCreate(&g_e0));
    CK(cudaEventCreate(&g_e1));

    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));

    printf("== tcgen05 mxf4 T1 microbench — build_id=%s  abi=%u\n",
           ferrite_kernel_build_id ? ferrite_kernel_build_id() : "(null)",
           ferrite_kernel_abi_version ? ferrite_kernel_abi_version() : 0u);
    printf("   device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);
    if (prop.major != 10 && prop.major != 11)
        printf("   WARNING: tcgen05 targets sm_103a (B300); this GPU is sm_%d%d — expect a\n"
               "            launch failure, not a slow number\n", prop.major, prop.minor);
    printf("   shape : dim=%d inter=%d slots=%d experts=%d rows(m)=%d limit=%.1f\n", DIM, IL, SLOTS,
           NEXP, ROWS, (double)LIMIT);
    printf("   pools : %.1f MiB each (BLOCK=%ld B/expert x %d)  x2 (ILV production + DIR loader)\n",
           (double)((double)BLOCK * NEXP) / (1024.0 * 1024.0), BLOCK, NEXP);
    printf("   layout: ILV[w1=%ld w1s=%ld w3s=%ld w2=%ld w2s=%ld]  DIR[w1=%ld w1s=%ld w3=%ld "
           "w3s=%ld w2=%ld w2s=%ld]\n",
           I_W1, I_W1S, I_W3S, I_W2, I_W2S, D_W1, D_W1S, D_W3, D_W3S, D_W2, D_W2S);
    printf("   timing: warmup=%d iters=%d (median)\n", WARMUP, ITERS);
    printf("   simt grids: gateup ilv1 rows=%d -> (ceil(%d/%d), %d, %d) = (%d, %d, %d)\n", ROWS,
           IL, 8, SLOTS, ROWS, (IL + 7) / 8, SLOTS, ROWS);
    printf("   tcgen05 grid: (2*inter/128, slots) = (%d, %d) = %d CTAs (of %d SMs)\n", (2 * IL) / 128,
           SLOTS, (2 * IL) / 128 * SLOTS, prop.multiProcessorCount);
    printf("   env   :");
    const char* envs[] = {"DSV41_EXPERT_TCGEN05", "DSV41_EXPERT_TCGEN05_MXF4", "DSV41_EXPERT_ILV",
                          "DSV41_GATEUP_FUSE",   "DSV41_DOWN_FUSE",            "DSV41_GATEUP_KSPLIT",
                          "DSV41_GATEUP_ROWS",   "DSV41_EXPERT_FP4_MODE",     "DSV41_DOWN_VEC4",
                          "DSV41_PDL",           "DSV41_W2_PREWARM"};
    for (const char* e : envs) printf(" %s=%s", e, env_or(e, "-"));
    printf("\n");

    // ---------------- weights: TWO full-size pools ------------------------
    uint8_t* pool_ilv = nullptr;
    uint8_t* pool_dir = nullptr;
    const size_t pool_bytes = (size_t)BLOCK * (size_t)NEXP;
    CK(cudaMalloc(&pool_ilv, pool_bytes));
    CK(cudaMalloc(&pool_dir, pool_bytes));
    fill_rand(pool_ilv, pool_bytes, 12345u, s);
    fill_rand(pool_dir, pool_bytes, 67890u, s);
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());

    const uint8_t* i_w1 = pool_ilv + I_W1;
    const uint8_t* i_w1s = pool_ilv + I_W1S;
    const uint8_t* i_w3 = pool_ilv + I_W3;   // aliased under ILV (never read)
    const uint8_t* i_w3s = pool_ilv + I_W3S;
    const uint8_t* i_w2 = pool_ilv + I_W2;
    const uint8_t* i_w2s = pool_ilv + I_W2S;
    const uint8_t* d_w1 = pool_dir + D_W1;
    const uint8_t* d_w1s = pool_dir + D_W1S;
    const uint8_t* d_w3 = pool_dir + D_W3;
    const uint8_t* d_w3s = pool_dir + D_W3S;
    // DIR also carries the loader's down planes (d_w2 = pool + D_W2, d_w2s =
    // pool + D_W2S) because the layout must stay complete; only the ILV pool
    // feeds the down arm today. The tcgen05 DOWN arm does not exist yet
    // (residual §3(b)-①) — it will want THOSE pointers, which is why the offsets
    // are computed and printed above.

    // ---------------- activations / outputs --------------------------------
    uint8_t* d_a = nullptr;        // [ROWS][DIM/2] packed fp4 activation
    float* d_asc = nullptr;        // [ROWS][DIM/32] f32 power-of-two scales
    float* d_act_ilv = nullptr;    // [ROWS][SLOTS][IL] gateup(fused) out -> down in
    float* d_out_gv = nullptr;     // [ROWS][SLOTS][2*IL] unfused gate|up (ilv0 arm)
    float* d_out_tc = nullptr;     // [SLOTS][2*IL] tcgen05 out (one activation row)
    float* d_out_down = nullptr;   // [ROWS*SLOTS][DIM], over-sized: timing only
    float* d_rw = nullptr;         // [ROWS*SLOTS] routing weights
    CK(cudaMalloc(&d_a, (size_t)ROWS * (DIM / 2)));
    CK(cudaMalloc(&d_asc, (size_t)ROWS * (DIM / 32) * sizeof(float)));
    CK(cudaMalloc(&d_act_ilv, (size_t)ROWS * SLOTS * IL * sizeof(float)));
    CK(cudaMalloc(&d_out_gv, (size_t)ROWS * SLOTS * 2 * IL * sizeof(float)));
    CK(cudaMalloc(&d_out_tc, (size_t)SLOTS * 2 * IL * sizeof(float)));
    CK(cudaMalloc(&d_out_down, (size_t)ROWS * SLOTS * DIM * sizeof(float)));
    CK(cudaMalloc(&d_rw, (size_t)ROWS * SLOTS * sizeof(float)));
    CK(cudaMalloc(&d_ids, (size_t)ROWS * SLOTS * sizeof(int)));
    fill_rand(d_a, (size_t)ROWS * (DIM / 2), 777u, s);
    fill_rand(reinterpret_cast<uint8_t*>(d_act_ilv),
              (size_t)ROWS * SLOTS * IL * sizeof(float), 778u, s);
    fill_rand(reinterpret_cast<uint8_t*>(d_out_down),
              (size_t)ROWS * SLOTS * DIM * sizeof(float), 779u, s);
    {
        // The activation scales must be POWERS OF TWO: the tcgen05 arm converts
        // them to e8m0 bytes by exponent copy (dsv41_experts_mxf4.cu:4034-4048),
        // and the ABI note (kernels.rs:159) says only the exponent is seen.
        std::vector<float> h((size_t)ROWS * (DIM / 32));
        for (size_t i = 0; i < h.size(); ++i) h[i] = ldexpf(1.0f, (int)(i % 5) - 2);
        CK(cudaMemcpy(d_asc, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> rw((size_t)ROWS * SLOTS);
        for (size_t i = 0; i < rw.size(); ++i) rw[i] = 0.5f + 0.1f * (float)(i % SLOTS);
        CK(cudaMemcpy(d_rw, rw.data(), rw.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    {
        std::vector<int> z((size_t)ROWS * SLOTS, 0);
        CK(cudaMemcpy(d_ids, z.data(), z.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CK(cudaStreamSynchronize(s));

    // ---------------- launch closures --------------------------------------
    const long act_slot = IL;              // fused gate/up writes `inter` f32 per slot
    const long out_2il = (long)(2 * IL);   // unfused / tcgen05 pitch

    auto gateup_ilv1 = [&](int rows) {
        return dsv41_expert_gate_up_fp4_batched(d_a, d_asc, d_act_ilv, act_slot, rows, DIM, IL,
                                                LIMIT, SLOTS, i_w1, BLOCK, i_w1s, BLOCK, i_w3,
                                                BLOCK, i_w3s, BLOCK, d_ids, /*ilv=*/1, s);
    };
    auto gateup_ilv0 = [&](int rows) {
        return dsv41_expert_gate_up_fp4_batched(d_a, d_asc, d_out_gv, out_2il, rows, DIM, IL, LIMIT,
                                                SLOTS, d_w1, BLOCK, d_w1s, BLOCK, d_w3, BLOCK,
                                                d_w3s, BLOCK, d_ids, /*ilv=*/0, s);
    };
    auto down_simt = [&](int rows) {
        return dsv41_expert_down_reduce_fp4_batched(d_act_ilv, act_slot, d_out_down, rows, DIM, IL,
                                                    d_rw, /*rw_stride=*/1, SLOTS, i_w2, BLOCK,
                                                    i_w2s, BLOCK, d_ids, s);
    };
    auto tcgen05_gateup = [&]() {
        // ids != nullptr => the production per-slot indirection (6 different
        // experts per launch). `inter` = IL, `out_slot_stride` = 2*IL (the same
        // [2*inter] form the ilv0 SIMT arm writes).
        return dsv41_expert_tcgen05_gate_up_mxf4(d_a, d_asc, d_out_tc, out_2il, IL, DIM, LIMIT,
                                                 SLOTS, d_w1, BLOCK, d_w1s, BLOCK, d_w3, BLOCK,
                                                 d_w3s, BLOCK, d_ids, s);
    };

    // ---------------- SIMT arms FIRST --------------------------------------
    // If the tcgen05 arm faults (misaligned TMA / illegal instruction) it
    // POISONS the context and every later launch returns the sticky error (the
    // parity harness learned this the hard way). The baseline numbers are
    // therefore taken first, so a bring-up fault still leaves them on record.
    printf("\n-- SIMT baselines (the arm tcgen05 must beat) --\n");
    bench("gateup_simt_ilv1", [&] { gateup_ilv1(ROWS); }, s);
    bench("gateup_simt_ilv1_r1", [&] { gateup_ilv1(1); }, s);
    bench("gateup_simt_ilv0", [&] { gateup_ilv0(ROWS); }, s);
    bench("gateup_simt_ilv0_r1", [&] { gateup_ilv0(1); }, s);
    bench("down_simt", [&] { down_simt(ROWS); }, s);
    bench("down_simt_r1", [&] { down_simt(1); }, s);

    // ---------------- tcgen05 arm (last) -----------------------------------
    printf("\n-- tcgen05 mxf4 gate/up --\n");
    bool tc_ran = false;
    {
        const size_t n = (size_t)SLOTS * (size_t)(2 * IL);
        std::vector<float> poison(n, kPoison);
        CK(cudaMemcpyAsync(d_out_tc, poison.data(), n * sizeof(float), cudaMemcpyHostToDevice, s));
        const int rc = tcgen05_gateup();
        const cudaError_t la = cudaStreamSynchronize(s);
        const cudaError_t st = cudaGetLastError();
        (void)cudaGetLastError();
        std::vector<float> got(n, 0.f);
        CK(cudaMemcpy(got.data(), d_out_tc, n * sizeof(float), cudaMemcpyDeviceToHost));
        size_t left = 0;
        for (size_t i = 0; i < n; ++i) {
            uint32_t x = 0, y = 0;
            memcpy(&x, &got[i], 4);
            memcpy(&y, &poison[i], 4);
            if (x == y) ++left;
        }
        tc_ran = (left == 0);
        printf("  sentinel: rc=%d sync=%s sticky=%s  outputs still poisoned %zu/%zu\n", rc,
               cudaGetErrorString(la), cudaGetErrorString(st), left, n);
        printf("  -> the gate %s\n",
               tc_ran ? "is ON and the kernel WROTE its output"
                      : "is OFF (rc==0 and nothing written — the entry's silent no-op path; "
                        "set DSV41_EXPERT_TCGEN05=1)");
        printf("TCGEN05_RAN %s\n", tc_ran ? "yes" : "no");
    }
    if (tc_ran) {
        bench("tcgen05_gateup", tcgen05_gateup, s);
        // Slot uniformity is NOT checked here (that is the parity harness's job);
        // one extra launch on a fresh sentinel also re-proves the arm ran inside
        // the timing loop.
        std::vector<float> poison((size_t)SLOTS * (size_t)(2 * IL), kPoison);
        CK(cudaMemcpy(d_out_tc, poison.data(), poison.size() * sizeof(float),
                      cudaMemcpyHostToDevice));
        tcgen05_gateup();
        CK(cudaStreamSynchronize(s));
        std::vector<float> got(poison.size(), 0.f);
        CK(cudaMemcpy(got.data(), d_out_tc, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        size_t left = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (memcmp(&got[i], &poison[i], 4) == 0) ++left;
        printf("  post-timing re-check: still poisoned %zu/%zu (0 = the loop really ran it)\n",
               left, got.size());
    } else {
        printf("  SKIP tcgen05_gateup bench (the gate is off — no number would mean anything)\n");
    }

    // ---------------- summary ---------------------------------------------
    printf("\n== done (ids rotations: %ld) ==\n", g_ids_calls);
    CK(cudaFree(pool_ilv));
    CK(cudaFree(pool_dir));
    CK(cudaFree(d_a));
    CK(cudaFree(d_asc));
    CK(cudaFree(d_act_ilv));
    CK(cudaFree(d_out_gv));
    CK(cudaFree(d_out_tc));
    CK(cudaFree(d_out_down));
    CK(cudaFree(d_rw));
    CK(cudaFree(d_ids));
    CK(cudaEventDestroy(g_e0));
    CK(cudaEventDestroy(g_e1));
    CK(cudaStreamDestroy(s));
    return 0;
}
