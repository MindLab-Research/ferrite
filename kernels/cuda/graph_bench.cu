// graph_bench.cu - CUDA-graph NODE DISPATCH overhead micro-benchmark (standalone).
//
// THE QUESTION IT ANSWERS
//   The DSV4.1 decode step issues ~1500-2000 small kernels. On the plain stream
//   launch path each one costs ~3.05 us of launch overhead (~5-6 ms/step, 13-16%).
//   Once the whole-step graph is replayed, that host cost is amortised, and what
//   remains is the per-node cost. Two figures are on the table and they differ 7x:
//       ~0.2 us/node -> node reduction is worth ~0.04 ms (rounding error)
//       ~1.5 us/node -> node reduction is worth ~0.5-1.2 ms (headline lever)
//   This program measures the per-node figure directly, without loading the 306 GB
//   model (a single B300 cannot hold it, so the in-model nsys route is unavailable).
//
// WHAT IT MEASURES (all on one non-default stream, kernels serialised as captured)
//   1. HOST launch cost    : CPU wall time to submit N kernel launches (no sync).
//   2. STREAM wall cost    : GPU-event time of N individual launches, sync'd.
//   3. GRAPH replay cost   : GPU-event time of one cudaGraphLaunch of an N-node graph.
//   4. NODE DISPATCH FLOOR : replay of an N-node graph of an EMPTY kernel. With the
//                            work removed, T/N is pure per-node dispatch (this is the
//                            0.2 vs 1.5 us number).
//   5. Sum(kernel exec)    : measured two ways - (a) globaltimer stamped inside the
//                            kernel, (b) T_real_graph - T_empty_graph, which cancels
//                            the dispatch and host terms.
//
// THE KEY SUBTLETY (why "0.2 us" can be an illusion)
//   A single cudaGraphLaunch of an N-node graph costs ONE host-side submit (~3-5 us)
//   regardless of N. Dividing by N gives "0.2 us/node" for N~2000 - but that is a
//   HOST-AVERAGED number, and it says nothing about the DEVICE-side gap between two
//   dependent graph nodes. When a node's kernel is large the next node's launch is
//   pipelined/hidden; when nodes are small the device dispatch latency shows up as
//   the in-graph gap (exactly what the nsys --cuda-graph-trace=node measurement was
//   chasing). The EMPTY-KERNEL graph (measurement 4) is what isolates the device
//   dispatch, because it removes the execution that would otherwise hide it. Both
//   numbers are printed; do not quote one without the other.
//
// BUILD (B300 = sm_103a; run on a remote node with nvcc):
//   nvcc -O3 -std=c++17 -arch=sm_103a -o /tmp/graph_bench /tmp/graph_bench.cu
// RUN
//   /tmp/graph_bench                 # defaults: 1000 nodes, 50 iters
//   /tmp/graph_bench -n 2000 -i 100  # match the decode node budget
//   /tmp/graph_bench --elems 4096 --blocks 16 --reps 8   # ~10 us per node
// NOTES
//   * Take medians, not means: the first replays include lazy init and clock ramp.
//   * Clocks matter; for a clean A/B lock them (sudo nvidia-smi -lgc). The program
//     prints the SM clock once if it can (best effort, no privileged call).
//   * The graph is built by STREAM CAPTURE (simple, matches the engine) - capture
//     preserves the stream order, so the N nodes form a serial dependency chain.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x)                                                          \
    do {                                                               \
        cudaError_t e_ = (x);                                          \
        if (e_ != cudaSuccess) {                                       \
            printf("ERR %s @%d\n", cudaGetErrorString(e_), __LINE__);  \
            exit(1);                                                   \
        }                                                              \
    } while (0)

// Global nanosecond timer, roughly synchronised across SMs on modern parts. Used
// to stamp pure kernel execution time (measurement 5a) instead of clock64, which
// is per-SM and would need a calibration to convert to ns.
__device__ __forceinline__ unsigned long long gt_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Pure dispatch probe: no memory traffic, no work.
__global__ void nop_kernel() {}

// The representative decode-shaped kernel: a vector add over `elems` floats,
// swept `reps` times so the per-node execution time can be tuned toward ~10 us.
// When `ts` is non-null the first thread of the last block stamps entry/exit.
__global__ void vecadd_kernel(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ c, int elems, int reps,
                              unsigned long long* __restrict__ ts) {
    unsigned long long t0 = 0;
    if (ts && threadIdx.x == 0 && blockIdx.x == 0) t0 = gt_ns();
    for (int r = 0; r < reps; r++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elems;
             i += gridDim.x * blockDim.x) {
            c[i] = a[i] + b[i];
        }
        __syncthreads();  // keep the reps from being collapsed/overlapped
    }
    if (ts && threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence();
        ts[0] = t0;
        ts[1] = gt_ns();
    }
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// GPU-event wall time of one graph replay, median of `iters` samples (the first
// `warm` are discarded: they carry lazy module load / clock ramp).
static double bench_replay(cudaGraphExec_t ge, cudaStream_t s, int iters, int warm = 5) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < warm; i++) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    std::vector<double> ms;
    for (int i = 0; i < iters; i++) {
        CK(cudaEventRecord(a, s));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(b, s));
        CK(cudaStreamSynchronize(s));
        float t = 0.f;
        CK(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return median_of(ms);
}

// Wall time of a single cudaGraphLaunch of a 1-node graph: the un-amortised
// host-side submit cost (what an N-node graph divides by N).
static double bench_single_graph_launch(cudaGraphExec_t ge, cudaStream_t s, int iters) {
    return bench_replay(ge, s, iters);
}

int main(int argc, char** argv) {
    int N = 1000;        // nodes per graph
    int iters = 50;      // timing samples
    int elems = 4096;    // vecadd length
    int blocks = 16;     // vecadd grid
    int threads = 256;   // vecadd block
    int reps = 8;        // vecadd passes per node (tune toward ~10 us)
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--elems") && i + 1 < argc) elems = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else { printf("usage: %s [-n nodes] [-i iters] [--elems E] [--blocks B] [--threads T] [--reps R]\n", argv[0]); return 2; }
    }

    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("config: N=%d nodes, iters=%d, vecadd elems=%d blocks=%d threads=%d reps=%d\n\n",
           N, iters, elems, blocks, threads, reps);

    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    float* a;
    float* b;
    float* c;
    CK(cudaMalloc(&a, (size_t)elems * 4));
    CK(cudaMalloc(&b, (size_t)elems * 4));
    CK(cudaMalloc(&c, (size_t)elems * 4));
    CK(cudaMemset(a, 0, (size_t)elems * 4));
    CK(cudaMemset(b, 0, (size_t)elems * 4));
    unsigned long long* ts;
    CK(cudaMalloc(&ts, 2 * sizeof(unsigned long long)));

    // ---------------------------------------------------------------------
    // 1) HOST launch cost: submit N launches, time the CPU loop only.
    //    Uses the EMPTY kernel on purpose: with a ~10 us body the pending-launch
    //    queue backs up and the launch call blocks on GPU progress, so the number
    //    would measure GPU throughput, not host submit cost. The empty kernel
    //    retires instantly, so the loop is pure submit cost (the API cost is the
    //    same for both kernels).
    // ---------------------------------------------------------------------
    nop_kernel<<<1, 32, 0, s>>>();
    CK(cudaStreamSynchronize(s));  // warm
    auto h0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) nop_kernel<<<1, 32, 0, s>>>();
    auto h1 = std::chrono::steady_clock::now();
    double host_us_per_launch =
        std::chrono::duration<double, std::micro>(h1 - h0).count() / N;
    CK(cudaStreamSynchronize(s));

    // ---------------------------------------------------------------------
    // 2) STREAM wall cost: N individual launches, GPU-event timed (host-bound).
    // ---------------------------------------------------------------------
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<double> launch_ms;
    for (int it = 0; it < iters; it++) {
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < N; i++)
            vecadd_kernel<<<blocks, threads, 0, s>>>(a, b, c, elems, reps, nullptr);
        CK(cudaEventRecord(e1, s));
        CK(cudaStreamSynchronize(s));
        float t = 0.f;
        CK(cudaEventElapsedTime(&t, e0, e1));
        launch_ms.push_back(t);
    }
    double stream_launch_ms = median_of(launch_ms);

    // ---------------------------------------------------------------------
    // 3) GRAPH replay cost: capture the same N launches, replay, GPU-event timed.
    // ---------------------------------------------------------------------
    cudaGraph_t g_real = nullptr;
    cudaGraphExec_t ge_real = nullptr;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < N; i++)
        vecadd_kernel<<<blocks, threads, 0, s>>>(a, b, c, elems, reps, nullptr);
    CK(cudaStreamEndCapture(s, &g_real));
    CK(cudaGraphInstantiate(&ge_real, g_real, 0));
    // Query with a real buffer (not the NULL-count form) so this compiles and runs
    // identically across CUDA versions.
    std::vector<cudaGraphNode_t> nodes(N + 1);
    size_t n_nodes = nodes.size();
    CK(cudaGraphGetNodes(g_real, nodes.data(), &n_nodes));
    double graph_real_ms = bench_replay(ge_real, s, iters);

    // ---------------------------------------------------------------------
    // 4) DISPATCH FLOOR: N empty nodes -> T/N is per-node dispatch.
    // ---------------------------------------------------------------------
    cudaGraph_t g_nop = nullptr;
    cudaGraphExec_t ge_nop = nullptr;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < N; i++) nop_kernel<<<1, 32, 0, s>>>();
    CK(cudaStreamEndCapture(s, &g_nop));
    CK(cudaGraphInstantiate(&ge_nop, g_nop, 0));
    size_t n_nop_nodes = nodes.size();
    CK(cudaGraphGetNodes(g_nop, nodes.data(), &n_nop_nodes));
    double graph_nop_ms = bench_replay(ge_nop, s, iters);

    // 5) 1-node graph: the un-amortised host-side graph submit.
    cudaGraph_t g_one = nullptr;
    cudaGraphExec_t ge_one = nullptr;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    nop_kernel<<<1, 32, 0, s>>>();
    CK(cudaStreamEndCapture(s, &g_one));
    CK(cudaGraphInstantiate(&ge_one, g_one, 0));
    double graph_one_ms = bench_single_graph_launch(ge_one, s, iters);

    // ---------------------------------------------------------------------
    // 6) Sum(kernel exec): globaltimer-stamped single kernel, median of samples.
    // ---------------------------------------------------------------------
    std::vector<double> exec_us;
    for (int it = 0; it < iters + 5; it++) {
        CK(cudaMemset(ts, 0, 2 * sizeof(unsigned long long)));
        vecadd_kernel<<<blocks, threads, 0, s>>>(a, b, c, elems, reps, ts);
        CK(cudaStreamSynchronize(s));
        unsigned long long hts[2] = {0, 0};
        CK(cudaMemcpy(hts, ts, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (it >= 5 && hts[1] > hts[0]) exec_us.push_back((double)(hts[1] - hts[0]) / 1000.0);
    }
    double kernel_exec_us = median_of(exec_us);

    // ---------------------------------------------------------------------
    // REPORT
    // ---------------------------------------------------------------------
    double graph_us_per_node = graph_real_ms * 1000.0 / N;
    double stream_us_per_node = stream_launch_ms * 1000.0 / N;
    double nop_us_per_node = graph_nop_ms * 1000.0 / N;
    double sum_kernel_via_diff_us = (graph_real_ms - graph_nop_ms) * 1000.0 / N;

    printf("== results (median, ms unless noted) ==\n");
    printf("  nodes captured            : %zu real, %zu empty\n", n_nodes, n_nop_nodes);
    printf("  host submit / launch      : %8.3f us   (CPU loop, no sync)\n", host_us_per_launch);
    printf("  stream: N launches wall   : %8.3f ms  = %6.3f us/launch\n",
           stream_launch_ms, stream_us_per_node);
    printf("  graph : N-node replay     : %8.3f ms  = %6.3f us/node  (node dispatch + exec)\n",
           graph_real_ms, graph_us_per_node);
    printf("  graph : N EMPTY nodes     : %8.3f ms  = %6.3f us/node  <<< DISPATCH FLOOR\n",
           graph_nop_ms, nop_us_per_node);
    printf("  graph : 1-node replay     : %8.3f ms             (un-amortised submit)\n",
           graph_one_ms);
    printf("  kernel exec (globaltimer) : %8.3f us\n", kernel_exec_us);
    printf("  kernel exec (graph - nop) : %8.3f us   (dispatch-cancelling cross-check)\n",
           sum_kernel_via_diff_us);
    printf("\n");

    // The graph advantage is about OVERHEAD, not total wall: stream_us_per_node is
    // exec-dominated once the kernel body is ~10 us, so subtracting it from the
    // dispatch would report the kernel body as "saved". Compare the two OVERHEADS:
    // the host submit cost on the stream path vs the device dispatch in the graph.
    double overhead_saving_us = host_us_per_launch - nop_us_per_node;
    printf("== the 0.2 vs 1.5 us question ==\n");
    printf("  per-node dispatch in the graph   : %.3f us   <<< the figure in question\n",
           nop_us_per_node);
    printf("  host submit cost per launch      : %.3f us\n", host_us_per_launch);
    printf("  overhead saved per node by graph : %.3f us\n", overhead_saving_us);
    printf("  => at 1800 nodes/step            : %.3f ms/step of launch overhead removed\n",
           overhead_saving_us * 1800 / 1000.0);
    printf("  measured wall (stream - graph)   : %.3f ms  = %.3f ms at 1800 nodes\n",
           stream_launch_ms - graph_real_ms,
           (stream_launch_ms - graph_real_ms) * 1800.0 / N);
    printf("  [wall diff is only what is EXPOSED: with %.1f us kernels the host submit\n",
           kernel_exec_us);
    printf("   (%.1f us) is mostly hidden behind execution, so the graph's wall win is far\n",
           host_us_per_launch);
    printf("   smaller than the overhead saving - it appears when kernels are short.]\n");
    if (nop_us_per_node <= 0.3) {
        printf("  >> 0.2 us CALIBER HOLDS: in-graph node dispatch is near-free; node\n");
        printf("     reduction is a rounding error (~%.2f ms/step). Spend effort on real\n",
               1800 * nop_us_per_node / 1000.0);
        printf("     kernel time, not on node count.\n");
    } else if (nop_us_per_node >= 1.5) {
        printf("  >> 1.5 us CALIBER HOLDS: in-graph node dispatch dominates; node reduction\n");
        printf("     is a HEADLINE lever (~%.2f ms/step at 1800 nodes).\n",
               1800 * nop_us_per_node / 1000.0);
    } else {
        printf("  >> INTERMEDIATE: node dispatch is a MEDIUM lever (~%.2f ms/step at 1800 nodes).\n",
               1800 * nop_us_per_node / 1000.0);
    }
    printf("  CAVEAT: the empty-node graph exposes dispatch fully. With real kernels the\n");
    printf("          next node's launch can be pipelined behind the current one, so the\n");
    printf("          EFFECTIVE per-node cost in the decode graph can be lower.\n");
    printf("  NOTE: the 1-node replay above (%.3f ms) is the host submit an N-node graph\n",
           graph_one_ms);
    printf("        amortises; dividing IT by N is the illusory '0.2 us', not dispatch.\n");

    CK(cudaGraphExecDestroy(ge_real));
    CK(cudaGraphDestroy(g_real));
    CK(cudaGraphExecDestroy(ge_nop));
    CK(cudaGraphDestroy(g_nop));
    CK(cudaGraphExecDestroy(ge_one));
    CK(cudaGraphDestroy(g_one));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(c));
    CK(cudaFree(ts));
    CK(cudaStreamDestroy(s));
    return 0;
}
