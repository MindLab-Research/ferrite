// tests_tcgen05_mxf4_gateup.cu — GPU numerical parity for the Phase-1 mxf4
// tcgen05 swapAB gate/up kernel (kernels/cuda/dsv41_experts_mxf4.cu, block gated
// by DSV41_TCGEN05_GATEUP_MXF4_SKELETON at :3512, launcher at :4007).
//
// WHAT IT ANSWERS, in this order:
//   1. does the kernel RUN at all on the GPU (rc == 0, no sticky error, and the
//      OUTPUT WAS ACTUALLY WRITTEN — the launcher returns 0 both on success and
//      when the env gate is OFF, so a "0 means pass" reading would green-light a
//      no-op; the sentinel below is what distinguishes the two);
//   2. is it numerically right, against THREE references at once:
//        (a) GOLDEN_Q  — exact-arithmetic CPU golden over the QUANTISED operands
//                        (fp4 x fp4 with power-of-two block scales is exact
//                        arithmetic; only the fp32 summation order differs), bar
//                        1e-4 row-L2-relative;
//        (b) GEMV_REF  — the proven SIMT path `dsv41_expert_gate_up_fp4_batched`
//                        on the SAME packed nibbles and the SAME e8m0 weight
//                        scales, bar 1e-3 (it accumulates in a different order);
//        (c) GOLDEN_F32— CPU golden over the UNQUANTISED activation, i.e. the
//                        fp4 quantisation noise itself, bar 5e-2 — this is the
//                        number that says "the arm is not a different model".
//      Plus `slot-uniform`: with a direct pool (no ids) every grid.y slot must
//      produce BIT-IDENTICAL output — a cheap grid.y addressing check.
//   3. `--gate-off`: the default-OFF contract is a REAL no-op (rc == 0 and the
//      output buffer is bit-for-bit untouched).
//   4. `--graph`: the launch is CUDA-graph capturable and a replay reproduces
//      the direct run bit-for-bit (the serve path captures the whole step; the
//      env gate is read once per process exactly so that this stays legal).
//
// Build (needs nvcc, no GPU) — the -D is MANDATORY, without it the block is not
// compiled and this file has nothing to call:
//   nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//        -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 \
//        -o /tmp/t_mxf4_gateup kernels/cuda/tests_tcgen05_mxf4_gateup.cu
//
// Run (needs ONE free GPU; peak allocation ~12 MB at the production shape):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_mxf4_gateup              # parity suite
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_mxf4_gateup --gate-off   # no-op contract
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_mxf4_gateup --graph      # capture replay
//
// Driven by scripts/dsv41_tcgen05_mxf4_verify.sh (step 3).
//
// ⚠️ TWO ENV STATICS ARE READ ONCE PER PROCESS, so they are set in main() BEFORE
// the first call and can never be changed afterwards in the same process:
//   DSV41_EXPERT_TCGEN05_MXF4=1  arms the .so side of the gate (lambda-static
//                                inside the extern "C" entry);
//   DSV41_GATEUP_FUSE=0          forces the reference GEMV to the UNFUSED body
//                                (n_total = 2*inter gate|up) — the fused body
//                                would emit the swiglu'd [inter] result and the
//                                two arms would no longer be comparable at all.
// DSV41_EXPERT_FP4_MODE cannot be set here (namespace-scope static, initialised
// before main) — export it if a bisection needs a non-default mode. The default
// (2) is the production mode and is what this suite assumes.
#include "dsv41_experts_mxf4.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// ---------------------------------------------------------------- quant.rs ---
// Byte-identical helpers to tests_tcgen05_mxf4.cu (the file whose MXFP4 tcgen05
// GEMM already reports "all cases EXACT" on this GPU). Same table, same scale
// rounding (ceil to a power of two), same e8m0 encoding.
const float kTab[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                        0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

uint32_t rng_state;
uint32_t xrand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

uint8_t e8m0_of_exp(int e) { return (uint8_t)(e + 127); }

uint8_t e2m1_enc_h(float v) {
    const float a = fminf(fabsf(v), 6.0f);
    uint8_t c;
    if (a <= 0.25f) c = 0; else if (a <= 0.75f) c = 1; else if (a <= 1.25f) c = 2;
    else if (a <= 1.75f) c = 3; else if (a <= 2.5f) c = 4; else if (a <= 3.5f) c = 5;
    else if (a <= 5.0f) c = 6; else c = 7;
    return (uint8_t)(c | (v < 0.f ? 8u : 0u));
}

// The kernel's own scale rule (quant_kernel / quant_fp4_fused_kernel): the block
// scale is the smallest POWER OF TWO >= amax/6. Every scale this suite produces
// is therefore an exact e8m0 value — which is what makes the tcgen05 arm's
// e8m0-byte-only activation scale (the documented ABI asymmetry, kernels.rs:159)
// lossless here, and what the power-of-two assertion below re-checks.
float fast_round_scale6_h(float amax) {
    if (!(amax > 0.f)) return ldexpf(1.f, -126);
    const float r = amax / 6.0f;
    int e; frexpf(r, &e);
    uint32_t bits; memcpy(&bits, &r, 4);
    int ex = (int)((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) ? 1 : 0);
    if (ex < -126) ex = -126;
    if (ex > 127) ex = 127;
    return ldexpf(1.f, ex);
}

uint8_t f_pow2_ue8m0_h(float s) {
    if (!(s > 0.f)) return 0;
    uint32_t bits; memcpy(&bits, &s, 4);
    int e = (int)((bits >> 23) & 0xFF) - 127;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (uint8_t)(e + 127);
}

// ----------------------------------------------------------------- the case ---
struct Case {
    int dim = 0, inter = 0, slots = 0;
    float limit = 0.f;
    int rows() const { return 2 * inter; }
    std::vector<uint8_t> w;    // [2*inter, dim/2]   packed e2m1, low nibble = even k
    std::vector<uint8_t> ws;   // [2*inter, dim/32]  e8m0
    std::vector<uint8_t> act;  // [dim/2]            packed e2m1
    std::vector<uint8_t> acts; // [dim/32]           e8m0  (tcgen05 ABI)
    std::vector<float> actf;   // [dim/32]           f32   (GEMV ABI, same values)
    std::vector<double> wq;    // [2*inter, dim]     dequantised, for the golden
    std::vector<double> aq;    // [dim]              dequantised activation
    std::vector<double> xraw;  // [dim]              the unquantised activation
};

Case build_case(int dim, int inter, float limit, uint32_t seed) {
    Case c;
    c.dim = dim; c.inter = inter; c.limit = limit;
    const int rows = c.rows(), nw = dim / 32;
    c.w.assign((size_t)rows * (dim / 2), 0);
    c.ws.assign((size_t)rows * nw, 0);
    c.wq.assign((size_t)rows * dim, 0.0);

    rng_state = seed;
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < dim; ++k) {
            const int b = k / 32;
            if (k % 32 == 0) c.ws[(size_t)r * nw + b] = e8m0_of_exp((int)(xrand() % 5) - 2);
            const uint8_t code = (uint8_t)(xrand() & 0xF);
            c.wq[(size_t)r * dim + k] = kTab[code] * std::ldexp(1.0, (int)c.ws[(size_t)r * nw + b] - 127);
            c.w[(size_t)r * (dim / 2) + k / 2] |= (uint8_t)((code & 0xF) << (4 * (k & 1)));
        }

    c.act.assign(dim / 2, 0);
    c.acts.assign(nw, 0);
    c.actf.assign(nw, 0.f);
    c.aq.assign(dim, 0.0);
    c.xraw.assign(dim, 0.0);
    // Uniform noise in (-1, 1) with a 3x outlier every 32 elements: the outlier
    // is what forces a non-trivial block scale (without it every scale would be
    // 2^-2 and a scale-addressing bug could hide) AND exercises e2m1 saturation
    // (|q| > 6 is clamped by e2m1_enc_h exactly as the kernel clamps it).
    for (int k = 0; k < dim; ++k) {
        float v = ((float)(int)(xrand() % 20001) - 10000.f) / 10000.f;
        if (k % 32 == 7) v *= 3.0f;
        c.xraw[k] = (double)v;
    }
    for (int b = 0; b < nw; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf((float)c.xraw[b * 32 + i]));
        const float sc = fast_round_scale6_h(amax);
        const uint8_t eb = f_pow2_ue8m0_h(sc);
        // The two arms take the SAME scale, one as an e8m0 byte and one as the
        // f32 it decodes to. If the scale were not a power of two these two
        // would disagree — hence the assertion in run_case.
        c.acts[b] = eb;
        c.actf[b] = ldexpf(1.f, (int)eb - 127);
        for (int i = 0; i < 32; ++i) {
            const int k = b * 32 + i;
            const uint8_t code = e2m1_enc_h((float)c.xraw[k] / sc);
            c.act[k / 2] |= (uint8_t)((code & 0xF) << (4 * (k & 1)));
            c.aq[k] = kTab[code] * std::ldexp(1.0, (int)eb - 127);
        }
    }
    return c;
}

// Exact-arithmetic golden for the gate/up convention (`split == inter`, the
// epilogue both arms implement: rows [0, inter) clamp upward only, rows
// [inter, 2*inter) clamp both ways).
void golden(const Case& c, const std::vector<double>& actv, std::vector<double>& out) {
    const int rows = c.rows();
    out.assign(rows, 0.0);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int k = 0; k < c.dim; ++k) acc += c.wq[(size_t)r * c.dim + k] * actv[k];
        out[r] = acc;
    }
    if (c.limit > 0.f) {
        for (int r = 0; r < rows; ++r)
            out[r] = (r < c.inter) ? std::fmin(out[r], (double)c.limit)
                                   : std::fmin(std::fmax(out[r], -(double)c.limit), (double)c.limit);
    }
}

// ------------------------------------------------------------- comparison ---
struct Diff {
    double max_rel = 0.0, max_abs = 0.0, rel_l2 = 0.0;
    size_t worst = 0, nbit = 0;
};

// Row-L2-relative error is the PASS basis: a per-element relative error blows
// up on the near-zero elements a random-sign dot product produces, while the
// row norm is stable and is what a model actually consumes downstream.
Diff cmp(const std::vector<float>& got, const std::vector<double>& ref) {
    Diff d;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = (double)got[i], r = ref[i], e = std::fabs(g - r);
        num += e * e;
        den += r * r;
        d.max_abs = std::fmax(d.max_abs, e);
        const double rel = e / std::fmax(std::fabs(r), 1e-3);
        if (rel > d.max_rel) { d.max_rel = rel; d.worst = i; }
    }
    d.rel_l2 = std::sqrt(num / std::fmax(den, 1e-30));
    return d;
}

Diff cmp_exact(const std::vector<float>& a, const std::vector<float>& b) {
    Diff d;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x = 0, y = 0;
        memcpy(&x, &a[i], 4);
        memcpy(&y, &b[i], 4);
        if (x != y) ++d.nbit;
    }
    return d;
}

const float kPoison = -1.2345e30f;  // output sentinel: proves the kernel RAN

bool same_bits(float a, float b) {
    uint32_t x = 0, y = 0;
    memcpy(&x, &a, 4);
    memcpy(&y, &b, 4);
    return x == y;
}

// ----------------------------------------------------------------- the run ---
struct Cfg { const char* name; int dim; int inter; int slots; float limit; };

// Every row is a legal launcher shape: dim % 128 == 0 (2X SF pairing), dim % 64
// == 0 (ring atom), dim <= kMaxDim = 5120, rows = 2*inter % 128 == 0.
// `min-shape` is the smallest legal case; `prod-shape` is the production one
// (dim 5120, inter 2048, 8 slots = the drained top-k) and is the case the serve
// A/B will actually exercise.
const Cfg kCases[] = {
    {"min-shape    dim=128  inter=64   slots=1 limit=0",    128,   64, 1,  0.f},
    {"slots8+clamp dim=128  inter=64   slots=8 limit=10",   128,   64, 8, 10.f},
    {"odd-ring     dim=640  inter=128  slots=1 limit=10",   640,  128, 1, 10.f},
    {"prod-dim     dim=5120 inter=256  slots=8 limit=10",  5120,  256, 8, 10.f},
    {"prod-shape   dim=5120 inter=2048 slots=8 limit=10",  5120, 2048, 8, 10.f},
};

int run_case(const Cfg& cfg, int graph_mode) {
    const Case c = build_case(cfg.dim, cfg.inter, cfg.limit, 0xC0FFEEu + (uint32_t)cfg.dim + cfg.inter);
    const int rows = c.rows();

    // The e8m0 byte and the f32 scale must decode to the same number, or the two
    // arms are not being fed the same activation scale (ABI note, kernels.rs:159).
    for (size_t i = 0; i < c.acts.size(); ++i)
        if (ldexpf(1.f, (int)c.acts[i] - 127) != c.actf[i]) {
            printf("  [%s] FAIL: activation scale %zu is not a power of two "
                   "(e8m0 byte %u vs f32 %g)\n", cfg.name, i, c.acts[i], (double)c.actf[i]);
            return 1;
        }

    std::vector<double> gq, graw;
    golden(c, c.aq, gq);
    golden(c, c.xraw, graw);

    uint8_t *dw = nullptr, *dws = nullptr, *da = nullptr, *das = nullptr;
    float *dout_tc = nullptr, *dout_gv = nullptr;
    int* dids = nullptr;
    const size_t wb = c.w.size(), wsb = c.ws.size();
    cudaMalloc(&dw, wb); cudaMalloc(&dws, wsb);
    cudaMalloc(&da, c.act.size()); cudaMalloc(&das, c.acts.size());
    cudaMalloc(&dout_tc, (size_t)cfg.slots * rows * 4);
    cudaMalloc(&dout_gv, (size_t)cfg.slots * rows * 4);
    cudaMalloc(&dids, sizeof(int));
    cudaMemcpy(dw, c.w.data(), wb, cudaMemcpyHostToDevice);
    cudaMemcpy(dws, c.ws.data(), wsb, cudaMemcpyHostToDevice);
    cudaMemcpy(da, c.act.data(), c.act.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(das, c.acts.data(), c.acts.size(), cudaMemcpyHostToDevice);
    cudaMemset(dids, 0, sizeof(int));

    std::vector<float> vpoison((size_t)cfg.slots * rows, kPoison);
    std::vector<float> tc((size_t)cfg.slots * rows), gv((size_t)cfg.slots * rows);
    cudaMemcpy(dout_tc, vpoison.data(), vpoison.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dout_gv, vpoison.data(), vpoison.size() * 4, cudaMemcpyHostToDevice);

    // ---- ARM 1: the tcgen05 kernel, through the EXACT entry serve would call.
    // rc == 0 means either "ran" or "gate off / shape rejected" — the sentinel
    // check below is what separates them.
    const int rc_tc = dsv41_expert_tcgen05_gate_up_mxf4(
        dw, dws, da, das, dout_tc, (long)(2 * cfg.inter), cfg.inter, cfg.dim, cfg.limit,
        cfg.slots, 0);
    cudaError_t e_tc = cudaDeviceSynchronize();

    // ---- ARM 2: the proven SIMT GEMV, same nibbles, same e8m0 weight scales,
    // w1 = pool rows [0, inter), w3 = pool rows [inter, 2*inter).
    const int zero_id = 0;
    (void)zero_id;
    const int rc_gv = dsv41_expert_gate_up_fp4_batched(
        da, c.actf.data(), dout_gv, (long)(2 * cfg.inter), 1, cfg.dim, cfg.inter, cfg.limit,
        cfg.slots, dw, 0, dws, 0, dw + (size_t)cfg.inter * (cfg.dim / 2), 0,
        dws + (size_t)cfg.inter * (cfg.dim / 32), 0, dids, /*ilv=*/0, 0);
    cudaError_t e_gv = cudaDeviceSynchronize();

    cudaMemcpy(tc.data(), dout_tc, tc.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gv.data(), dout_gv, gv.size() * 4, cudaMemcpyDeviceToHost);

    int fails = 0;
    if (rc_tc != 0 || e_tc != cudaSuccess) {
        printf("  [%s] FAIL: tcgen05 rc=%d err=%s\n", cfg.name, rc_tc, cudaGetErrorString(e_tc));
        ++fails;
    }
    // Gate actually fired? (the launcher's silent no-op path)
    size_t poison_left = 0;
    for (size_t i = 0; i < tc.size(); ++i) if (same_bits(tc[i], kPoison)) ++poison_left;
    if (poison_left) {
        printf("  [%s] FAIL: tcgen05 did not write %zu/%zu outputs — the env gate is OFF or the "
               "launcher rejected the shape (set DSV41_EXPERT_TCGEN05_MXF4=1 / check the "
               "dim %% 128 contract)\n", cfg.name, poison_left, tc.size());
        ++fails;
    }
    if (rc_gv != 0 || e_gv != cudaSuccess) {
        printf("  [%s] FAIL: gemv ref rc=%d err=%s\n", cfg.name, rc_gv, cudaGetErrorString(e_gv));
        ++fails;
    }

    // slot uniformity: direct pool (no ids) => every grid.y slot bit-identical.
    int slot_bad = 0;
    for (int s = 1; s < cfg.slots; ++s)
        for (int r = 0; r < rows; ++r)
            if (!same_bits(tc[(size_t)r], tc[(size_t)s * rows + r])) ++slot_bad;

    const std::vector<double> gq0(gq.begin(), gq.begin() + rows);
    const std::vector<double> gr0(graw.begin(), graw.begin() + rows);
    std::vector<float> tc0(tc.begin(), tc.begin() + rows);
    std::vector<float> gv0(gv.begin(), gv.begin() + rows);
    const Diff d_q = cmp(tc0, gq0);       // vs quantised golden   (bar 1e-4)
    const Diff d_f = cmp(tc0, gr0);       // vs unquantised golden (bar 5e-2)
    const Diff d_g = cmp(tc0, gv0);       // vs the SIMT path      (bar 1e-3)
    // and the same three for the reference arm, so a failure can be localised.
    const Diff r_q = cmp(gv0, gq0);
    const Diff r_f = cmp(gv0, gr0);

    const bool ok = d_q.rel_l2 < 1e-4 && d_g.rel_l2 < 1e-3 && d_f.rel_l2 < 5e-2 &&
                    poison_left == 0 && rc_tc == 0 && rc_gv == 0 && slot_bad == 0;
    printf("  [%s]\n", cfg.name);
    printf("      rc=%d/%d err=%s   tc-vs-GOLDEN_Q relL2=%.2e (bar 1e-4)   "
           "tc-vs-GEMV relL2=%.2e (bar 1e-3)   tc-vs-GOLDEN_F32 relL2=%.2e (bar 5e-2)\n",
           rc_tc, rc_gv, cudaGetErrorString(e_tc), d_q.rel_l2, d_g.rel_l2, d_f.rel_l2);
    printf("      maxrel: q=%.2e gemv=%.2e f32=%.2e   maxabs f32=%.3e   "
           "gemv-vs-GOLDEN_Q relL2=%.2e  slot-mismatch=%d  %s\n",
           d_q.max_rel, d_g.max_rel, d_f.max_rel, d_f.max_abs, r_q.rel_l2, slot_bad,
           ok ? "PASS" : "FAIL");
    if (r_f.rel_l2 > 1.0) printf("      (reference arm deviates from GOLDEN_F32 by %.2e — that is "
                                 "the fp4 quantisation floor, not a bug)\n", r_f.rel_l2);
    if (!ok) ++fails;

    if (graph_mode == 1 && fails == 0) {
        // Capture + replay: the serve path captures the entire step, and a
        // process-level env gate is exactly what keeps this legal (plan §5).
        cudaStream_t s = nullptr;
        cudaStreamCreate(&s);
        std::vector<float> vp2 = vpoison;
        cudaMemcpy(dout_tc, vp2.data(), vp2.size() * 4, cudaMemcpyHostToDevice);
        cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        const int rc_cap = dsv41_expert_tcgen05_gate_up_mxf4(
            dw, dws, da, das, dout_tc, (long)(2 * cfg.inter), cfg.inter, cfg.dim, cfg.limit,
            cfg.slots, s);
        cudaGraph_t graph = nullptr;
        const cudaError_t e_end = cudaStreamEndCapture(s, &graph);
        cudaGraphExec_t inst = nullptr;
        cudaError_t e_inst = e_end == cudaSuccess ? cudaGraphInstantiate(&inst, graph, 0)
                                                  : e_end;
        cudaError_t e_rep = e_inst == cudaSuccess ? cudaGraphLaunch(inst, 0) : e_inst;
        e_rep = e_rep == cudaSuccess ? cudaDeviceSynchronize() : e_rep;
        std::vector<float> tc_rep(tc.size());
        cudaMemcpy(tc_rep.data(), dout_tc, tc_rep.size() * 4, cudaMemcpyDeviceToHost);
        const Diff d_rep = cmp_exact(tc, tc_rep);
        printf("      graph: capture rc=%d end=%s instantiate=%s launch=%s  bitdiff=%zu %s\n",
               rc_cap, cudaGetErrorString(e_end), cudaGetErrorString(e_inst),
               cudaGetErrorString(e_rep), d_rep.nbit, d_rep.nbit == 0 ? "REPLAY-EXACT" : "DIFFERS");
        if (graph) cudaGraphDestroy(graph);
        if (inst) cudaGraphExecDestroy(inst);
        cudaStreamDestroy(s);
    }

    cudaFree(dw); cudaFree(dws); cudaFree(da); cudaFree(das);
    cudaFree(dout_tc); cudaFree(dout_gv); cudaFree(dids);
    return fails;
}

// The default-OFF contract: with the gate unset the entry must return 0 AND
// leave the output untouched. This is the only way to tell "disabled" apart from
// "ran and produced zeros" — the failure mode the project calls its #1
// measurement-bias trap (an ON arm that silently measures the OLD path).
int run_gate_off() {
    const int dim = 128, inter = 64, rows = 2 * inter;
    const Case c = build_case(dim, inter, 0.f, 7u);
    uint8_t *dw = nullptr, *dws = nullptr, *da = nullptr, *das = nullptr;
    float* dout = nullptr;
    cudaMalloc(&dw, c.w.size()); cudaMalloc(&dws, c.ws.size());
    cudaMalloc(&da, c.act.size()); cudaMalloc(&das, c.acts.size());
    cudaMalloc(&dout, (size_t)rows * 4);
    cudaMemcpy(dw, c.w.data(), c.w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dws, c.ws.data(), c.ws.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(da, c.act.data(), c.act.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(das, c.acts.data(), c.acts.size(), cudaMemcpyHostToDevice);
    std::vector<float> vp((size_t)rows, kPoison), got((size_t)rows);
    cudaMemcpy(dout, vp.data(), vp.size() * 4, cudaMemcpyHostToDevice);
    const int rc = dsv41_expert_tcgen05_gate_up_mxf4(dw, dws, da, das, dout, rows, inter, dim,
                                                    0.f, 1, 0);
    const cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    size_t intact = 0;
    for (size_t i = 0; i < got.size(); ++i) if (same_bits(got[i], kPoison)) ++intact;
    printf("  [gate-off] rc=%d err=%s  outputs untouched=%zu/%zu  %s\n", rc,
           cudaGetErrorString(e), intact, got.size(),
           (rc == 0 && e == cudaSuccess && intact == got.size()) ? "PASS (real no-op)"
                                                                 : "FAIL (the OFF arm is not inert!)");
    cudaFree(dw); cudaFree(dws); cudaFree(da); cudaFree(das); cudaFree(dout);
    return (rc == 0 && e == cudaSuccess && intact == got.size()) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int graph_mode = 0;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "--gate-off") == 0) return run_gate_off();  // MUST NOT set the gate below
        if (strcmp(a, "--graph") == 0) graph_mode = 1;
        else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            printf("usage: %s [--graph] | --gate-off\n", argv[0]);
            return 0;
        }
        else { printf("unknown arg: %s (try --help)\n", a); return 2; }
    }

    // BEFORE the first call to either entry — both gates are process statics.
    setenv("DSV41_EXPERT_TCGEN05_MXF4", "1", 1);  // arm the .so side of the tcgen05 gate
    setenv("DSV41_GATEUP_FUSE", "0", 1);          // reference arm = UNFUSED [2*inter]

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("== tcgen05 mxf4 swapAB gate/up — GPU parity suite ==\n");
    printf("   device %d: %s (sm_%d%d)  compute capability %d.%d\n", dev, prop.name,
           prop.major, prop.minor, prop.major, prop.minor);
    if (prop.major != 10 && prop.major != 11) {
        printf("   WARNING: the tcgen05 path targets sm_103a (B300); this GPU is sm_%d%d — "
               "expect a launch failure, not a numerics failure\n", prop.major, prop.minor);
    }
    printf("   gate: DSV41_EXPERT_TCGEN05_MXF4=1  reference: DSV41_GATEUP_FUSE=0 (unfused)\n");

    int fails = 0;
    for (const Cfg& cfg : kCases) fails += run_case(cfg, graph_mode);

    printf(fails ? "RESULT: %d case(s) FAILED\n" : "RESULT: all cases PASS\n", fails);
    return fails ? 1 : 0;
}
