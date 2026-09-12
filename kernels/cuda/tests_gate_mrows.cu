// tests_gate_mrows.cu — the bit-parity acceptance test for the MoE gate's
// ROW-FOLD: `Device::gemv_bf16_v2_mrows` -> `ferrite_gemv_bf16_v2_mrows` /
// `Device::gemv_bf16_mrows` -> `ferrite_gemv_bf16_nt`, the m-activation-row form
// of the single-row gate GEMV.
//
// WHY THIS FILE EXISTS. `moe_rows` (chain_dev.rs) runs the bf16 gate
// `gemv_bf16` once per activation row r. For the gate's shape (n = n_routed <
// `GEMV_V2_MAX_N` = 2048) `gemv_bf16` dispatches to
// `ferrite_gemv_bf16_v2(..., nrows = 1)` — i.e. `gemv_bf16_v2_kernel<WPR>` with
// `WPR = gv2_wpr(n)`, a row K-SPLIT across WPR warps whose partials are folded
// in shared memory (`sum += part[(warp/WPR)*WPR + j]`, j ascending). The row
// fold replaces the m calls with ONE multi-row call (arm B / arm C below). The
// parity claim is
//
//     ONE multi-row call (nrows = m)  ==  m v2 calls (nrows = 1), BIT FOR BIT,
//
// per (token, output row). `tests_dsv41_head_mrows.cu` makes the same claim for
// the head — but the head is WPR == 1 (`out_f >= 16384`), so the smem partial
// fold never runs there. The GATE lives at WPR = 8, which is exactly the case
// that is NOT covered by that suite, and it is what this one pins:
//
//  1. BIT-IDENTITY vs the production single-row launch: an m-row nt call is
//     bit-identical (raw f32 bits, uint32 compare) to m `ferrite_gemv_bf16_v2`
//     calls with the SAME argument order device.rs's `gemv_bf16` uses
//     (`x, w, bias = null, out, in_f = k, out_f = n, nrows = 1`).
//  2. FULL COVERAGE + ROW INDEPENDENCE: the outputs are pre-filled with a qNaN
//     sentinel and compared WHOLE, so a kernel that folded only some rows (or
//     reused row 0's accumulator) cannot pass.
//  3. BOTH WPR domains the fold can meet below `GEMV_V2_MAX_N`: WPR = 8
//     (n < 1024), WPR = 4 (1024 <= n < 4096 — still below the Rust bound), and a
//     k that is NOT a multiple of `32*8*WPR` so each warp takes a partial
//     K-slice (the `kper` rounding path).
//  4. BOTH multi-row ENTRIES: the historical `ferrite_gemv_bf16_nt` (arm B) AND
//     the gate's own `ferrite_gemv_bf16_v2_mrows` (arm C, `DSV41_GATE_MROWS` —
//     the symbol `moe_rows` now calls, which delegates to the same program
//     rather than re-typing v2's body). Each is compared to arm A on the whole
//     buffer with its own coverage sweep, so an entry that first grew a
//     divergent body would fail arm C while arm B stayed green.
//
// ⚠️ NOT `head_gemv_bf16_mrows`: that kernel is the WPR == 1 program only, so it
// cannot reproduce the gate's K-split. The fold deliberately routes through the
// multi-row v2 program, which carries v2's exact program including the fold.
//
// Build (needs nvcc, NO GPU; the reference entries live in the OTHER TU
// (ferrite_kernels.cu), which is linked as a second input file — exactly how
// tests_dsv41_head_mrows.cu builds, so neither TU is #included):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math \
//        -std=c++17 -o /tmp/t_gate_mrows kernels/cuda/tests_gate_mrows.cu \
//        kernels/cuda/ferrite_kernels.cu
// Run (needs ONE free GPU):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_gate_mrows
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The two entries under test, from ferrite_kernels.cu (linked as a second TU).
extern "C" cudaError_t ferrite_gemv_bf16_v2(const float* x, const void* w, const float* bias,
                                            float* out, int in_f, int out_f, int nrows,
                                            cudaStream_t s);
extern "C" cudaError_t ferrite_gemv_bf16_nt(const float* x, const void* w, const float* bias,
                                            float* out, int in_f, int out_f, int nrows,
                                            cudaStream_t s);
// The gate's own multi-row entry (`DSV41_GATE_MROWS`): a separate symbol in the
// same TU, delegating to the ONE multi-row v2 program (`ferrite_gemv_bf16_nt` /
// `gemv_bf16_nt_kernel<NT, WPR>`) with the plan's name and its 1..=8 bound. It
// is a third ARM here, not a third program: the test pins that its output is
// bit-identical to arm A too, so a future rewrite of the entry that stops
// delegating (a re-typed body — the FOLD failure mode) cannot pass silently.
extern "C" cudaError_t ferrite_gemv_bf16_v2_mrows(const float* x, const void* w,
                                                  const float* bias, float* out, int in_f,
                                                  int out_f, int nrows, cudaStream_t s);

namespace {

int g_fails = 0;

uint32_t g_rng = 0x12345678u;
uint32_t xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
float frand() { return (float)((int)(xr() % 4001) - 2000) * 1.0e-3f; }

// ARM: one m-row nt call vs m single-row v2 calls, bit for bit, whole buffer.
void case_gate(int m, int n, int k) {
    const size_t xn = (size_t)m * (size_t)k;
    const size_t on = (size_t)m * (size_t)n;

    // bf16 weights [n, k]
    std::vector<__nv_bfloat16> hw((size_t)n * (size_t)k);
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = __float2bfloat16(frand());
    // activations [m, k]
    std::vector<float> hx(xn);
    for (size_t i = 0; i < xn; ++i) hx[i] = frand();
    // the qNaN sentinel: any element the kernel does not write stays a NaN and
    // the two arms cannot match by accident
    std::vector<float> poison(on);
    for (size_t i = 0; i < on; ++i) {
        const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
        memcpy(&poison[i], &bits, 4);
    }

    __nv_bfloat16* dw = nullptr;
    float *dx = nullptr, *out_a = nullptr, *out_b = nullptr, *out_c = nullptr;
    cudaMalloc(&dw, hw.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&dx, xn * sizeof(float));
    cudaMalloc(&out_a, on * sizeof(float));
    cudaMalloc(&out_b, on * sizeof(float));
    cudaMalloc(&out_c, on * sizeof(float));
    cudaMemcpy(dw, hw.data(), hw.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx.data(), xn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_a, poison.data(), on * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_b, poison.data(), on * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_c, poison.data(), on * sizeof(float), cudaMemcpyHostToDevice);

    // ---- arm A: m single-row v2 launches (the production call shape) ----
    for (int r = 0; r < m; ++r) {
        const cudaError_t rc = ferrite_gemv_bf16_v2(
            dx + (size_t)r * (size_t)k, dw, /*bias=*/nullptr, out_a + (size_t)r * (size_t)n, k, n,
            /*nrows=*/1, /*stream=*/0);
        if (rc != cudaSuccess) {
            printf("    FAIL v2 r=%d rc=%d\n", r, (int)rc);
            ++g_fails;
        }
    }
    // ---- arm B: ONE m-row nt launch (the historical fold entry) ----
    const cudaError_t rc =
        ferrite_gemv_bf16_nt(dx, dw, /*bias=*/nullptr, out_b, k, n, m, /*stream=*/0);
    if (rc != cudaSuccess) {
        printf("    FAIL nt rc=%d\n", (int)rc);
        ++g_fails;
    }
    // ---- arm C: ONE m-row `ferrite_gemv_bf16_v2_mrows` launch — the gate's own
    // entry (`DSV41_GATE_MROWS`), the symbol `moe_rows` now calls. Same ABI, same
    // program, so the SAME whole-buffer bit-identity against arm A is required:
    // if this entry ever stops delegating to the multi-row v2 program and grows a
    // re-typed body (the FOLD failure mode), arm C goes red while arm B stays
    // green — which is exactly the drift this arm exists to catch.
    const cudaError_t rc_c =
        ferrite_gemv_bf16_v2_mrows(dx, dw, /*bias=*/nullptr, out_c, k, n, m, /*stream=*/0);
    if (rc_c != cudaSuccess) {
        printf("    FAIL v2_mrows rc=%d\n", (int)rc_c);
        ++g_fails;
    }

    std::vector<float> a(on), b(on), c(on);
    cudaMemcpy(a.data(), out_a, on * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), out_b, on * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c.data(), out_c, on * sizeof(float), cudaMemcpyDeviceToHost);
    const bool eq = memcmp(a.data(), b.data(), on * sizeof(float)) == 0;
    const bool eq_c = memcmp(a.data(), c.data(), on * sizeof(float)) == 0;
    const bool covered = [&] {
        for (size_t i = 0; i < on; ++i) {
            const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
            uint32_t got;
            memcpy(&got, &b[i], 4);
            if (got == bits) return false;   // sentinel survived = row not written
        }
        return true;
    }();
    // arm C gets its OWN coverage sweep: an entry that wrote only some rows could
    // still tie arm A on the rows it did write.
    const bool covered_c = [&] {
        for (size_t i = 0; i < on; ++i) {
            const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
            uint32_t got;
            memcpy(&got, &c[i], 4);
            if (got == bits) return false;
        }
        return true;
    }();

    if (eq && eq_c && covered && covered_c) {
        printf("  ok   m=%d n=%d k=%d (WPR=%d)\n", m, n, k, n >= 1024 ? 4 : 8);
    } else {
        ++g_fails;
        printf("  FAIL m=%d n=%d k=%d (WPR=%d) eq=%d eq_c=%d covered=%d covered_c=%d\n", m, n, k,
               n >= 1024 ? 4 : 8, (int)eq, (int)eq_c, (int)covered, (int)covered_c);
        int shown = 0;
        for (size_t i = 0; i < on && shown < 8; ++i) {
            uint32_t ua, ub, uc;
            memcpy(&ua, &a[i], 4);
            memcpy(&ub, &b[i], 4);
            memcpy(&uc, &c[i], 4);
            if (ua != ub || ua != uc) {
                printf("       i=%zu (row %zu col %zu) v2=%g/%08x nt=%g/%08x v2_mrows=%g/%08x\n",
                       i, i / n, i % n, a[i], ua, b[i], ub, c[i], uc);
                ++shown;
            }
        }
    }

    cudaFree(dw);
    cudaFree(dx);
    cudaFree(out_a);
    cudaFree(out_b);
    cudaFree(out_c);
}

}  // namespace

int main() {
    printf("dsv41 gate row-fold self-test (v2_mrows/nt(nrows=m) == m x v2(nrows=1))\n");
    // the production MoE gate: n = the routed expert count (WPR = 8), k = dim
    case_gate(6, 384, 5120);
    case_gate(5, 384, 5120);
    case_gate(2, 384, 5120);
    // another small-n expert count, same domain
    case_gate(6, 256, 5120);
    case_gate(6, 512, 5120);
    // WPR = 4: still below the Rust fold bound (n < 2048)
    case_gate(6, 1024, 5120);
    case_gate(6, 1200, 5120);
    // k NOT a multiple of 32*8*WPR: each warp takes a partial K-slice, so the
    // `kper` rounding + the short final pass of BOTH programs is exercised
    case_gate(6, 384, 520);
    case_gate(6, 384, 64);
    case_gate(6, 1024, 520);

    if (g_fails == 0) {
        printf("all cases passed\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fails);
    return 1;
}
