// tests_rmsnorm_rope_mrows.cu — the bit-parity acceptance test for B4
// (`dsv41_rmsnorm_rope_mrows`, DSV41_RMSNORM_ROPE_MROWS, default OFF).
//
// WHY THIS FILE EXISTS. `attention_rows` (chain_dev.rs) prepares the kv block's
// `[m, hd]` rows with TWO launches:
//     norm_rows_on(kv_r, kv_norm, kv_r, m, hd, eps)      -> dsv41_rmsnorm_rows
//     apply_rope_on(kv_r, cos, sin, m, hd, rd, half, pos_ctr, 1, 0, 1, false)
//                                                        -> dsv41_apply_rope
// B4 folds them into ONE launch: phase 1 is `dsv41_rmsnorm_rows_kernel`'s body
// verbatim (same 1024-thread block, so the cross-warp fold is the same sum in
// the same order), phase 2 is `apply_rope_kernel`'s trailing-`2*half` rotation
// at `pos_rows[row]` — no reduction, so its thread mapping cannot move a value.
// The parity claim is
//
//     ONE fused launch  ==  rmsnorm_rows + apply_rope, BIT FOR BIT, whole buffer,
//
// which this suite pins, and it pins the THREE things that could break it:
//
//  1. THE GEOMETRY OF PHASE 1: a fused kernel launched at a different blockDim
//     would fold `red[32]` in a different order. The comparison is on the whole
//     [m, dim] buffer with raw f32 bits, and the output is qNaN-sentinel-filled
//     so a kernel that wrote only part of a row cannot tie by accident.
//  2. THE POSITION: the reference computes the RoPE table index as
//     `t = pos_base*mul + off + row*step` (the call site's mul=1, off=0, step=1);
//     the fused kernel reads `pos_rows[row]`. The random `pos_base` here is not
//     0, and the sweep covers rows whose index differs from the row number, so a
//     kernel that used `row` (or a fixed position) as its phase would go red.
//  3. THE ROPE REGION: `rope_off = dim - 2*half` (the trailing columns, not the
//     leading ones) — a fused kernel that rotated columns [0, 2*half) instead
//     would mismatch every element of the complementary region.
//
// It also covers `inverse = 1` (the flag `apply_rope_kernel` scales the sine
// by) even though the kv call site passes 0: the fused kernel takes the flag,
// so the arm that is not exercised in production is the one most likely to rot.
//
// Build (needs nvcc, NO GPU; both entries live in dsv41_kernels.cu):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
//        -o /tmp/t_rmsnorm_rope_mrows kernels/cuda/tests_rmsnorm_rope_mrows.cu \
//        kernels/cuda/dsv41_kernels.cu
// Run (needs ONE free GPU):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_rmsnorm_rope_mrows
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The two-launch reference pair (dsv41_kernels.cu).
extern "C" int dsv41_rmsnorm_rows(const float* x, const float* w, float* out, int rows, int dim,
                                  float eps, cudaStream_t s);
extern "C" int dsv41_apply_rope(float* x, const float* cos, const float* sin, int rows,
                                int row_len, int dim, int half, const int* base, int mul, int off,
                                int step, int inverse, cudaStream_t s);
// B4 under test.
extern "C" int dsv41_rmsnorm_rope_mrows(const float* x, const float* w, float* out, int rows,
                                        int dim, float eps, const float* cos, const float* sin,
                                        int rope_off, int half, const int* pos_rows, int inverse,
                                        cudaStream_t s);

namespace {

int g_fails = 0;

uint32_t g_rng = 0x2545F491u;
uint32_t xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
float frand() { return (float)((int)(xr() % 4001) - 2000) * 1.0e-3f; }

void case_norm_rope(int rows, int dim, int rd, int pos_base, int inverse) {
    const int half = rd / 2;
    const int rope_off = dim - rd;   // the trailing region, as the call site
    const size_t xn = (size_t)rows * (size_t)dim;
    const int max_t = pos_base + rows + 2;
    const size_t cn = (size_t)max_t * (size_t)half;

    std::vector<float> hx(xn), hw(dim), hcos(cn), hsin(cn);
    for (size_t i = 0; i < xn; ++i) hx[i] = frand();
    for (int i = 0; i < dim; ++i) hw[i] = 1.f + 0.01f * frand();
    for (size_t i = 0; i < cn; ++i) hcos[i] = 0.9f + 0.1f * frand();
    for (size_t i = 0; i < cn; ++i) hsin[i] = 0.9f + 0.1f * frand();

    // qNaN poison: every element must be written by BOTH arms.
    std::vector<float> poison(xn);
    for (size_t i = 0; i < xn; ++i) {
        const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
        memcpy(&poison[i], &bits, 4);
    }
    std::vector<int32_t> hpos(rows);
    for (int r = 0; r < rows; ++r) hpos[r] = pos_base + r;

    float *dx = nullptr, *dw = nullptr, *dc = nullptr, *ds = nullptr;
    float *out_a = nullptr, *out_b = nullptr;
    int *dbase = nullptr, *dpos = nullptr;
    cudaMalloc(&dx, xn * sizeof(float));
    cudaMalloc(&dw, (size_t)dim * sizeof(float));
    cudaMalloc(&dc, cn * sizeof(float));
    cudaMalloc(&ds, cn * sizeof(float));
    cudaMalloc(&out_a, xn * sizeof(float));
    cudaMalloc(&out_b, xn * sizeof(float));
    cudaMalloc(&dbase, sizeof(int));
    cudaMalloc(&dpos, (size_t)rows * sizeof(int));
    cudaMemcpy(dx, hx.data(), xn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, hw.data(), (size_t)dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hcos.data(), cn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hsin.data(), cn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_a, poison.data(), xn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_b, poison.data(), xn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dbase, &pos_base, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, hpos.data(), (size_t)rows * sizeof(int), cudaMemcpyHostToDevice);

    // ---- arm A: the two-launch reference ----
    const int rc_n = dsv41_rmsnorm_rows(dx, dw, out_a, rows, dim, /*eps=*/1.0e-6f, /*stream=*/0);
    if (rc_n != 0) {
        printf("    FAIL rmsnorm_rows rc=%d\n", rc_n);
        ++g_fails;
    }
    // rows = `rows`, row_len = dim, dim = rd, the call site's mul=1/off=0/step=1.
    const int rc_r = dsv41_apply_rope(out_a, dc, ds, rows, dim, rd, half, dbase, /*mul=*/1,
                                      /*off=*/0, /*step=*/1, inverse, /*stream=*/0);
    if (rc_r != 0) {
        printf("    FAIL apply_rope rc=%d\n", rc_r);
        ++g_fails;
    }

    // ---- arm B: ONE fused launch (B4) ----
    const int rc_f =
        dsv41_rmsnorm_rope_mrows(dx, dw, out_b, rows, dim, /*eps=*/1.0e-6f, dc, ds, rope_off, half,
                                 dpos, inverse, /*stream=*/0);
    if (rc_f != 0) {
        printf("    FAIL rmsnorm_rope_mrows rc=%d\n", rc_f);
        ++g_fails;
    }

    std::vector<float> a(xn), b(xn);
    cudaMemcpy(a.data(), out_a, xn * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), out_b, xn * sizeof(float), cudaMemcpyDeviceToHost);

    const bool eq = memcmp(a.data(), b.data(), xn * sizeof(float)) == 0;
    const bool covered = [&] {
        for (size_t i = 0; i < xn; ++i) {
            const uint32_t bits = 0x7FC00000u | (uint32_t)(i & 0x3FFu);
            uint32_t got;
            memcpy(&got, &b[i], 4);
            if (got == bits) return false;   // sentinel survived = element not written
        }
        return true;
    }();

    if (eq && covered) {
        printf("  ok   rows=%d dim=%d rd=%d rope_off=%d pos_base=%d inverse=%d\n", rows, dim, rd,
               rope_off, pos_base, inverse);
    } else {
        ++g_fails;
        printf("  FAIL rows=%d dim=%d rd=%d rope_off=%d pos_base=%d inverse=%d eq=%d covered=%d\n",
               rows, dim, rd, rope_off, pos_base, inverse, (int)eq, (int)covered);
        int shown = 0;
        for (size_t i = 0; i < xn && shown < 8; ++i) {
            uint32_t ua, ub;
            memcpy(&ua, &a[i], 4);
            memcpy(&ub, &b[i], 4);
            if (ua != ub) {
                printf("       i=%zu (row %zu col %zu) ref=%g/%08x fused=%g/%08x\n", i, i / dim,
                       i % dim, a[i], ua, b[i], ub);
                ++shown;
            }
        }
    }

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dc);
    cudaFree(ds);
    cudaFree(out_a);
    cudaFree(out_b);
    cudaFree(dbase);
    cudaFree(dpos);
}

}  // namespace

int main() {
    printf("B4 rmsnorm+rope mrows parity (fused == rmsnorm_rows + apply_rope, bit for bit)\n");
    // The verify shape: m rows of hd = 512, rope_head_dim = 64 (the trailing
    // region), a non-zero position base.
    case_norm_rope(6, 512, 64, 0, 0);
    case_norm_rope(6, 512, 64, 37, 0);
    case_norm_rope(5, 512, 64, 1000, 0);
    case_norm_rope(2, 512, 64, 1, 0);
    // the flag the kv call site does not pass (the arm most likely to rot)
    case_norm_rope(6, 512, 64, 37, 1);
    // other head widths: the dim-not-a-multiple-of-1024 stride path
    case_norm_rope(6, 128, 64, 37, 0);
    case_norm_rope(6, 576, 128, 37, 0);
    case_norm_rope(6, 1024, 64, 37, 0);

    if (g_fails == 0) {
        printf("all cases passed\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fails);
    return 1;
}
