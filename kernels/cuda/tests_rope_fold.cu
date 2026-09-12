// tests_rope_fold.cu — the bit-parity acceptance test for the ROW-FOLD rope
// (`dsv41_apply_rope_mrows`, kernels/cuda/dsv41_kernels.cu).
//
// WHY THIS FILE EXISTS. `attention_rows` (chain_dev.rs) issues `apply_rope`
// once per verify row — the q rope and the inverse o rope, m launches each per
// layer — because every row owns a DIFFERENT position (`pos_base + r`). The
// row-fold replaces those m launches with ONE launch whose kernel loops r
// ascending and reads the position from the device array `pos_rows[r]` (the
// array `attention_rows` already uploads for `ring_append`/`window_idxs`). The
// fold is legal only if
//
//     the fused launch  ==  the m per-row launches, BIT FOR BIT
//
// for every row — the verify's iron rule ("row r of an m-row launch == the
// single-row decode of row r"). This suite pins exactly that:
//
//  1. BIT-IDENTITY (raw f32 bits, uint32 compare — ±0 and NaN payloads count as
//     differences) of the WHOLE buffer after the fused call vs the same buffer
//     after m `dsv41_apply_rope` calls issued with the production argument
//     order (`base = pos_ctr, mul = 1, off = r, step = 0`).
//  2. THE POSITION ARRAY IS READ PER ROW: an arm gives row r the position
//     `pos_base + 3r` (the reference passes `off = 3r` to the per-row form), so
//     a kernel that roped every row at `pos_base`, or reused row 0's position,
//     cannot pass.
//  3. NO WRITE OUTSIDE THE ROW BLOCK: the buffers are allocated with a canary
//     past the last row and with the fragmented interior of the real layout
//     (`row_stride > rows * row_len`: this rank only writes its leading
//     `nlh*hd` of each `nh*hd`-wide row) — compared WHOLE.
//  4. m = 1..6, several (rows, row_len, rope_dim), both the forward (q) and the
//     inverse (o) flag, early positions (base 0/1) and steady-state ones.
//
// The kernel's own header carries the instruction-level argument (same table
// index `t*half + i`, same `x0/x1` pair, same rotation expression as
// `apply_rope_kernel`); this suite is the empirical half.
//
// NOT checked here: the Rust wiring (the `DSV41_ROW_FOLD_ROPE` gate), end-to-end
// model parity, and launch performance.
//
// Build (needs nvcc, NO GPU — the TU #includes dsv41_kernels.cu, exactly how
// tests_dsv41_attn.cu builds, so dsv41_kernels.cu stays a single input file):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math \
//        -std=c++17 -o /tmp/t_rope_fold kernels/cuda/tests_rope_fold.cu
// Run (needs ONE free GPU):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_rope_fold
#include "dsv41_kernels.cu"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_fails = 0;

uint32_t g_rng = 0x9E3779B9u;
uint32_t xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
float frand() { return (float)((int)(xr() % 4001) - 2000) * 1.0e-3f; }

std::vector<uint32_t> bits(const float* h, size_t n) {
    std::vector<uint32_t> v(n);
    memcpy(v.data(), h, n * sizeof(float));
    return v;
}

// One (m, rows, row_stride, row_len, rope_dim, inverse, position) case.
//
// `pos_mul` generalises the production position: `pos_rows[r] = pos_base +
// r*pos_mul` and the reference's per-row `off = r*pos_mul` (production is
// `pos_mul == 1`, and then `off = r`). Anything but 1 proves the kernel reads a
// PER-ROW position.
void case_rope(const char* tag, int m, int rows, int row_stride, int row_len, int rope_dim,
               int inverse, int pos_base, int pos_mul) {
    const int half = rope_dim / 2;
    const size_t n = (size_t)m * (size_t)row_stride;   // the roped row block
    const size_t canary = 64;                          // past the last row
    const size_t total = n + canary;
    const int max_pos = pos_base + (m - 1) * pos_mul;
    const size_t tlen = (size_t)(max_pos + 2) * (size_t)half;   // cos/sin tables

    float *x_ref = nullptr, *x_fus = nullptr, *cos = nullptr, *sin = nullptr;
    int *pos_ctr = nullptr, *pos_rows = nullptr;
    cudaMalloc(&x_ref, total * sizeof(float));
    cudaMalloc(&x_fus, total * sizeof(float));
    cudaMalloc(&cos, tlen * sizeof(float));
    cudaMalloc(&sin, tlen * sizeof(float));
    cudaMalloc(&pos_ctr, sizeof(int));
    cudaMalloc(&pos_rows, (size_t)m * sizeof(int));

    std::vector<float> h(total), hc(tlen), hs(tlen);
    for (size_t i = 0; i < total; ++i) h[i] = frand();
    for (size_t i = 0; i < tlen; ++i) {
        hc[i] = frand();
        hs[i] = frand();
    }
    std::vector<int> hr(m);
    for (int r = 0; r < m; ++r) hr[r] = pos_base + r * pos_mul;

    cudaMemcpy(x_ref, h.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_fus, h.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cos, hc.data(), tlen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sin, hs.data(), tlen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_ctr, &pos_base, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_rows, hr.data(), (size_t)m * sizeof(int), cudaMemcpyHostToDevice);

    // ---- arm A: the m per-row launches (the production call shape) ----
    for (int r = 0; r < m; ++r) {
        const int rc = dsv41_apply_rope(
            x_ref + (size_t)r * (size_t)row_stride, cos, sin, rows, row_len, rope_dim, half,
            pos_ctr, 1, r * pos_mul, 0, inverse, /*stream=*/0);
        if (rc != 0) {
            printf("    FAIL %s: per-row launch r=%d rc=%d\n", tag, r, rc);
            ++g_fails;
        }
    }
    // ---- arm B: the fused launch ----
    const int rc = dsv41_apply_rope_mrows(x_fus, cos, sin, m, rows, row_stride, row_len,
                                          rope_dim, half, pos_rows, inverse, /*stream=*/0);
    if (rc != 0) {
        printf("    FAIL %s: fused launch rc=%d\n", tag, rc);
        ++g_fails;
    }

    std::vector<float> a(total), b(total);
    cudaMemcpy(a.data(), x_ref, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), x_fus, total * sizeof(float), cudaMemcpyDeviceToHost);

    const std::vector<uint32_t> ba = bits(a.data(), total), bb = bits(b.data(), total);
    if (ba == bb) {
        printf("  ok   %-28s m=%d rows=%d stride=%d len=%d rd=%d inv=%d base=%d mul=%d\n", tag, m,
               rows, row_stride, row_len, rope_dim, inverse, pos_base, pos_mul);
    } else {
        ++g_fails;
        printf("  FAIL %-28s m=%d rows=%d stride=%d len=%d rd=%d inv=%d base=%d mul=%d\n", tag, m,
               rows, row_stride, row_len, rope_dim, inverse, pos_base, pos_mul);
        int shown = 0;
        for (size_t i = 0; i < total && shown < 8; ++i) {
            if (ba[i] != bb[i]) {
                printf("       i=%zu (row %d head %d col %zu) ref=%g/%08x fused=%g/%08x\n", i,
                       (int)(i / (size_t)row_stride),
                       (int)((i % (size_t)row_stride) / (size_t)row_len),
                       i % (size_t)row_len, a[i], ba[i], b[i], bb[i]);
                ++shown;
            }
        }
    }

    cudaFree(x_ref);
    cudaFree(x_fus);
    cudaFree(cos);
    cudaFree(sin);
    cudaFree(pos_ctr);
    cudaFree(pos_rows);
}

}  // namespace

int main() {
    printf("dsv41 apply_rope_mrows row-fold self-test (fused == m per-row launches)\n");
    // the production shape: 6 verify rows; this rank owns nlh = 8 of nh = 64
    // heads of head_dim 512, so the row pitch is nh*hd = 32768 while only
    // nlh*hd = 4096 is written; rope_head_dim 64 -> half 32.
    case_rope("production q", 6, 8, 32768, 512, 64, 0, 4096, 1);
    case_rope("production o (inverse)", 6, 8, 32768, 512, 64, 1, 4096, 1);
    // the per-row position must be read per row (not once for the block)
    case_rope("positions differ", 6, 8, 32768, 512, 64, 0, 4096, 3);
    case_rope("positions differ (inv)", 6, 8, 32768, 512, 64, 1, 4096, 3);
    // the sequence's very first positions (base 0 / 1)
    case_rope("start of sequence", 5, 4, 1024, 128, 64, 0, 0, 1);
    case_rope("start of sequence (inv)", 5, 4, 1024, 128, 64, 1, 1, 1);
    // every m the verify can pass (VERIFY_ROWS = 6)
    for (int m = 1; m <= 6; ++m) {
        char tag[64];
        snprintf(tag, sizeof(tag), "m=%d small", m);
        case_rope(tag, m, 3, 384, 128, 32, 0, 7, 1);
        snprintf(tag, sizeof(tag), "m=%d one head", m);
        case_rope(tag, m, 1, 64, 64, 64, 1, 100000, 1);
    }
    // rope_dim == row_len (the whole row rotates) and a narrow tail
    case_rope("full-row rope", 3, 2, 512, 128, 128, 0, 33, 1);
    case_rope("narrow tail (inv)", 3, 2, 512, 128, 16, 1, 33, 1);

    if (g_fails == 0) {
        printf("all cases passed\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fails);
    return 1;
}
