// tests_dsv41_glue.cu — numerical self-test for the five glue kernels in
// dsv41_glue.cu: engram_apply / swiglu_limit / gather_rows / scatter_add_rows /
// hc_collapse. The CPU references mirror crates/ferrite-dsv41/src/ops.rs
// (engram_forward, the expert swiglu epilogue, hc_pre) line by line; gather /
// scatter follow the index-copy / accumulate semantics of the ABI.
//
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//            tests_dsv41_glue.cu -o /tmp/t_glue
// Run:   CUDA_VISIBLE_DEVICES=0 ./t_glue
#include "dsv41_glue.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

#define CHECK(expr)                                                                     \
    do {                                                                                \
        cudaError_t _e = (expr);                                                        \
        if (_e != cudaSuccess) {                                                        \
            printf("  CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__,      \
                   __LINE__);                                                           \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

uint32_t rng = 12345;
uint32_t xr() { rng = rng * 1664525u + 1013904223u; return rng; }
float frand() { return (float)((int)(xr() % 2001) - 1000) / 1000.f; }

float maxdiff_of(const std::vector<float>& a, const std::vector<float>& b) {
    float md = 0.f;
    for (size_t i = 0; i < a.size(); ++i) md = std::fmax(md, std::fabs(a[i] - b[i]));
    return md;
}

// --------------------------------------------------------------------------
// reference: ops.rs::engram_forward (dots_out receives the pre-clamp scaled
// dots so the case can assert both signs occur)
// --------------------------------------------------------------------------
std::vector<float> ref_engram(const std::vector<float>& x0, const std::vector<float>& kv,
                              const std::vector<float>& qw, const std::vector<float>& kw, int rows,
                              int hc, int dim, float eps, const std::vector<uint8_t>* mask,
                              std::vector<float>* dots_out) {
    const float clamp_value = 1e-6f;
    const int span = hc * dim + dim;
    std::vector<float> out = x0;
    if (dots_out) dots_out->assign((size_t)rows * hc, 0.f);
    for (int r = 0; r < rows; ++r) {
        const float* key = &kv[(size_t)r * span];
        const float* value = &kv[(size_t)r * span + hc * dim];
        for (int i = 0; i < hc; ++i) {
            const float* h = &x0[((size_t)r * hc + i) * dim];
            const float* k = &key[i * dim];
            float hss = 0.f, kss = 0.f;
            for (int c = 0; c < dim; ++c) {
                hss += h[c] * h[c];
                kss += k[c] * k[c];
            }
            const float rstd = (1.f / std::sqrt(hss / dim + eps)) *
                               (1.f / std::sqrt(kss / dim + eps));
            float dot = 0.f;
            for (int c = 0; c < dim; ++c) dot += h[c] * qw[i * dim + c] * kw[i * dim + c] * k[c];
            dot *= rstd * std::pow((float)dim, -0.5f);
            if (dots_out) (*dots_out)[r * hc + i] = dot;
            float mag = std::sqrt(std::fmax(std::fabs(dot), clamp_value));
            mag = std::copysign(mag, dot);
            float gate = 1.f / (1.f + std::exp(-mag));
            if (mask && !(*mask)[r]) gate = 0.f;
            for (int c = 0; c < dim; ++c)
                out[((size_t)r * hc + i) * dim + c] = h[c] + gate * value[c];
        }
    }
    return out;
}

// --------------------------------------------------------------------------
// engram_apply case: mask on/off, crafted dot signs + the clamp path
// --------------------------------------------------------------------------
int run_engram_case(int rows, int hc, int dim, float eps, uint32_t seed) {
    rng = seed;
    const size_t span = (size_t)hc * dim + dim;
    const size_t xn = (size_t)rows * hc * dim;
    const size_t kvn = (size_t)rows * span;
    std::vector<float> x(xn), kv(kvn), qw((size_t)hc * dim), kw((size_t)hc * dim);
    for (auto& v : x) v = frand();
    for (auto& v : kv) v = frand();
    for (auto& v : qw) v = frand();
    for (auto& v : kw) v = frand();

    // crafted slots (rows = 4, hc = 2):
    //   (r0,i0): h = 0 -> dot = +0 exactly -> the max(|dot|,1e-6) clamp fires.
    //   (r0,i1): negative dot  (qw = kw = 1, k = -0.5, h = +1).
    //   (r2,i0): positive dot  (same, k = +0.5).
    for (int c = 0; c < dim; ++c) x[c] = 0.f;
    for (int c = 0; c < dim; ++c) {
        x[(size_t)dim + c] = 1.f;
        qw[(size_t)dim + c] = 1.f;
        kw[(size_t)dim + c] = 1.f;
        kv[(size_t)dim + c] = -0.5f;
    }
    for (int c = 0; c < dim; ++c) {
        x[((size_t)2 * hc) * dim + c] = 1.f;
        qw[c] = 1.f;
        kw[c] = 1.f;
        kv[(size_t)2 * span + c] = 0.5f;
    }

    std::vector<uint8_t> mask(rows, 1);
    for (int r = 0; r < rows; ++r) mask[r] = (r % 2 == 0) ? 1 : 0;  // rows 1, 3 masked

    std::vector<float> dots;
    const std::vector<float> exp_on = ref_engram(x, kv, qw, kw, rows, hc, dim, eps, &mask, &dots);
    const std::vector<float> exp_off = ref_engram(x, kv, qw, kw, rows, hc, dim, eps, nullptr,
                                                  nullptr);

    float *dx, *dkv, *dqw, *dkw;
    uint8_t* dmask;
    CHECK(cudaMalloc(&dx, xn * 4));
    CHECK(cudaMalloc(&dkv, kvn * 4));
    CHECK(cudaMalloc(&dqw, qw.size() * 4));
    CHECK(cudaMalloc(&dkw, kw.size() * 4));
    CHECK(cudaMalloc(&dmask, rows));

    auto run = [&](bool use_mask, std::vector<float>& got) -> int {
        CHECK(cudaMemcpy(dx, x.data(), xn * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dkv, kv.data(), kvn * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dqw, qw.data(), qw.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dkw, kw.data(), kw.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dmask, mask.data(), rows, cudaMemcpyHostToDevice));
        const int rc = dsv41_engram_apply(dx, dkv, dqw, dkw, use_mask ? dmask : nullptr, rows, hc,
                                          dim, eps, 0);
        const cudaError_t e = cudaDeviceSynchronize();
        if (rc != 0 || e != cudaSuccess) {
            printf("  rc=%d err=%s\n", rc, cudaGetErrorString(e));
            return 1;
        }
        CHECK(cudaMemcpy(got.data(), dx, xn * 4, cudaMemcpyDeviceToHost));
        return 0;
    };

    std::vector<float> got_on(xn), got_off(xn);
    int fails = 0;
    fails += run(true, got_on);
    fails += run(false, got_off);

    const float md_on = maxdiff_of(got_on, exp_on);
    const float md_off = maxdiff_of(got_off, exp_off);
    float md_masked = 0.f, md_maskeffect = 0.f;
    for (int r = 1; r < rows; r += 2) {
        for (size_t k = (size_t)r * hc * dim; k < (size_t)(r + 1) * hc * dim; ++k) {
            md_masked = std::fmax(md_masked, std::fabs(got_on[k] - x[k]));
            md_maskeffect = std::fmax(md_maskeffect, std::fabs(got_on[k] - got_off[k]));
        }
    }
    int npos = 0, nneg = 0;
    float dmin = 1e30f, dmax = 0.f;
    for (float d : dots) {
        if (d > 0.f) ++npos;
        if (d < 0.f) ++nneg;
        dmin = std::fmin(dmin, std::fabs(d));
        dmax = std::fmax(dmax, std::fabs(d));
    }
    const bool ok = md_on < 1e-5f && md_off < 1e-5f && md_masked < 1e-6f &&
                    md_maskeffect > 1e-6f && npos > 0 && nneg > 0;
    printf("  [engram rows=%d hc=%d dim=%d eps=%g] mask-on maxdiff=%.3e | mask-off maxdiff=%.3e | "
           "masked-rows exact=%.1e | mask effect=%.3e | dots +%d/-%d (|dot| in [%.3e, %.3e]) %s\n",
           rows, hc, dim, (double)eps, (double)md_on, (double)md_off, (double)md_masked,
           (double)md_maskeffect, npos, nneg, (double)dmin, (double)dmax,
           ok ? "OK" : "MISMATCH");
    printf("      crafted slots (c=0): (0,0) clamp-path %.6f/%.6f | (0,1) neg-dot %.6f/%.6f | "
           "(2,0) pos-dot %.6f/%.6f  (got/exp)\n",
           (double)got_on[0], (double)exp_on[0], (double)got_on[dim], (double)exp_on[dim],
           (double)got_on[((size_t)2 * hc) * dim], (double)exp_on[((size_t)2 * hc) * dim]);
    cudaFree(dx);
    cudaFree(dkv);
    cudaFree(dqw);
    cudaFree(dkw);
    cudaFree(dmask);
    return ok ? 0 : 1;
}

// --------------------------------------------------------------------------
// swiglu_limit case (canary after the buffer catches out-of-bounds stores)
// --------------------------------------------------------------------------
int run_swiglu_case(int rows, int inter, float limit, uint32_t seed) {
    rng = seed;
    const size_t n = (size_t)rows * 2 * inter;
    std::vector<float> gu(n);
    for (auto& v : gu) v = frand() * 3.f;
    if (inter == 5) {  // crafted: clamps fired / not fired, positive and negative
        const float g0[5] = {12.f, -3.f, 5.f, 20.f, 0.4f};
        const float u0[5] = {25.f, 3.f, -30.f, 0.1f, -2.f};
        const float g1[5] = {0.5f, -1.f, 2.f, 0.1f, 3.f};
        const float u1[5] = {1.f, -2.f, 0.5f, 0.3f, 0.9f};
        for (int i = 0; i < 5; ++i) {
            gu[i] = g0[i];
            gu[inter + i] = u0[i];
            gu[2 * inter + i] = g1[i];
            gu[3 * inter + i] = u1[i];
        }
    }

    // reference: ops.rs expert epilogue (clamps only when limit > 0)
    std::vector<float> exp = gu;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < inter; ++i) {
            float g = gu[(size_t)r * 2 * inter + i];
            float u = gu[(size_t)r * 2 * inter + inter + i];
            if (limit > 0.f) {
                u = std::fmin(std::fmax(u, -limit), limit);
                g = std::fmin(g, limit);
            }
            exp[(size_t)r * 2 * inter + i] = (g / (1.f + std::exp(-g))) * u;
        }

    const size_t canary = 32;
    std::vector<float> host(n + canary);
    std::copy(gu.begin(), gu.end(), host.begin());
    for (size_t i = 0; i < canary; ++i) host[n + i] = 123.25f + (float)i;

    float* dgu;
    CHECK(cudaMalloc(&dgu, (n + canary) * 4));
    CHECK(cudaMemcpy(dgu, host.data(), (n + canary) * 4, cudaMemcpyHostToDevice));
    const int rc = dsv41_swiglu_limit(dgu, rows, inter, limit, 0);
    const cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> got(n + canary);
    CHECK(cudaMemcpy(got.data(), dgu, (n + canary) * 4, cudaMemcpyDeviceToHost));

    float md = 0.f, md_up = 0.f;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < inter; ++i) {
            md = std::fmax(md, std::fabs(got[(size_t)r * 2 * inter + i] -
                                         exp[(size_t)r * 2 * inter + i]));
            md_up = std::fmax(md_up, std::fabs(got[(size_t)r * 2 * inter + inter + i] -
                                               gu[(size_t)r * 2 * inter + inter + i]));
        }
    bool canary_ok = true;
    for (size_t i = 0; i < canary; ++i)
        if (got[n + i] != host[n + i]) canary_ok = false;
    const bool ok = md < 1e-4f && md_up == 0.f && canary_ok;
    printf("  [swiglu rows=%d inter=%d limit=%g] rc=%d err=%s maxdiff=%.3e | up-half untouched "
           "(%.1e) canary=%s %s\n",
           rows, inter, (double)limit, rc, cudaGetErrorString(e), (double)md, (double)md_up,
           canary_ok ? "ok" : "HIT", ok ? "OK" : "MISMATCH");
    printf("      row0 gate-half[0..2] got %.5f %.5f %.5f | exp %.5f %.5f %.5f\n",
           (double)got[0], (double)got[1], (double)got[2], (double)exp[0], (double)exp[1],
           (double)exp[2]);
    cudaFree(dgu);
    return ok ? 0 : 1;
}

// --------------------------------------------------------------------------
// gather_rows case (repeating indices, exact copy)
// --------------------------------------------------------------------------
int run_gather_case(uint32_t seed) {
    rng = seed;
    const int n = 6, dim = 8;
    const std::vector<int32_t> idx = {3, 0, 3, 5, 2, 3};  // idx repeats
    std::vector<float> src((size_t)n * dim), exp((size_t)n * dim);
    for (auto& v : src) v = frand();
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < dim; ++c) exp[(size_t)i * dim + c] = src[(size_t)idx[i] * dim + c];

    float *dsrc, *dout;
    int32_t* didx;
    CHECK(cudaMalloc(&dsrc, src.size() * 4));
    CHECK(cudaMalloc(&didx, idx.size() * 4));
    CHECK(cudaMalloc(&dout, exp.size() * 4));
    CHECK(cudaMemcpy(dsrc, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    const int rc = dsv41_gather_rows(dsrc, didx, dout, n, dim, 0);
    const cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> got(exp.size());
    CHECK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));

    const float md = maxdiff_of(got, exp);
    const bool ok = md == 0.f;  // pure copy: must be bitwise exact
    printf("  [gather n=%d dim=%d idx=3,0,3,5,2,3] rc=%d err=%s maxdiff=%.3e exact=%s %s\n", n, dim,
           rc, cudaGetErrorString(e), (double)md, md == 0.f ? "yes" : "no", ok ? "OK" : "MISMATCH");
    cudaFree(dsrc);
    cudaFree(didx);
    cudaFree(dout);
    return ok ? 0 : 1;
}

// --------------------------------------------------------------------------
// scatter_add_rows case: duplicate idx (row 1 gets three contributions),
// untouched rows must keep their initial value; two runs reported for the
// run-to-run (atomic-order) observation.
// --------------------------------------------------------------------------
std::vector<float> ref_scatter(const std::vector<float>& dst0, const std::vector<float>& src,
                               const std::vector<int32_t>& idx, const std::vector<float>& w,
                               int n, int dim) {
    std::vector<float> dst = dst0;
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < dim; ++c)
            dst[(size_t)idx[i] * dim + c] += src[(size_t)i * dim + c] * w[i];
    return dst;
}

int run_scatter_case(uint32_t seed) {
    rng = seed;
    const int n = 6, dim = 8, drows = 6;
    const std::vector<int32_t> idx = {1, 0, 1, 2, 1, 4};  // row 1 x3, rows 3/5 untouched
    const std::vector<float> w = {1.5f, -0.5f, 2.f, 0.75f, 1.25f, -1.f};
    std::vector<float> src((size_t)n * dim), dst0((size_t)drows * dim, 0.5f);
    for (auto& v : src) v = frand();
    const std::vector<float> exp = ref_scatter(dst0, src, idx, w, n, dim);

    float *dsrc, *ddst, *dw;
    int32_t* didx;
    CHECK(cudaMalloc(&dsrc, src.size() * 4));
    CHECK(cudaMalloc(&didx, idx.size() * 4));
    CHECK(cudaMalloc(&dw, w.size() * 4));
    CHECK(cudaMalloc(&ddst, dst0.size() * 4));
    CHECK(cudaMemcpy(dsrc, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));

    auto run = [&](std::vector<float>& got) -> int {
        CHECK(cudaMemcpy(ddst, dst0.data(), dst0.size() * 4, cudaMemcpyHostToDevice));
        const int rc = dsv41_scatter_add_rows(dsrc, didx, dw, ddst, n, dim, 0);
        const cudaError_t e = cudaDeviceSynchronize();
        if (rc != 0 || e != cudaSuccess) {
            printf("  rc=%d err=%s\n", rc, cudaGetErrorString(e));
            return 1;
        }
        CHECK(cudaMemcpy(got.data(), ddst, got.size() * 4, cudaMemcpyDeviceToHost));
        return 0;
    };

    std::vector<float> got1(dst0.size()), got2(dst0.size());
    int fails = 0;
    fails += run(got1);
    fails += run(got2);
    const float md = maxdiff_of(got1, exp);
    const float md12 = maxdiff_of(got1, got2);  // observed atomic-order spread (info only)
    bool untouched_ok = true;
    for (int r : {3, 5})
        for (int c = 0; c < dim; ++c)
            if (got1[(size_t)r * dim + c] != 0.5f) untouched_ok = false;
    const bool ok = md < 1e-5f && untouched_ok;
    printf("  [scatter n=%d dim=%d idx=1,0,1,2,1,4 rows=6] maxdiff(vs seq ref)=%.3e | run-to-run "
           "spread=%.3e | untouched rows 3,5 exact=%s %s\n",
           n, dim, (double)md, (double)md12, untouched_ok ? "yes" : "no", ok ? "OK" : "MISMATCH");
    printf("      row 1 (3 contributions): got %8.5f %8.5f ... | ref %8.5f %8.5f ...\n",
           (double)got1[dim], (double)got1[dim + 1], (double)exp[dim], (double)exp[dim + 1]);
    cudaFree(dsrc);
    cudaFree(didx);
    cudaFree(dw);
    cudaFree(ddst);
    return ok ? 0 : 1;
}

// --------------------------------------------------------------------------
// hc_collapse case (ops.rs::hc_pre)
// --------------------------------------------------------------------------
int run_hc_collapse_case(int rows, int hc, int dim, uint32_t seed) {
    rng = seed;
    std::vector<float> x((size_t)rows * hc * dim), pre((size_t)rows * hc);
    for (auto& v : x) v = frand() * 2.f;
    for (auto& v : pre) v = frand();
    std::vector<float> exp((size_t)rows * dim, 0.f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < dim; ++c) {
            float acc = 0.f;
            for (int i = 0; i < hc; ++i)
                acc += pre[r * hc + i] * x[((size_t)r * hc + i) * dim + c];
            exp[(size_t)r * dim + c] = acc;
        }

    float *dx, *dpre, *dout;
    CHECK(cudaMalloc(&dx, x.size() * 4));
    CHECK(cudaMalloc(&dpre, pre.size() * 4));
    CHECK(cudaMalloc(&dout, exp.size() * 4));
    CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dpre, pre.data(), pre.size() * 4, cudaMemcpyHostToDevice));
    const int rc = dsv41_hc_collapse(dx, dpre, dout, rows, hc, dim, 0);
    const cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> got(exp.size());
    CHECK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));

    const float md = maxdiff_of(got, exp);
    const bool ok = md < 1e-5f;
    printf("  [hc_collapse rows=%d hc=%d dim=%d] rc=%d err=%s maxdiff=%.3e %s\n", rows, hc, dim,
           rc, cudaGetErrorString(e), (double)md, ok ? "OK" : "MISMATCH");
    cudaFree(dx);
    cudaFree(dpre);
    cudaFree(dout);
    return ok ? 0 : 1;
}

// --------------------------------------------------------------------------
// occupancy report (launch shapes of this file)
// --------------------------------------------------------------------------
void report_occupancy() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return;
    printf("--- occupancy (device %s, %d SMs, %d threads/SM) ---\n", prop.name,
           prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, engram_apply_kernel, 128, 0);
    printf("  engram_apply     block=128 -> %d blocks/SM (%4d threads/SM)\n", nb, nb * 128);
    nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, swiglu_limit_kernel, 256, 0);
    printf("  swiglu_limit     block=256 -> %d blocks/SM (%4d threads/SM)\n", nb, nb * 256);
    nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, gather_rows_kernel, 256, 0);
    printf("  gather_rows      block=256 -> %d blocks/SM (%4d threads/SM)\n", nb, nb * 256);
    nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, scatter_add_rows_kernel, 256, 0);
    printf("  scatter_add_rows block=256 -> %d blocks/SM (%4d threads/SM)\n", nb, nb * 256);
    nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, hc_collapse_kernel, 256, 0);
    printf("  hc_collapse      block=256 -> %d blocks/SM (%4d threads/SM)\n", nb, nb * 256);
}

}  // namespace

int main() {
    printf("== dsv41 glue kernels self-test ==\n");
    int fails = 0;

    printf("--- engram_apply ---\n");
    fails += run_engram_case(4, 2, 8, 1e-6f, 101);

    printf("--- swiglu_limit ---\n");
    fails += run_swiglu_case(3, 5, 10.f, 202);  // clamps fire
    fails += run_swiglu_case(3, 5, 0.f, 202);   // limit <= 0: no clamps

    printf("--- gather / scatter ---\n");
    fails += run_gather_case(303);
    fails += run_scatter_case(404);

    printf("--- hc_collapse ---\n");
    fails += run_hc_collapse_case(4, 3, 8, 505);

    report_occupancy();

    printf(fails ? "RESULT: %d case(s) FAILED\n" : "RESULT: all cases passed\n", fails);
    return fails ? 1 : 0;
}
