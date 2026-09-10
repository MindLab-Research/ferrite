// tests_dsv41_attn.cu — numerical self-test for the two attention-side kernels
// added to dsv41_kernels.cu: dsv41_indexer_topk and dsv41_compressor.
//
// CPU references follow crates/ferrite-dsv41/src/ops.rs (indexer_topk and
// compressor_forward) line by line. For the compressor the reference reuses the
// kernel's own quantised activation (read back from the device) so that the
// comparison isolates the GEMM/pool/norm arithmetic from the quantisation rule,
// which is verified separately by construction (same block-32 pow2 scales).
//
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//            tests_dsv41_attn.cu -o /tmp/t_attn
#include "dsv41_kernels.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

uint32_t rng = 12345;
uint32_t xr() { rng = rng * 1664525u + 1013904223u; return rng; }
float frand() { return (float)((int)(xr() % 2001) - 1000) / 1000.f; }

// host-side dequantisation helpers (same formulas as the device ones)
float e4m3_h(uint8_t b) {
    if ((b & 0x7Fu) == 0x7Fu) return NAN;  // e4m3 NaN (0x7f / 0xff)
    const float s = (b & 0x80u) ? -1.f : 1.f;
    const int e = (b >> 3) & 0xFu, m = b & 0x7u;
    if (e == 0) return s * (float)m * (1.f / 512.f);
    return s * (1.f + (float)m * 0.125f) * ldexpf(1.f, e - 7);
}
float ue8m0_h(uint8_t b) { return ldexpf(1.f, (int)b - 127); }

// --------------------------------------------------------------------------
// indexer_topk reference (ops.rs::indexer_topk)
// --------------------------------------------------------------------------
std::vector<int32_t> ref_indexer_topk(const std::vector<float>& q, const std::vector<float>& ik,
                                      const std::vector<float>& w, const std::vector<uint8_t>& cand,
                                      bool use_cand, const std::vector<int32_t>& lens, int b, int m,
                                      int nh, int hd, int n_pos, int topk, int offset,
                                      float softmax_scale, float head_scale) {
    const int cols = std::min(topk, n_pos);
    std::vector<int32_t> out((size_t)b * m * cols, -1);
    for (int bb = 0; bb < b; ++bb)
        for (int mm = 0; mm < m; ++mm) {
            const int cl = std::min<int>(lens[mm], n_pos);
            std::vector<float> score(n_pos, 0.f);
            for (int p = 0; p < n_pos; ++p) {
                float acc = 0.f;
                for (int h = 0; h < nh; ++h) {
                    float dot = 0.f;
                    for (int c = 0; c < hd; ++c)
                        dot += q[((size_t)(bb * m + mm) * nh + h) * hd + c] *
                               ik[((size_t)bb * n_pos + p) * hd + c];
                    acc += std::max(dot, 0.f) * w[(size_t)(bb * m + mm) * nh + h];
                }
                float s = acc * softmax_scale * head_scale;
                if (p >= cl) s = -INFINITY;
                if (use_cand && !cand[(size_t)(bb * m + mm) * n_pos + p]) s = -INFINITY;
                score[p] = s;
            }
            std::vector<int> order(n_pos);
            std::iota(order.begin(), order.end(), 0);
            // stable descending sort (equal scores keep the lower position first)
            std::stable_sort(order.begin(), order.end(),
                             [&](int x, int y) { return score[x] > score[y]; });
            std::vector<int> picked(order.begin(), order.begin() + cols);
            std::sort(picked.begin(), picked.end());
            for (int i = 0; i < cols; ++i) {
                const int p = picked[i];
                out[((size_t)(bb * m + mm)) * cols + i] = (p < cl) ? (p + offset) : -1;
            }
        }
    return out;
}

int run_indexer_case(int b, int m, int nh, int hd, int n_pos, int topk, int offset, bool use_cand,
                     uint32_t seed) {
    rng = seed;
    std::vector<float> q((size_t)b * m * nh * hd), ik((size_t)b * n_pos * hd),
        w((size_t)b * m * nh);
    for (auto& v : q) v = frand();
    for (auto& v : ik) v = frand();
    for (auto& v : w) v = std::fabs(frand());
    std::vector<uint8_t> cand((size_t)b * m * n_pos, 1);
    for (auto& v : cand)
        if (use_cand) v = (uint8_t)(xr() % 4 != 0);  // ~75% allowed
    std::vector<int32_t> lens(m);
    for (int i = 0; i < m; ++i) lens[i] = (int)(xr() % (unsigned)(n_pos + 1));
    if (m >= 2) {
        lens[0] = n_pos;      // full row
        lens[1] = n_pos / 2;  // half row (some picks fall past the cut -> -1)
    }
    const float softmax_scale = 1.f / std::sqrt((float)hd);
    const float head_scale = 1.f / std::sqrt((float)nh);

    const int cols = std::min(topk, n_pos);
    std::vector<int32_t> got((size_t)b * m * cols, 0);
    float *dq, *dik, *dw;
    uint8_t* dcand;
    int32_t *dlens, *dout;
    cudaMalloc(&dq, q.size() * 4);
    cudaMalloc(&dik, ik.size() * 4);
    cudaMalloc(&dw, w.size() * 4);
    cudaMalloc(&dcand, cand.size());
    cudaMalloc(&dlens, lens.size() * 4);
    cudaMalloc(&dout, got.size() * 4);
    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dik, ik.data(), ik.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dcand, cand.data(), cand.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dlens, lens.data(), lens.size() * 4, cudaMemcpyHostToDevice);
    int rc = dsv41_indexer_topk(dq, dik, dw, use_cand ? dcand : nullptr, dlens, dout, b, m, nh, hd,
                                n_pos, topk, offset, softmax_scale, head_scale, use_cand ? 1 : 0,
                                0);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    const std::vector<int32_t> exp = ref_indexer_topk(q, ik, w, cand, use_cand, lens, b, m, nh, hd,
                                                      n_pos, topk, offset, softmax_scale,
                                                      head_scale);
    int bad = 0;
    size_t first = (size_t)-1;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != exp[i]) {
            ++bad;
            if (first == (size_t)-1) first = i;
        }
    printf("  [indexer b=%d m=%d nh=%d hd=%d n_pos=%d topk=%d cand=%d] rc=%d err=%s -> %s",
           b, m, nh, hd, n_pos, topk, (int)use_cand, rc, cudaGetErrorString(e),
           bad == 0 ? "EXACT\n" : "MISMATCH\n");
    if (bad) {
        const size_t r = first / cols, i = first % cols;
        printf("      first mismatch row=%zu slot=%zu: got %d expect %d\n", r, i, got[first],
               exp[first]);
        printf("      got : ");
        for (size_t k = 0; k < std::min<size_t>(16, got.size()); ++k) printf("%d ", got[k]);
        printf("\n      exp : ");
        for (size_t k = 0; k < std::min<size_t>(16, exp.size()); ++k) printf("%d ", exp[k]);
        printf("\n");
    }
    cudaFree(dq); cudaFree(dik); cudaFree(dw); cudaFree(dcand); cudaFree(dlens); cudaFree(dout);
    return bad == 0 ? 0 : 1;
}

// --------------------------------------------------------------------------
// compressor reference (ops.rs::compressor_forward) over the kernel's own
// quantised activations
// --------------------------------------------------------------------------
struct RefState {
    std::vector<double> kv, score;
    explicit RefState(size_t n, double neginf) : kv(n, 0.0), score(n, neginf) {}
};

// dequantised projection of one row block; xd is [rows, dim] (already dequantised
// exactly like the kernel's fp8 path), wd is [hd, dim]
void ref_proj(const std::vector<double>& xd, const std::vector<double>& wd, int rows, int dim,
              int hd, std::vector<double>& out) {
    out.assign((size_t)rows * hd, 0.0);
    for (int r = 0; r < rows; ++r)
        for (int o = 0; o < hd; ++o) {
            double acc = 0.0;
            for (int k = 0; k < dim; ++k)
                acc += xd[(size_t)r * dim + k] * wd[(size_t)o * dim + k];
            out[(size_t)r * hd + o] = acc;
        }
}

void ref_rmsnorm(std::vector<double>& rows, int nrows, int hd, const std::vector<float>& nw,
                 float eps) {
    for (int r = 0; r < nrows; ++r) {
        double ss = 0.0;
        for (int c = 0; c < hd; ++c) {
            const double v = rows[(size_t)r * hd + c];
            ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / (double)hd + (double)eps);
        for (int c = 0; c < hd; ++c)
            rows[(size_t)r * hd + c] = rows[(size_t)r * hd + c] * inv * (double)nw[c];
    }
}

// `dstate_kv` / `dstate_sc` are DEVICE pointers sized b*ratio*hd; the caller owns
// them so a decode chain carries the state across steps (like the real runtime).
int run_compressor_case(int b, int seqlen, int dim, int hd, int ratio, int start_pos,
                        const std::vector<float>& x, const std::vector<uint8_t>& wkv,
                        const std::vector<uint8_t>& wkv_s, const std::vector<uint8_t>& wgate,
                        const std::vector<uint8_t>& wgate_s, const std::vector<float>& norm_w,
                        RefState& ref, float eps, uint32_t seed, const char* tag,
                        float* dstate_kv, float* dstate_sc, bool reset_state) {
    rng = seed;
    const int rows = b * seqlen;
    const int nb = dim / 32;
    // --- host reference projections (over the kernel's quantised activations)
    std::vector<uint8_t> xq((size_t)rows * dim);
    std::vector<float> xqs((size_t)rows * nb);   // f32 pow2 activation scales
    {
        uint8_t *dxq;
        float *dxs;   // activation scales are f32 (see gemm_fp8_kernel's note)
        float* dx;
        cudaMalloc(&dxq, xq.size());
        cudaMalloc(&dxs, xqs.size() * sizeof(float));
        cudaMalloc(&dx, x.size() * 4);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        const int units = rows * nb;
        quant_e4m3_pow2_kernel<<<(units + 3) / 4, 128>>>(dx, dxq, dxs, rows, nb);
        cudaDeviceSynchronize();
        cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(xqs.data(), dxs, xqs.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dxq); cudaFree(dxs); cudaFree(dx);
    }
    std::vector<double> xd((size_t)rows * dim), wd((size_t)hd * dim), gd((size_t)hd * dim);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < dim; ++k)
            xd[(size_t)r * dim + k] =
                (double)e4m3_h(xq[(size_t)r * dim + k]) * (double)xqs[(size_t)r * nb + k / 32];
    for (int o = 0; o < hd; ++o)
        for (int k = 0; k < dim; ++k) {
            wd[(size_t)o * dim + k] =
                (double)e4m3_h(wkv[(size_t)o * dim + k]) * (double)ue8m0_h(wkv_s[(size_t)(o / 32) * nb + k / 32]);
            if (ratio > 1)
                gd[(size_t)o * dim + k] =
                    (double)e4m3_h(wgate[(size_t)o * dim + k]) * (double)ue8m0_h(wgate_s[(size_t)(o / 32) * nb + k / 32]);
        }
    std::vector<double> kvp, scp;
    ref_proj(xd, wd, rows, dim, hd, kvp);
    if (ratio > 1) ref_proj(xd, gd, rows, dim, hd, scp);

    // --- expected latents / state update (per ops.rs, with -inf empty slots)
    std::vector<double> exp_lat;
    int exp_rows = 0;
    if (ratio == 1) {
        exp_lat = kvp;
        ref_rmsnorm(exp_lat, rows, hd, norm_w, eps);
        exp_rows = seqlen;
    } else {
        bool should;
        if (start_pos == 0) {
            const int ngroups = seqlen / ratio;
            exp_rows = ngroups;
            should = seqlen >= ratio;
            const int rem = seqlen % ratio, cut = seqlen - rem;
            for (int bb = 0; bb < b; ++bb)
                for (int t = 0; t < rem; ++t)
                    for (int c = 0; c < hd; ++c) {
                        ref.kv[((size_t)bb * ratio + t) * hd + c] =
                            kvp[((size_t)bb * seqlen + cut + t) * hd + c];
                        ref.score[((size_t)bb * ratio + t) * hd + c] =
                            scp[((size_t)bb * seqlen + cut + t) * hd + c];
                    }
        } else {
            should = (start_pos + 1) % ratio == 0;
            exp_rows = 1;
            const int slot = start_pos % ratio;
            for (int bb = 0; bb < b; ++bb)
                for (int c = 0; c < hd; ++c) {
                    ref.kv[((size_t)bb * ratio + slot) * hd + c] = kvp[(size_t)bb * hd + c];
                    ref.score[((size_t)bb * ratio + slot) * hd + c] = scp[(size_t)bb * hd + c];
                }
        }
        if (should) {
            exp_lat.assign((size_t)b * exp_rows * hd, 0.0);
            for (int bb = 0; bb < b; ++bb)
                for (int g = 0; g < exp_rows; ++g)
                    for (int c = 0; c < hd; ++c) {
                        const int rr = std::min(ratio, 32);
                        double mx = -INFINITY;
                        std::vector<double> sv(rr), vv(rr);
                        for (int r = 0; r < rr; ++r) {
                            if (start_pos == 0) {
                                sv[r] = scp[((size_t)bb * seqlen + g * ratio + r) * hd + c];
                                vv[r] = kvp[((size_t)bb * seqlen + g * ratio + r) * hd + c];
                            } else {
                                sv[r] = ref.score[((size_t)bb * ratio + r) * hd + c];
                                vv[r] = ref.kv[((size_t)bb * ratio + r) * hd + c];
                            }
                            mx = std::max(mx, sv[r]);
                        }
                        double den = 0.0, acc = 0.0;
                        for (int r = 0; r < rr; ++r) {
                            const double e = std::exp(sv[r] - mx);
                            den += e;
                            acc += e * vv[r];
                        }
                        exp_lat[((size_t)bb * exp_rows + g) * hd + c] = den > 0.0 ? acc / den : 0.0;
                    }
            ref_rmsnorm(exp_lat, b * exp_rows, hd, norm_w, eps);
        } else {
            exp_rows = 0;
        }
    }

    // --- GPU call
    float* dx;
    uint8_t *dwkv, *dwkvs, *dwg, *dwgs;
    float *dnw, *dlat;
    int32_t* dout_rows;
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&dwkv, wkv.size());
    cudaMalloc(&dwkvs, wkv_s.size());
    cudaMalloc(&dwg, wgate.size());
    cudaMalloc(&dwgs, wgate_s.size());
    cudaMalloc(&dnw, norm_w.size() * 4);
    cudaMalloc(&dlat, (size_t)b * std::max(seqlen, 1) * hd * 4 + 16);
    cudaMalloc(&dout_rows, 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwkv, wkv.data(), wkv.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dwkvs, wkv_s.data(), wkv_s.size(), cudaMemcpyHostToDevice);
    if (ratio > 1) {
        cudaMemcpy(dwg, wgate.data(), wgate.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dwgs, wgate_s.data(), wgate_s.size(), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dnw, norm_w.data(), norm_w.size() * 4, cudaMemcpyHostToDevice);
    // the state starts life with -inf scores (CompressorState::new); a decode
    // chain only resets it on the first step
    if (reset_state) {
        std::vector<float> skv((size_t)b * ratio * hd, 0.f),
            ssc((size_t)b * ratio * hd, -INFINITY);
        cudaMemcpy(dstate_kv, skv.data(), skv.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dstate_sc, ssc.data(), ssc.size() * 4, cudaMemcpyHostToDevice);
    }
    cudaMemset(dlat, 0, (size_t)b * std::max(seqlen, 1) * hd * 4 + 16);
    int rc = dsv41_compressor(x.data() == nullptr ? dx : dx, dwkv, dwkvs, ratio > 1 ? dwg : nullptr,
                              ratio > 1 ? dwgs : nullptr, dnw, dstate_kv, dstate_sc, dlat,
                              dout_rows, b, seqlen, dim, hd, ratio, start_pos, eps, 0);
    cudaError_t e = cudaDeviceSynchronize();
    int32_t got_rows = -1;
    cudaMemcpy(&got_rows, dout_rows, 4, cudaMemcpyDeviceToHost);
    std::vector<float> lat((size_t)b * std::max(seqlen, 1) * hd);
    cudaMemcpy(lat.data(), dlat, lat.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<float> gkv(ref.kv.size()), gsc(ref.score.size());
    cudaMemcpy(gkv.data(), dstate_kv, gkv.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gsc.data(), dstate_sc, gsc.size() * 4, cudaMemcpyDeviceToHost);

    double md = 0.0;
    if (exp_rows == 0) {
        for (size_t i = 0; i < lat.size(); ++i) {
            double d = std::fabs((double)lat[i]);
            if (std::isnan(d)) d = 1e30;
            md = std::max(md, d);
        }
    } else {
        for (int bb = 0; bb < b; ++bb)
            for (int g = 0; g < exp_rows; ++g)
                for (int c = 0; c < hd; ++c) {
                    double d = std::fabs((double)lat[((size_t)bb * exp_rows + g) * hd + c] -
                                         exp_lat[((size_t)bb * exp_rows + g) * hd + c]);
                    if (std::isnan(d)) d = 1e30;  // NaN must never be swallowed by max()
                    md = std::max(md, d);
                }
    }
    // state comparison (the compressor carries kv/score across steps)
    double smd_kv = 0.0, smd_sc = 0.0;
    for (size_t i = 0; i < ref.kv.size(); ++i) {
        smd_kv = std::max(smd_kv, std::fabs((double)gkv[i] - ref.kv[i]));
        const double ds = std::fabs((double)gsc[i] - (double)ref.score[i]);
        if (!(std::isinf((double)ref.score[i]) && std::isinf((double)gsc[i])))
            smd_sc = std::max(smd_sc, ds);
    }
    const bool rows_ok = got_rows == exp_rows;
    // include the -inf-aware score comparison in the verdict
    double smd_sc_chk = 0.0;
    for (size_t i = 0; i < ref.score.size(); ++i) {
        const double a = (double)gsc[i], b2 = (double)ref.score[i];
        const bool ainf = std::isinf(a) && a < 0, binf = std::isinf(b2) && b2 < 0;
        if (ainf && binf) continue;
        double d = std::fabs(a - b2);
        if (std::isnan(d)) d = 1e30;
        smd_sc_chk = std::max(smd_sc_chk, d);
    }
    const bool ok = md < 1e-4 && rows_ok && smd_kv < 1e-4 && smd_sc_chk < 1e-4;
    if (!ok && exp_rows > 0) {
        printf("      state slots (b=%d ratio=%d): ", b, ratio);
        for (int sl = 0; sl < ratio; ++sl)
            printf("[slot%d kv %.3f/%.3f sc %.3f/%.3f] ", sl, (double)gkv[(size_t)sl * hd],
                   ref.kv[(size_t)sl * hd], (double)gsc[(size_t)sl * hd], ref.score[(size_t)sl * hd]);
        printf("\n");
    }
    printf("  [compressor %s b=%d seqlen=%d dim=%d hd=%d ratio=%d sp=%d] rc=%d err=%s "
           "out_rows=%d (exp %d) maxdiff=%.3e state_kv=%.3e state_sc=%.3e %s\n",
           tag, b, seqlen, dim, hd, ratio, start_pos, rc, cudaGetErrorString(e), got_rows, exp_rows,
           md, smd_kv, smd_sc_chk, ok ? "OK" : "MISMATCH");
    if (exp_rows > 0 && (std::string(tag) == "ratio1")) {
        printf("      ratio1 check: lat c0..15 = ");
        for (int c = 0; c < 16; ++c) printf("%8.4f ", (double)lat[c]);
        printf("\n                    exp c0..15 = ");
        for (int c = 0; c < 16; ++c) printf("%8.4f ", exp_lat[c]);
        printf("\n");
    }
    if (!ok && exp_rows > 0) {
        printf("      latents g=0 c0..3: got %.6f %.6f %.6f %.6f | exp %.6f %.6f %.6f %.6f\n",
               (double)lat[0], (double)lat[1], (double)lat[2], (double)lat[3], exp_lat[0],
               exp_lat[1], exp_lat[2], exp_lat[3]);
        if (exp_rows > 1)
            printf("      latents g=1 c0..3: got %.6f %.6f %.6f %.6f | exp %.6f %.6f %.6f %.6f\n",
                   (double)lat[hd], (double)lat[hd + 1], (double)lat[hd + 2], (double)lat[hd + 3],
                   exp_lat[hd], exp_lat[hd + 1], exp_lat[hd + 2], exp_lat[hd + 3]);
    }
    cudaFree(dx); cudaFree(dwkv); cudaFree(dwkvs); cudaFree(dwg); cudaFree(dwgs); cudaFree(dnw);
    cudaFree(dlat); cudaFree(dout_rows);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    printf("== dsv41 attention-side kernels self-test ==\n");
    int fails = 0;

    // ---- indexer_topk
    fails += run_indexer_case(1, 4, 4, 8, 32, 8, 128, false, 1);
    fails += run_indexer_case(1, 4, 4, 8, 32, 8, 128, true, 2);
    fails += run_indexer_case(2, 3, 2, 4, 16, 12, 64, true, 3);   // topk > valid -> -1s
    fails += run_indexer_case(1, 2, 4, 16, 64, 32, 0, false, 4);

    // ---- compressor (shared weight tensors for all cases)
    const int dim = 64, hd = 64;
    std::vector<uint8_t> wkv((size_t)hd * dim), wkv_s((size_t)(hd / 32) * (dim / 32));
    std::vector<uint8_t> wgate((size_t)hd * dim), wgate_s((size_t)(hd / 32) * (dim / 32));
    rng = 777;
    // NOTE: 0x7f is e4m3 NaN -- keep the weights inside the valid positive
    // range 0x30..0x5f (2^-1 .. 1.875*2^4) so the reference is well defined.
    for (auto& v : wkv) v = (uint8_t)(0x30 + (xr() % 0x30));
    for (auto& v : wgate) v = (uint8_t)(0x30 + (xr() % 0x30));
    for (auto& v : wkv_s) v = (uint8_t)(120 + (xr() % 8));
    for (auto& v : wgate_s) v = (uint8_t)(120 + (xr() % 8));
    std::vector<float> norm_w(hd);
    for (auto& v : norm_w) v = 0.5f + std::fabs(frand());

    {   // ratio == 1
        const int b = 2, seqlen = 4;
        std::vector<float> x((size_t)b * seqlen * dim);
        for (auto& v : x) v = frand() * 3.f;
        RefState st(0, 0.0);
        float *dkv, *dsc;
        cudaMalloc(&dkv, 16);
        cudaMalloc(&dsc, 16);
        fails += run_compressor_case(b, seqlen, dim, hd, 1, 0, x, wkv, wkv_s, {}, {}, norm_w, st,
                                     1e-5f, 11, "ratio1", dkv, dsc, true);
        cudaFree(dkv);
        cudaFree(dsc);
    }
    {   // ratio == 2 prefill with a remainder (state slot 0 must be written)
        const int b = 1, seqlen = 5, ratio = 2;
        std::vector<float> x((size_t)b * seqlen * dim);
        for (auto& v : x) v = frand() * 3.f;
        RefState st((size_t)b * ratio * hd, -INFINITY);
        float *dkv, *dsc;
        cudaMalloc(&dkv, (size_t)b * ratio * hd * 4);
        cudaMalloc(&dsc, (size_t)b * ratio * hd * 4);
        fails += run_compressor_case(b, seqlen, dim, hd, ratio, 0, x, wkv, wkv_s, wgate, wgate_s,
                                     norm_w, st, 1e-5f, 12, "prefill5", dkv, dsc, true);
        cudaFree(dkv);
        cudaFree(dsc);
    }
    {   // ratio == 2 decode chain: the DEVICE state is allocated once and carried
        // across the steps, exactly like the runtime does.
        const int b = 1, ratio = 2;
        RefState st((size_t)b * ratio * hd, -INFINITY);
        float *dkv, *dsc;
        cudaMalloc(&dkv, (size_t)b * ratio * hd * 4);
        cudaMalloc(&dsc, (size_t)b * ratio * hd * 4);
        for (int step = 0; step < 6; ++step) {
            std::vector<float> x((size_t)b * dim);
            for (auto& v : x) v = frand() * 3.f;
            fails += run_compressor_case(b, 1, dim, hd, ratio, step, x, wkv, wkv_s, wgate,
                                         wgate_s, norm_w, st, 1e-5f, 100 + step, "decode", dkv, dsc,
                                         step == 0);
        }
        cudaFree(dkv);
        cudaFree(dsc);
    }
    {   // seqlen < ratio on prefill: no group completes (out_rows = 0)
        const int b = 1, seqlen = 1, ratio = 2;
        std::vector<float> x((size_t)b * seqlen * dim);
        for (auto& v : x) v = frand() * 3.f;
        RefState st((size_t)b * ratio * hd, -INFINITY);
        float *dkv, *dsc;
        cudaMalloc(&dkv, (size_t)b * ratio * hd * 4);
        cudaMalloc(&dsc, (size_t)b * ratio * hd * 4);
        fails += run_compressor_case(b, seqlen, dim, hd, ratio, 0, x, wkv, wkv_s, wgate, wgate_s,
                                     norm_w, st, 1e-5f, 13, "shortprefill", dkv, dsc, true);
        cudaFree(dkv);
        cudaFree(dsc);
    }

    printf(fails ? "RESULT: %d case(s) FAILED\n" : "RESULT: all cases passed\n", fails);
    return fails ? 1 : 0;
}
