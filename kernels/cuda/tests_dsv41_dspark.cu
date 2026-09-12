// tests_dsv41_dspark.cu — numerical self-test for the DSpark draft head kernel
// in dsv41_glue.cu (dsv41_dspark_markov_head / dspark_markov_head_kernel).
//
// The CPU reference below is a line-by-line transcription of
// crates/ferrite-models/src/dsv41/dspark.rs::forward_head's sequential loop
// (the logits bias, the Gumbel-max/Markov sample, the confidence score) plus
// ops.rs::gumbel_argmax, which it drives with the SAME constant `u == 1.0` the
// chain uploads (chain.rs::forward_spec). Two things are asserted:
//
//   1. the sampled block `ids[1..=bs]` is IDENTICAL to the reference's at
//      temperature 0 AND at temperature 1 (ops.rs::gumbel_argmax with u == 1 is
//      `argmax` either way - softmax is monotone and its denominator is a
//      positive constant across the vocabulary);
//   2. the biased logits and the confidence match within f32 tolerance (the
//      kernel splits each [mr]-dot across the warp and folds it as a tree,
//      while the reference accumulates it serially, so the two differ by a
//      few ULP - the sampled ids are the only bit-level contract).
//
// Cases: a small random shape (mr % 4 == 0, the float4 arm), a small shape with
// mr % 4 != 0 (the scalar arm), a deliberate all-ties case (checks the
// lowest-index tie rule), and the FULL production shape
// (vocab 129280 x mr 256, dim 5120, bs 5).
//
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//            tests_dsv41_dspark.cu -o /tmp/t_dspark
// Run:   CUDA_VISIBLE_DEVICES=0 ./t_dspark
#include "dsv41_glue.cu"

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

uint32_t rng = 987654321u;
uint32_t xr() { rng = rng * 1664525u + 1013904223u; return rng; }
float frand() { return (float)((int)(xr() % 2001) - 1000) / 500.f; }

// --------------------------------------------------------------------------
// reference: ops.rs::gumbel_argmax, called with the chain's constant u == 1.0
// --------------------------------------------------------------------------
int ref_gumbel_argmax(const float* logits, int vocab, float temperature) {
    if (temperature == 0.0f) {
        int best = 0;
        float bv = -INFINITY;
        for (int i = 0; i < vocab; ++i) {
            if (logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        }
        return best;
    }
    const float t = std::fmax(temperature, 1e-5f);
    float mx = -INFINITY;
    for (int i = 0; i < vocab; ++i) mx = std::fmax(mx, logits[i]);
    float den = 0.f;
    for (int i = 0; i < vocab; ++i) den += std::exp((logits[i] - mx) / t);
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < vocab; ++i) {
        const float p = std::exp((logits[i] - mx) / t) / den;
        const float v = p / std::fmax(1.0f, 1e-30f);   // u == 1.0 for every entry
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return best;
}

// --------------------------------------------------------------------------
// reference: dspark.rs::forward_head's Markov loop
// --------------------------------------------------------------------------
std::vector<int> ref_markov(const std::vector<float>& logits0, const std::vector<float>& h,
                            const std::vector<float>& m_embed, const std::vector<float>& m_head,
                            const std::vector<float>& conf_proj, int bs, int vocab, int mr,
                            int dim, int t0, float temperature, std::vector<float>* logits_out,
                            std::vector<float>* conf_out) {
    std::vector<float> lg = logits0;
    std::vector<int> ids(bs + 1, 0);
    ids[0] = t0;
    std::vector<float> conf(bs, 0.f);
    for (int i = 0; i < bs; ++i) {
        const float* er = &m_embed[(size_t)ids[i] * mr];
        for (int v = 0; v < vocab; ++v) {
            const float* wr = &m_head[(size_t)v * mr];
            float acc = 0.f;
            for (int c = 0; c < mr; ++c) acc += wr[c] * er[c];
            lg[(size_t)i * vocab + v] += acc;
        }
        ids[i + 1] = ref_gumbel_argmax(&lg[(size_t)i * vocab], vocab, temperature);
        float a = 0.f;
        for (int c = 0; c < dim; ++c) a += conf_proj[c] * h[(size_t)i * dim + c];
        for (int c = 0; c < mr; ++c) a += conf_proj[dim + c] * er[c];
        conf[i] = a;
    }
    if (logits_out) *logits_out = lg;
    if (conf_out) *conf_out = conf;
    return ids;
}

float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float md = 0.f;
    for (size_t i = 0; i < a.size(); ++i) md = std::fmax(md, std::fabs(a[i] - b[i]));
    return md;
}

// --------------------------------------------------------------------------
// one case: build the tensors, run bs launches, compare against the reference
// --------------------------------------------------------------------------
int run_case(const char* name, int vocab, int mr, int dim, int bs, float temperature, int t0,
             bool all_ties) {
    const size_t n_lg = (size_t)bs * vocab;
    std::vector<float> logits(n_lg);
    for (size_t i = 0; i < n_lg; ++i) logits[i] = all_ties ? 0.f : frand() * 4.f;
    std::vector<float> h((size_t)bs * dim);
    for (auto& v : h) v = frand();
    std::vector<float> m_embed((size_t)vocab * mr);
    for (auto& v : m_embed) v = all_ties ? 0.f : frand();
    std::vector<float> m_head((size_t)vocab * mr);
    for (auto& v : m_head) v = all_ties ? 0.f : frand();
    std::vector<float> conf_proj((size_t)dim + mr);
    for (auto& v : conf_proj) v = frand();

    std::vector<float> ref_lg, ref_conf;
    std::vector<int> ref_ids = ref_markov(logits, h, m_embed, m_head, conf_proj, bs, vocab, mr, dim,
                                          t0, temperature, &ref_lg, &ref_conf);

    // ---- device ----
    float *d_lg = nullptr, *d_h = nullptr, *d_me = nullptr, *d_mh = nullptr, *d_cp = nullptr,
          *d_conf = nullptr;
    int* d_ids = nullptr;
    unsigned long long* d_part = nullptr;
    unsigned* d_ctr = nullptr;
    CHECK(cudaMalloc(&d_lg, n_lg * sizeof(float)));
    CHECK(cudaMalloc(&d_h, h.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_me, m_embed.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_mh, m_head.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_cp, conf_proj.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_conf, (size_t)bs * sizeof(float)));
    CHECK(cudaMalloc(&d_ids, (size_t)(bs + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_part, (size_t)DSPARK_MARKOV_MAX_BLOCKS * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_ctr, sizeof(unsigned)));
    CHECK(cudaMemset(d_ctr, 0, sizeof(unsigned)));

    std::vector<int> ids_host(bs + 1, 0);
    ids_host[0] = t0;
    CHECK(cudaMemcpy(d_lg, logits.data(), n_lg * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_h, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_me, m_embed.data(), m_embed.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mh, m_head.data(), m_head.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cp, conf_proj.data(), conf_proj.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ids, ids_host.data(), (size_t)(bs + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    // one launch per draft row, in order (the kernel's documented contract:
    // `ids[step]` in, `ids[step + 1]` out, same stream)
    for (int step = 0; step < bs; ++step) {
        int rc = dsv41_dspark_markov_head(d_lg, d_h, d_me, d_mh, d_cp, d_ids, d_conf, dim, vocab,
                                          mr, step, d_part, d_ctr, 0);
        if (rc != (int)cudaSuccess) {
            printf("  [%s] launch failed: %s\n", name, cudaGetErrorString((cudaError_t)rc));
            return 1;
        }
    }
    CHECK(cudaDeviceSynchronize());

    std::vector<int> got_ids(bs + 1, 0);
    std::vector<float> got_lg(n_lg, 0.f), got_conf(bs, 0.f);
    CHECK(cudaMemcpy(got_ids.data(), d_ids, (size_t)(bs + 1) * sizeof(int),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_lg.data(), d_lg, n_lg * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_conf.data(), d_conf, (size_t)bs * sizeof(float), cudaMemcpyDeviceToHost));

    int rc = 0;
    // (1) the sampled block must be bit-identical
    for (int i = 0; i <= bs; ++i) {
        if (got_ids[i] != ref_ids[i]) {
            printf("  [%s] ids[%d]: got %d, ref %d\n", name, i, got_ids[i], ref_ids[i]);
            rc = 1;
        }
    }
    // (2) the biased logits and the confidence only need f32 tolerance
    const float lg_md = maxdiff(got_lg, ref_lg);
    const float cf_md = maxdiff(got_conf, ref_conf);
    const float lg_scale = all_ties ? 1.f : 16.f;
    if (lg_md > 1e-3f * lg_scale) {
        printf("  [%s] biased logits maxdiff %.3e\n", name, lg_md);
        rc = 1;
    }
    if (cf_md > 1e-4f * (float)mr) {
        printf("  [%s] confidence maxdiff %.3e\n", name, cf_md);
        rc = 1;
    }
    printf("  [%s] vocab=%d mr=%d dim=%d bs=%d t=%.1f ties=%d -> ids=%d..%d lg_md=%.2e "
           "cf_md=%.2e %s\n",
           name, vocab, mr, dim, bs, (double)temperature, (int)all_ties, got_ids[0],
           got_ids[bs], (double)lg_md, (double)cf_md, rc ? "FAIL" : "ok");

    cudaFree(d_lg);
    cudaFree(d_h);
    cudaFree(d_me);
    cudaFree(d_mh);
    cudaFree(d_cp);
    cudaFree(d_conf);
    cudaFree(d_ids);
    cudaFree(d_part);
    cudaFree(d_ctr);
    return rc;
}

}  // namespace

int main() {
    int fails = 0;
    printf("dsv41 dspark markov head self-test\n");
    // small: the float4 arm of the per-warp dot
    fails += run_case("small-vec4", 37, 8, 6, 3, 0.0f, 11, false);
    // small: the scalar arm (mr % 4 != 0)
    fails += run_case("small-scalar", 29, 6, 5, 2, 1.0f, 3, false);
    // the whole vocabulary ties -> the lowest index must win at every step
    fails += run_case("all-ties", 41, 8, 4, 3, 1.0f, 7, true);
    // temperature 1 with real logits: the softmax path of ops::gumbel_argmax
    fails += run_case("small-temp1", 53, 12, 8, 4, 1.0f, 5, false);
    // FULL production shape (bs 5, vocab 129280, mr 256, dim 5120)
    fails += run_case("full", 129280, 256, 5120, 5, 1.0f, 1000, false);
    printf("%s\n", fails ? "FAILED" : "all cases passed");
    return fails ? 1 : 0;
}
