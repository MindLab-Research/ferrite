// tests_indexer_topk.cu — self-test for the P0-D streaming indexer top-k
// (`indexer_topk_heap_kernel` in ferrite_kernels.cu).
//
// WHAT IT CHECKS
//   1. SET EQUALITY between the three selectors, per row, sorted element-wise:
//        cpu   — host reference (same pool semantics, causal jmax, top-k by
//                (score desc, index asc))
//        old   — ferrite_indexer_topk (the O(select_k × t) slow path)
//        heap  — ferrite_indexer_topk_heap (the new streaming path)
//      This is the acceptance criterion: "新旧路径的集合一致".
//   2. ORDER: whether heap reproduces old's slot order byte-for-byte (it
//      should: both emit (score desc, index asc), and the scoring keeps the
//      old H-SPLIT's FP association). Reported as a WARN, not a FAIL, because
//      --use_fast_math may reassociate the two loop nestings by ~1 ulp; a
//      slot-order difference is only an fp artifact when the SCORES match.
//   3. TIES: a case whose pools all score exactly equal → the selection must
//      be {0, 1, ..., select_k-1} (the old `>` comparison keeps the lowest
//      index on ties). This pins the tie semantics.
//   4. SCALE: a 300K-pool row (≈1.2M tokens at kpool=4) so the streaming
//      chunk loop + the τ-prunes actually run (old kernel: smem overflows
//      around t > 2048, so only `heap` is checked there).
//
// BUILD (compile only, no GPU needed):
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//        kernels/cuda/tests_indexer_topk.cu -o /tmp/t_idx
// RUN (needs one GPU; the old path is skipped for t > 2048 by construction):
//   /tmp/t_idx            # exit code 0 = all hard checks passed
//
// The .cu is INCLUDED (not linked) so the test can never run against a stale
// libferrite_kernels.so — the same convention as tests_dsv41_attn.cu.
#include "ferrite_kernels.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

uint32_t g_rng = 0x1234abcd;
uint32_t xr() { g_rng = g_rng * 1664525u + 1013904223u; return g_rng; }
// wide, well-separated scores so a ~1 ulp difference can never reorder them
float frand() { return (float)((int)(xr() % 200001) - 100000) / 1000.f; }

int g_fail = 0;
int g_warn = 0;

void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok   %s\n", what.c_str());
    } else {
        printf("  FAIL %s\n", what.c_str());
        ++g_fail;
    }
}

// ---------------------------------------------------------------------------
// host reference: mirrors indexer_topk_kernel's pool semantics exactly
//   t = ceil(total/kpool); select_k = min(topk_max, t);
//   ctx0 = total - n_fixed; ctx0_pools = ctx0/kpool;
//   jmax(row) = min(ctx0_pools + row + 1, t);
//   score(i,j) = Σ_h w[i,h]·relu(q[i,h]·k[j]) · rsqrt(d)   (h-quarter sums, the
//   same association the kernel uses) ; top-k by (score desc, index asc)
// Selection is by the packed u64 key, so the CPU side shares the kernel's
// tie rule by construction.
// ---------------------------------------------------------------------------
std::vector<float> ref_scores(const std::vector<float>& qi, const std::vector<float>& ki,
                              const std::vector<float>& w, int row, int h, int d, int jmax) {
    const float inv = rsqrtf((float)d);
    const int nq = (h >= 4 && (h & 3) == 0) ? 4 : 1;
    const int hq = h / nq;
    std::vector<float> s((size_t)jmax, 0.f);
    for (int j = 0; j < jmax; ++j) {
        float acc = 0.f;
        for (int qq = 0; qq < nq; ++qq) {
            float part = 0.f;
            for (int hi = qq * hq; hi < (qq + 1) * hq; ++hi) {
                const float* qp = &qi[((size_t)row * h + hi) * d];
                const float* kp = &ki[(size_t)j * d];
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                for (int l = 0; l + 3 < d; l += 4) {
                    a0 += qp[l] * kp[l];
                    a1 += qp[l + 1] * kp[l + 1];
                    a2 += qp[l + 2] * kp[l + 2];
                    a3 += qp[l + 3] * kp[l + 3];
                }
                const float dot = (a0 + a1) + (a2 + a3);
                part += w[(size_t)row * h + hi] * std::max(dot, 0.f);
            }
            acc += part;
        }
        s[j] = acc * inv;
    }
    return s;
}

// one row's reference selection → the same [select_k] float convention the
// kernels emit (selected index, or -1 when fewer than select_k candidates)
std::vector<float> ref_row(const std::vector<float>& qi, const std::vector<float>& ki,
                           const std::vector<float>& w, int row, int h, int d, int topk_max,
                           int kpool, int total, int n_fixed) {
    const int t = (total + kpool - 1) / kpool;
    const int select_k = std::min(topk_max, t);
    const int ctx0 = total - n_fixed;
    const int jmax = std::min(ctx0 / kpool + row + 1, t);
    std::vector<float> out((size_t)select_k, -1.f);
    if (select_k <= 0) return out;
    const std::vector<float> sc = ref_scores(qi, ki, w, row, h, d, jmax);
    std::vector<int> ord((size_t)jmax);
    for (int j = 0; j < jmax; ++j) ord[j] = j;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (sc[a] != sc[b]) return sc[a] > sc[b]; // NaN would break this; see below
        return a < b;                             // ties → lower index
    });
    for (int r = 0; r < select_k; ++r) {
        const int j = (r < jmax) ? ord[r] : -1;
        out[r] = (j >= 0 && sc[j] == sc[j]) ? (float)j : -1.f;
    }
    return out;
}

std::vector<int> set_of(const std::vector<float>& row, int select_k) {
    std::vector<int> v;
    for (int r = 0; r < select_k && r < (int)row.size(); ++r)
        if (row[r] >= 0.f) v.push_back((int)row[r]);
    std::sort(v.begin(), v.end());
    return v;
}

std::string set_str(const std::vector<int>& v) {
    std::string s = "{";
    for (size_t i = 0; i < v.size(); ++i) {
        char b[24];
        snprintf(b, sizeof(b), "%s%d", i ? "," : "", v[i]);
        s += b;
    }
    return s + "}";
}

struct Case {
    const char* name;
    int n, h, d, t, kpool, topk_max, n_fixed;
    bool all_zero; // force every pool score to be exactly equal
    bool old_ok;   // may the OLD kernel be run? (t <= 2048, its frozen smem)
};

// ---------------------------------------------------------------------------
int run_case(const Case& c) {
    const int total = c.t * c.kpool; // exact: t = ceil(total/kpool)
    const int select_k = std::min(c.topk_max, c.t);
    printf("== case %s: n=%d h=%d d=%d t=%d kpool=%d topk_max=%d total=%d select_k=%d%s\n",
           c.name, c.n, c.h, c.d, c.t, c.kpool, c.topk_max, total, select_k,
           c.all_zero ? " (all-equal scores)" : "");

    const size_t nq = (size_t)c.n * c.h * c.d;
    const size_t nk = (size_t)c.t * c.d;
    const size_t nw = (size_t)c.n * c.h;
    std::vector<float> qi(nq), ki(nk), w(nw);
    for (size_t i = 0; i < nq; ++i) qi[i] = c.all_zero ? 0.f : frand();
    for (size_t i = 0; i < nk; ++i) ki[i] = c.all_zero ? 1.f : frand();
    // all_zero: ki = 1, qi = 0 → every dot is 0 → every score exactly 0.
    for (size_t i = 0; i < nw; ++i) w[i] = c.all_zero ? 1.f : frand();

    const size_t nidx = (size_t)c.n * c.topk_max;
    std::vector<float> idx_old(nidx, -12345.f), idx_heap(nidx, -12345.f);

    float *dqi = nullptr, *dki = nullptr, *dw = nullptr, *dold = nullptr, *dheap = nullptr;
    int* dtotal = nullptr;
    if (cudaMalloc(&dqi, nq * 4) != cudaSuccess) return 2;
    if (cudaMalloc(&dki, nk * 4) != cudaSuccess) return 2;
    if (cudaMalloc(&dw, nw * 4) != cudaSuccess) return 2;
    if (cudaMalloc(&dold, nidx * 4) != cudaSuccess) return 2;
    if (cudaMalloc(&dheap, nidx * 4) != cudaSuccess) return 2;
    if (cudaMalloc(&dtotal, 4) != cudaSuccess) return 2;
    cudaMemcpy(dqi, qi.data(), nq * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dki, ki.data(), nk * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), nw * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtotal, &total, 4, cudaMemcpyHostToDevice);
    cudaMemset(dold, 0, nidx * 4);
    cudaMemset(dheap, 0, nidx * 4);

    // ---- new path (the direct entry: the env gate is read once per process,
    //      so the test cannot rely on FERRITE_INDEXER_HEAP) ----
    cudaError_t eh = ferrite_indexer_topk_heap(dqi, dki, dw, dheap, c.n, c.h, c.d, c.topk_max,
                                               dtotal, c.kpool, c.n_fixed, 0);
    if (eh != cudaSuccess) {
        printf("  FAIL heap launch: %s\n", cudaGetErrorString(eh));
        ++g_fail;
        return 3;
    }
    eh = cudaDeviceSynchronize();
    if (eh != cudaSuccess) {
        printf("  FAIL heap sync: %s\n", cudaGetErrorString(eh));
        ++g_fail;
        return 3;
    }
    cudaMemcpy(idx_heap.data(), dheap, nidx * 4, cudaMemcpyDeviceToHost);

    // ---- old path (needs the whole-row score staging: t <= 2048) ----
    bool have_old = false;
    if (c.old_ok) {
        cudaError_t eo = ferrite_indexer_topk(dqi, dki, dw, dold, c.n, c.h, c.d, c.topk_max,
                                              dtotal, c.kpool, c.n_fixed, 0);
        if (eo != cudaSuccess) {
            printf("  FAIL old launch: %s\n", cudaGetErrorString(eo));
            ++g_fail;
            return 3;
        }
        eo = cudaDeviceSynchronize();
        if (eo != cudaSuccess) {
            printf("  FAIL old sync: %s\n", cudaGetErrorString(eo));
            ++g_fail;
            return 3;
        }
        cudaMemcpy(idx_old.data(), dold, nidx * 4, cudaMemcpyDeviceToHost);
        have_old = true;
    }

    // ---- per-row comparisons ----
    for (int row = 0; row < c.n; ++row) {
        std::vector<float> hrow(idx_heap.begin() + (size_t)row * c.topk_max,
                                idx_heap.begin() + (size_t)(row + 1) * c.topk_max);
        const std::vector<int> hset = set_of(hrow, select_k);
        const std::vector<float> rrow =
            ref_row(qi, ki, w, row, c.h, c.d, c.topk_max, c.kpool, total, c.n_fixed);
        const std::vector<int> rset = set_of(rrow, select_k);
        const std::string tag = "case " + std::string(c.name) + " row " + std::to_string(row);

        if (c.all_zero) {
            // every score is exactly 0 → the selection is the lowest select_k
            // indices, in that order
            std::vector<int> want((size_t)select_k);
            for (int i = 0; i < select_k; ++i) want[i] = i;
            check(hset == want, tag + " all-tie set == {0..select_k-1}" +
                                    (hset == want ? "" : " got " + set_str(hset)));
            static bool once = false;
            if (!once) {
                once = true;
                check(hrow.size() >= (size_t)select_k &&
                          (int)hrow[0] == 0 && (int)hrow[select_k - 1] == select_k - 1,
                      tag + " all-tie ORDER is 0,1,..,select_k-1 (index asc)");
            }
        } else {
            check(hset == rset, tag + " heap set == cpu reference set" +
                                    (hset == rset ? "" : " heap=" + set_str(hset) +
                                                                " cpu=" + set_str(rset)));
        }
        if (!hrow.empty() && (int)hrow[select_k - 1] < 0 && select_k > 0)
            check(false, tag + " slot select_k-1 is not a real index (-1)");
        if (have_old) {
            std::vector<float> orow(idx_old.begin() + (size_t)row * c.topk_max,
                                    idx_old.begin() + (size_t)(row + 1) * c.topk_max);
            const std::vector<int> oset = set_of(orow, select_k);
            check(hset == oset, tag + " heap set == old path set" +
                                    (hset == oset ? "" : " heap=" + set_str(hset) +
                                                           " old=" + set_str(oset)));
            bool ordered = true;
            int first_bad = -1;
            for (int r = 0; r < c.topk_max; ++r) {
                if (hrow[r] != orow[r]) { ordered = false; if (first_bad < 0) first_bad = r; }
            }
            // tail beyond select_k is the -1 padding in both paths
            for (int r = select_k; r < c.topk_max; ++r)
                if (hrow[r] != -1.f) { check(false, tag + " padding slot is not -1"); break; }
            if (!ordered) {
                ++g_warn;
                printf("  warn %s slot order differs from the old path (first at r=%d: heap=%g"
                       " old=%g) — only an fp artifact if the sets match\n",
                       tag.c_str(), first_bad, (double)hrow[std::max(first_bad, 0)],
                       (double)orow[std::max(first_bad, 0)]);
            }
        }
    }
    if (c.n > 0 && have_old) {
        check(memcmp(idx_old.data(), idx_heap.data(), nidx * 4) == 0,
              std::string("case ") + c.name + ": old and heap outputs are byte-identical "
                                             "(set + order + padding)");
    }
    cudaFree(dqi); cudaFree(dki); cudaFree(dw);
    cudaFree(dold); cudaFree(dheap); cudaFree(dtotal);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        printf("no CUDA device — this test needs one GPU\n");
        return 2;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("device: %s (sm_%d%d, %d SMs, %zu B smem/block)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.sharedMemPerBlockOptin);

    // ---- --gate-check: validate FERRITE_INDEXER_HEAP wiring in a FRESH
    // process (the launcher caches the gate on the first call, so this cannot
    // be folded into the A/B run below): the env must make ferrite_indexer_topk
    // take the heap path, byte-identical to the direct ferrite_indexer_topk_heap.
    if (argc > 1 && strcmp(argv[1], "--gate-check") == 0) {
        setenv("FERRITE_INDEXER_HEAP", "1", 1);
        const Case c{"gate", 2, 8, 16, 200, 4, 8, 2, false, false};
        const int total = c.t * c.kpool, select_k = std::min(c.topk_max, c.t);
        const size_t nq = (size_t)c.n * c.h * c.d, nk = (size_t)c.t * c.d, nw = (size_t)c.n * c.h;
        std::vector<float> qi(nq), ki(nk), w(nw);
        for (size_t i = 0; i < nq; ++i) qi[i] = frand();
        for (size_t i = 0; i < nk; ++i) ki[i] = frand();
        for (size_t i = 0; i < nw; ++i) w[i] = frand();
        const size_t nidx = (size_t)c.n * c.topk_max;
        std::vector<float> a(nidx, 0.f), b(nidx, 0.f);
        float *dqi, *dki, *dw, *da, *db; int* dt;
        cudaMalloc(&dqi, nq * 4); cudaMalloc(&dki, nk * 4); cudaMalloc(&dw, nw * 4);
        cudaMalloc(&da, nidx * 4); cudaMalloc(&db, nidx * 4); cudaMalloc(&dt, 4);
        cudaMemcpy(dqi, qi.data(), nq * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dki, ki.data(), nk * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), nw * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dt, &total, 4, cudaMemcpyHostToDevice);
        cudaMemset(da, 0, nidx * 4); cudaMemset(db, 0, nidx * 4);
        // gated entry (heap because the env says so)
        check(ferrite_indexer_topk(dqi, dki, dw, da, c.n, c.h, c.d, c.topk_max, dt, c.kpool,
                                   c.n_fixed, 0) == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess,
              "gate-check: gated ferrite_indexer_topk launches");
        check(ferrite_indexer_topk_heap(dqi, dki, dw, db, c.n, c.h, c.d, c.topk_max, dt, c.kpool,
                                        c.n_fixed, 0) == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess,
              "gate-check: direct heap entry launches");
        cudaMemcpy(a.data(), da, nidx * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), db, nidx * 4, cudaMemcpyDeviceToHost);
        check(memcmp(a.data(), b.data(), nidx * 4) == 0,
              "gate-check: FERRITE_INDEXER_HEAP=1 output == direct heap entry");
        for (int row = 0; row < c.n; ++row) {
            const std::vector<int> hset =
                set_of(std::vector<float>(a.begin() + (size_t)row * c.topk_max,
                                          a.begin() + (size_t)(row + 1) * c.topk_max), select_k);
            const std::vector<float> rrow =
                ref_row(qi, ki, w, row, c.h, c.d, c.topk_max, c.kpool, total, c.n_fixed);
            check(hset == set_of(rrow, select_k), "gate-check: row " + std::to_string(row) +
                                                      " set == cpu reference");
        }
        printf("\n%s: %d fail, %d warn\n", g_fail ? "FAILED" : "PASSED", g_fail, g_warn);
        return g_fail ? 1 : 0;
    }

    // The A/B below needs ferrite_indexer_topk to be the OLD path — pin the
    // gate OFF before the first call so an inherited FERRITE_INDEXER_HEAP in
    // the environment cannot silently turn the reference into the new path.
    setenv("FERRITE_INDEXER_HEAP", "0", 1);

    // n, h, d, t, kpool, topk_max, n_fixed, all_zero, old_ok
    //  A: small, slow path (select_k < jmax) — most rows, ties exercised by C
    run_case({"A-slow", 4, 8, 16, 64, 4, 16, 4, false, true});
    //  B: select_k >= jmax → the FAST path in both kernels (byte-equal)
    run_case({"B-fast", 4, 8, 16, 64, 4, 64, 4, false, true});
    //  C: many equal scores → the tie rule (lowest index first)
    run_case({"C-ties", 3, 8, 16, 128, 4, 24, 3, true, true});
    //  D: t = 2048 (the OLD kernel's exact staging ceiling) with a small
    //     select_k → the chunk loop runs 2 chunks and the running buffer
    //     crosses cap-chunk, i.e. the MIDDLE prune is exercised while the old
    //     path can still be used as the reference.
    run_case({"D-prune", 2, 4, 8, 2048, 4, 32, 2, false, true});
    //  E: h not a multiple of 4 (nq = 1 branch). The OLD kernel is NOT run:
    //     its H-split derives the quarter size as h>>2, which is 0 for h < 4,
    //     so it scores every pool 0 there (pre-existing bug, out of scope) —
    //     the cpu reference uses the same nq/hq rule as the heap kernel.
    run_case({"E-h2", 2, 2, 8, 300, 4, 12, 2, false, false});
    //  F: 300K pools (~1.2M tokens at kpool=4) — the streaming path only
    //     (the old kernel's frozen 2048-pool score staging would overflow)
    run_case({"F-300K", 1, 4, 8, 300000, 4, 512, 1, false, false});

    printf("\n%s: %d fail, %d warn\n", g_fail ? "FAILED" : "PASSED", g_fail, g_warn);
    return g_fail ? 1 : 0;
}
