// GPU self-test for `dsv41_verify_ring_win`: the m-row block append + the
// per-row CAUSAL window indices.
//
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
//            tests_dsv41_verify_rw.cu -o /tmp/t_vrw
// Run:   CUDA_VISIBLE_DEVICES=0 ./t_vrw
#include "dsv41_glue.cu"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;

// Host reference, mirroring window_idxs_kernel's decode branch per row.
static void ref_row(std::vector<int32_t>& row, int base_pos, int window) {
    for (int c = 0; c < window; ++c) {
        if (base_pos == 0) { row[c] = (c == 0) ? 0 : -1; continue; }
        const int oldest = (base_pos % window) + 1;
        long long v = ((long long)c < (long long)window - oldest)
                          ? (long long)oldest + c
                          : (long long)c - ((long long)window - oldest);
        if (v > (long long)base_pos) v = -1;
        row[c] = (int)v;
    }
}

static void case_causal(int window, int m, int base) {
    // device buffers
    float* ring; int* pos_ctr; int32_t* idxs; float* kv;
    cudaMalloc(&ring, (size_t)window * 64 * sizeof(float));
    cudaMalloc(&pos_ctr, 4);
    cudaMalloc(&idxs, (size_t)m * window * sizeof(int32_t));
    cudaMalloc(&kv, (size_t)m * 64 * sizeof(float));
    cudaMemset(ring, 0, (size_t)window * 64 * sizeof(float));
    cudaMemset(idxs, 0xAB, (size_t)m * window * sizeof(int32_t));  // poison
    cudaMemcpy(pos_ctr, &base, 4, cudaMemcpyHostToDevice);
    std::vector<float> kvh((size_t)m * 64);
    for (size_t i = 0; i < kvh.size(); ++i) kvh[i] = (float)(i + 1);
    cudaMemcpy(kv, kvh.data(), kvh.size() * sizeof(float), cudaMemcpyHostToDevice);

    dsv41_verify_ring_win(ring, kv, pos_ctr, window, 64, m, idxs, 0);

    // indices: row r's causal window ends at base + r
    std::vector<int32_t> got((size_t)m * window), exp((size_t)m * window);
    cudaMemcpy(got.data(), idxs, got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int r = 0; r < m; ++r) ref_row(std::vector<int32_t>(exp.begin() + (size_t)r * window, exp.begin() + (size_t)((size_t)r + 1) * window), base + r, window), 0;
    // (the above comma trick does not work; redo plainly)
    for (int r = 0; r < m; ++r) {
        std::vector<int32_t> row((size_t)window);
        ref_row(row, base + r, window);
        for (int c = 0; c < window; ++c) exp[(size_t)r * window + c] = row[c];
    }
    bool idx_ok = got == exp;
    if (!idx_ok) {
        ++failures;
        printf("  [causal w=%d m=%d base=%d] IDX MISMATCH\n", window, m, base);
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < 12 && c < window; ++c)
                printf("    r%d c%d got=%d exp=%d\n", r, c, got[(size_t)r * window + c], exp[(size_t)r * window + c]);
        }
    }

    // append: row j lands in slot (base + j) % window
    std::vector<float> ringh((size_t)window * 64);
    cudaMemcpy(ringh.data(), ring, ringh.size() * sizeof(float), cudaMemcpyDeviceToHost);
    bool app_ok = true;
    for (int j = 0; j < m; ++j) {
        const int slot = (base + j) % window;
        for (int c = 0; c < 64; ++c) {
            const float want = kvh[(size_t)j * 64 + c];
            if (ringh[(size_t)slot * 64 + c] != want) { app_ok = false; ++failures;
                printf("  [causal w=%d m=%d base=%d] APPEND MISMATCH slot=%d c=%d got=%f want=%f\n",
                       window, m, base, slot, c, ringh[(size_t)slot * 64 + c], want);
                break; }
        }
        if (!app_ok) break;
    }
    printf("  [causal w=%d m=%d base=%d] idx=%s append=%s\n", window, m, base,
           idx_ok ? "ok" : "FAIL", app_ok ? "ok" : "FAIL");
    cudaFree(ring); cudaFree(pos_ctr); cudaFree(idxs); cudaFree(kv);
}

int main() {
    printf("dsv41 verify_ring_win self-test\n");
    case_causal(128, 6, 0);      // prefill edge: base 0
    case_causal(128, 6, 1);      // earliest decode
    case_causal(128, 6, 127);    // window boundary -1
    case_causal(128, 6, 128);    // exactly one full lap
    case_causal(128, 6, 1000);   // steady state
    case_causal(32, 6, 31);      // small window boundary
    case_causal(8, 6, 7);        // window barely > m
    if (failures == 0) printf("all cases passed\n");
    else { printf("%d FAILURES\n", failures); return 1; }
    return 0;
}
