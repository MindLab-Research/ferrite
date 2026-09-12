// tests_dsv41_argmax_rows.cu — the acceptance test for the VOCABULARY-SLICED
// verify head's cross-rank argmax (`dsv41_argmax_sliced_rows`,
// kernels/cuda/dsv41_kernels.cu).
//
// WHY THIS FILE EXISTS. `DSV41_VERIFY_HEAD_SLICED` (default ON) makes each TP
// rank project only its own `seg = vocab / world` rows of the replicated lm_head
// and then needs ONE global argmax per verify row. The read side is
// `argmax_kernel` per row (the slice's packed key carries the GLOBAL index) plus
// ONE `argmax_xchg_v5_rows_kernel` round that publishes all `rows` keys into
// every peer's current parity slot, stamps once, advances the shared v5 `epoch`
// once and maximises over the ranks. The properties that make that legal are
// exactly what this suite pins:
//
//  1. PARITY WITH THE PRODUCTION FULL-VOCABULARY ARGMAX. For every row the sliced
//     result must equal `dsv41_argmax_sliced(v_full, n = vocab, idx_off = 0,
//     world = 1)` — the SAME program the eager step's unsliced head ends in. This
//     is the whole correctness contract of the split (the head is replicated, so
//     the union of the slices IS the full vocabulary) and it covers the GLOBAL
//     index packing: a slice-local index leaking through would show up as an
//     off-by-`rank*seg` mismatch on every rank but 0.
//  2. THE LOWEST-INDEX TIE RULE ACROSS RANKS. The packed key is monotone in
//     (value, -index) and the reduce walks ranks ascending, so a tie must resolve
//     to the GLOBALLY lowest index — including a tie BETWEEN two ranks, which is
//     the case `argmax_sliced` itself can never see (world == 1). The cases below
//     include a deliberate all-equal row and a cross-rank neighbour tie.
//  3. ONE EPOCH ROUND PER BLOCK, NOT ONE PER ROW. `*epoch` must advance by
//     EXACTLY 1 for `rows` = 1, 3 and 6 alike. This is the AR-v5 deadlock
//     contract that forced the batching: every `dsv41_argmax_sliced` call IS an
//     unconditional epoch round (publish + stamp + advance + poll), so m rows
//     would advance the epoch m times where the single-row eager head advances it
//     once. Under v5 a rank that issues MORE rounds than a peer spins forever and
//     one that issues FEWER silently reads the previous round's staging (the
//     SEED_ALIGN `pos == 0` early-exit signature, need-cur = 3). The count is
//     asserted here so a future refactor that loops the single-row entry per row
//     fails this suite instead of hanging a serve.
//  4. THE DECLINE ARM. `rows * 8 > stride_bytes` must return 1 (the sentinel
//     `dsv41_argmax_sliced` also uses) and must NOT touch `*epoch` — the caller
//     (`verify_head_geom`) gates the geometry on `VERIFY_ROWS * 8 <= bytes`, and
//     the Rust side treats a decline as an inconsistency, so the entry must be
//     the side that declines cleanly rather than publishing outside its slot.
//
// The multi-rank protocol is replayed DETERMINISTICALLY in one process: every
// rank's staging is a real device buffer, the `[world]` pointer tables are real
// device arrays, and before each rank's call every rank's ready row is pre-stamped
// to `e + 1` and every rank's key block is pre-staged into the current parity
// half. The kernel's publish/stamp/advance then run for real and its poll passes
// immediately, so the reduce is exercised over the true union with no spinning.
//
// Build (needs nvcc, NO GPU — the entries under test live in dsv41_kernels.cu,
// which is linked as a SECOND input file; the suite #includes nothing). One line:
//   nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17
//        -o /tmp/t_argmax_rows kernels/cuda/tests_dsv41_argmax_rows.cu
//        kernels/cuda/dsv41_kernels.cu
// Run (needs ONE free GPU):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_argmax_rows [--quick]
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_fails = 0;

uint32_t rng = 20260912u;
uint32_t xr() { rng = rng * 1664525u + 1013904223u; return rng; }

// ---------------------------------------------------------------------------
// The two entries under test, exactly as device.rs declares them.
// ---------------------------------------------------------------------------
extern "C" int dsv41_argmax_sliced(const float* v, int n, int idx_off, int* out,
                                   unsigned long long* packed, int* pos_ctr,
                                   unsigned long long* const* staging_tbl,
                                   unsigned* const* ready_tbl, unsigned* epoch,
                                   unsigned long long* staging_local,
                                   const unsigned* ready_local, int world, int rank,
                                   long stride_bytes, cudaStream_t s);
extern "C" int dsv41_argmax_sliced_rows(
    const float* v, int n, int idx_off, int row_stride, int* out,
    unsigned long long* packed, int* pos_ctr, unsigned long long* const* staging_tbl,
    unsigned* const* ready_tbl, unsigned* epoch, unsigned long long* staging_local,
    const unsigned* ready_local, int world, int rank, int rows, long stride_bytes,
    cudaStream_t s);

constexpr int kRows = 6;  // VERIFY_ROWS (chain_dev.rs:84)

// One simulated rank: its staging (two parity halves of `world` slots each — the
// Collective's layout) lives in `stg`; `rdy` is its [world] stamp row and `ep` its
// device round counter. `stride` is the Collective's slot size in BYTES — the
// verify head is gated on `VERIFY_ROWS * 8 <= bytes`, so the full-shape case uses
// the production value and the boundary case the exact minimum.
struct Sim {
    int world = 0;
    long stride = 0;
    std::vector<unsigned long long*> stg;
    std::vector<unsigned*> rdy;
    std::vector<unsigned*> ep;
    unsigned long long** d_stg = nullptr;
    unsigned** d_rdy = nullptr;

    int init(int world_, long stride_bytes) {
        world = world_;
        stride = stride_bytes;
        stg.assign(world, nullptr);
        rdy.assign(world, nullptr);
        ep.assign(world, nullptr);
        for (int r = 0; r < world; r++) {
            if (cudaMalloc((void**)&stg[r], (size_t)2 * (size_t)world * (size_t)stride) !=
                cudaSuccess)
                return 1;
            if (cudaMemset(stg[r], 0, (size_t)2 * (size_t)world * (size_t)stride) !=
                cudaSuccess)
                return 1;
            if (cudaMalloc((void**)&rdy[r], (size_t)world * 4) != cudaSuccess) return 1;
            if (cudaMemset(rdy[r], 0, (size_t)world * 4) != cudaSuccess) return 1;
            if (cudaMalloc((void**)&ep[r], 4) != cudaSuccess) return 1;
            if (cudaMemset(ep[r], 0, 4) != cudaSuccess) return 1;
        }
        if (cudaMalloc((void**)&d_stg, (size_t)world * sizeof(void*)) != cudaSuccess)
            return 1;
        if (cudaMalloc((void**)&d_rdy, (size_t)world * sizeof(void*)) != cudaSuccess)
            return 1;
        if (cudaMemcpy(d_stg, stg.data(), (size_t)world * sizeof(void*),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return 1;
        if (cudaMemcpy(d_rdy, rdy.data(), (size_t)world * sizeof(void*),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return 1;
        return 0;
    }

    void free_all() {
        for (int r = 0; r < world; r++) {
            cudaFree(stg[r]);
            cudaFree(rdy[r]);
            cudaFree(ep[r]);
        }
        if (d_stg) cudaFree(d_stg);
        if (d_rdy) cudaFree(d_rdy);
    }

    // Pre-place rank `r`'s key block (its `rows` packed keys) into EVERY rank's
    // CURRENT parity half (parity 0 == round 0), i.e. the state every peer is in
    // after its own publish. The address is the kernel's: slot `r` of parity 0 is
    // `r * stride` bytes into any staging base, key i at `+ i * 8`.
    int prestage(int r, const std::vector<unsigned long long>& keys) {
        for (int s = 0; s < world; s++) {
            char* dst = reinterpret_cast<char*>(stg[s]) + (size_t)r * (size_t)stride;
            if (cudaMemcpy(dst, keys.data(), keys.size() * 8, cudaMemcpyHostToDevice) !=
                cudaSuccess)
                return 1;
        }
        return 0;
    }

    // Set every rank's ready row to `stamp` == "every peer already finished the
    // previous round", so the kernel's poll passes on its first probe.
    int prestamp(unsigned stamp) {
        std::vector<unsigned> row(world, stamp);
        for (int r = 0; r < world; r++)
            if (cudaMemcpy(rdy[r], row.data(), (size_t)world * 4,
                           cudaMemcpyHostToDevice) != cudaSuccess)
                return 1;
        return 0;
    }

    unsigned epoch_of(int r) const {
        unsigned e = 0xdeadbeefu;
        cudaMemcpy(&e, ep[r], 4, cudaMemcpyDeviceToHost);
        return e;
    }
};

// The production full-vocabulary reference for one row, through the program the
// eager step's unsliced head ends in (`dsv41_argmax_sliced`, world == 1).
int full_vocab_argmax(const std::vector<float>& row, int* out) {
    Sim s1;
    if (s1.init(1, (long)row.size() * 4)) return 1;
    float* d_v = nullptr;
    unsigned long long* d_pk = nullptr;
    int* d_out = nullptr;
    if (cudaMalloc((void**)&d_v, row.size() * 4) != cudaSuccess) return 1;
    if (cudaMalloc((void**)&d_pk, 8) != cudaSuccess) return 1;
    if (cudaMalloc((void**)&d_out, 4) != cudaSuccess) return 1;
    if (cudaMemcpy(d_v, row.data(), row.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
        return 1;
    const int rc = dsv41_argmax_sliced(d_v, (int)row.size(), 0, d_out, d_pk, nullptr,
                                       s1.d_stg, s1.d_rdy, s1.ep[0], s1.stg[0], s1.rdy[0], 1,
                                       0, (long)row.size() * 4, 0);
    if (rc != 0) {
        printf("    reference dsv41_argmax_sliced rc=%d\n", rc);
        return 1;
    }
    if (cudaMemcpy(out, d_out, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    cudaFree(d_v);
    cudaFree(d_pk);
    cudaFree(d_out);
    s1.free_all();
    return 0;
}

// ---------------------------------------------------------------------------
// One case: `world` ranks of `seg` rows each, `rows` verify rows.
// ---------------------------------------------------------------------------
int case_sliced(int world, int seg, int rows, bool all_ties, bool cross_rank_tie,
                long stride_bytes, bool decline_arm) {
    const int vocab = world * seg;
    printf("  world=%d seg=%d vocab=%d rows=%d stride=%ld%s%s%s\n", world, seg, vocab, rows,
           stride_bytes, all_ties ? " all-ties" : "", cross_rank_tie ? " cross-rank-tie" : "",
           decline_arm ? " decline-arm" : "");

    // ---- the logits: rank r owns the global range [r*seg, (r+1)*seg) ---------
    std::vector<std::vector<float>> full(rows, std::vector<float>(vocab, 0.f));
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < vocab; k++)
            // A deliberately coarse value grid: values collide often, so ties are
            // the COMMON case rather than a corner (the near-tie regime the packed
            // key must order correctly).
            full[r][k] = (float)((int)(xr() % 17) - 8) * 0.25f;
    if (all_ties)
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < vocab; k++) full[r][k] = 1.5f;
    if (cross_rank_tie) {
        // Row 0: the SAME value at global index 1 (rank 0) and `seg + 1` (rank 1),
        // higher than everything else -> rank 0 must win (lowest GLOBAL index).
        for (int k = 0; k < vocab; k++) full[0][k] = 0.f;
        full[0][1] = 3.f;
        if (world > 1) full[0][seg + 1] = 3.f;
        // Row 1: the same value inside rank 1's slice only (an ordinary tie).
        for (int k = 0; k < vocab; k++) full[1][k] = 0.f;
        if (world > 1) {
            full[1][seg] = 2.f;
            full[1][2 * seg - 1] = 2.f;
        }
    }
    std::vector<std::vector<float>> slice(world, std::vector<float>(rows * seg, 0.f));
    for (int k = 0; k < vocab; k++)
        for (int r = 0; r < world; r++) slice[r][k % seg] = full[k / seg][k];

    // ---- the expected: the production full-vocabulary argmax per row ---------
    std::vector<int> want(rows, -1);
    for (int r = 0; r < rows; r++)
        if (full_vocab_argmax(full[r], &want[r])) return 1;

    // ---- the simulation -----------------------------------------------------
    Sim s;
    if (s.init(world, stride_bytes)) return 1;
    std::vector<float*> d_v(world, nullptr);
    std::vector<unsigned long long*> d_pk(world, nullptr);
    std::vector<int*> d_out(world, nullptr);
    for (int r = 0; r < world; r++) {
        if (cudaMalloc((void**)&d_v[r], (size_t)rows * seg * 4) != cudaSuccess) return 1;
        if (cudaMalloc((void**)&d_pk[r], (size_t)rows * 8) != cudaSuccess) return 1;
        if (cudaMalloc((void**)&d_out[r], (size_t)rows * 4) != cudaSuccess) return 1;
        if (cudaMemcpy(d_v[r], slice[r].data(), (size_t)rows * seg * 4,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return 1;
    }
    const unsigned e0 = 0;  // parity (e & 1) == 0 for the first round

    if (decline_arm) {
        // The entry must decline BEFORE publishing: epoch and `out` untouched.
        std::vector<int> before(rows, 0x7f7f7f7f);
        cudaMemcpy(d_out[0], before.data(), (size_t)rows * 4, cudaMemcpyHostToDevice);
        if (s.prestamp(e0 + 1)) return 1;
        const int rc = dsv41_argmax_sliced_rows(
            d_v[0], seg, 0, seg, d_out[0], d_pk[0], nullptr, s.d_stg, s.d_rdy, s.ep[0],
            s.stg[0], s.rdy[0], world, 0, rows, (long)rows * 8 - 1, 0);
        if (rc != 1) {
            printf("    FAIL: the decline arm returned %d, expected the 1 sentinel\n", rc);
            ++g_fails;
        }
        if (s.epoch_of(0) != e0) {
            printf("    FAIL: the decline arm advanced the epoch to %u\n", s.epoch_of(0));
            ++g_fails;
        }
        std::vector<int> after(rows, 0);
        cudaMemcpy(after.data(), d_out[0], (size_t)rows * 4, cudaMemcpyDeviceToHost);
        if (after != before) {
            printf("    FAIL: the decline arm wrote `out`\n");
            ++g_fails;
        }
        for (int r = 0; r < world; r++) {
            cudaFree(d_v[r]);
            cudaFree(d_pk[r]);
            cudaFree(d_out[r]);
        }
        s.free_all();
        return 0;
    }

    // ---- per-row local reduce, once per rank (the head loop's first half) ----
    // Each rank's own `argmax_kernel` pass over its slice, with the GLOBAL index
    // offset — run through the entry with world == 1 so the exchange is trivial
    // and only the slice's packed key is observed.
    std::vector<std::vector<unsigned long long>> keys(world,
                                                      std::vector<unsigned long long>(rows, 0));
    Sim one;
    if (one.init(1, stride_bytes)) return 1;
    for (int r = 0; r < world; r++) {
        const int rc = dsv41_argmax_sliced_rows(d_v[r], seg, r * seg, seg, d_out[r], d_pk[r],
                                                nullptr, one.d_stg, one.d_rdy, one.ep[0],
                                                one.stg[0], one.rdy[0], 1, 0, rows,
                                                stride_bytes, 0);
        if (rc != 0) {
            printf("    FAIL: the local reduce of rank %d returned %d\n", r, rc);
            ++g_fails;
            return 1;
        }
        if (cudaMemcpy(keys[r].data(), d_pk[r], (size_t)rows * 8, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return 1;
    }

    // ---- the REAL exchange: every rank, over the pre-staged union ------------
    for (int r = 0; r < world; r++)
        if (s.prestage(r, keys[r])) return 1;
    if (s.prestamp(e0 + 1)) return 1;
    for (int r = 0; r < world; r++) {
        const int rc = dsv41_argmax_sliced_rows(d_v[r], seg, r * seg, seg, d_out[r], d_pk[r],
                                                nullptr, s.d_stg, s.d_rdy, s.ep[r], s.stg[r],
                                                s.rdy[r], world, r, rows, stride_bytes, 0);
        if (rc != 0) {
            printf("    FAIL: the exchange of rank %d returned %d\n", r, rc);
            ++g_fails;
            return 1;
        }
    }
    for (int r = 0; r < world; r++) {
        // ★ THE FOOTPRINT CONTRACT: one round for the WHOLE block, whatever `rows`.
        if (s.epoch_of(r) != e0 + 1) {
            printf("    FAIL: rank %d epoch=%u, expected exactly one round (%u) for rows=%d\n",
                   r, s.epoch_of(r), e0 + 1, rows);
            ++g_fails;
        }
    }
    for (int r = 0; r < world; r++) {
        std::vector<int> got(rows, -1);
        if (cudaMemcpy(got.data(), d_out[r], (size_t)rows * 4, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return 1;
        for (int i = 0; i < rows; i++) {
            if (got[i] != want[i]) {
                printf("    FAIL: rank %d row %d got %d, the full-vocabulary argmax is %d\n",
                       r, i, got[i], want[i]);
                ++g_fails;
            }
        }
    }

    // ---- the epoch contract on its own one-rank setup (rows = 1, 3, 6) -------
    for (int trial = 0; trial < 3; trial++) {
        const int trial_rows = (trial == 0) ? 1 : (trial == 1 ? 3 : kRows);
        Sim t;
        if (t.init(1, stride_bytes)) return 1;
        const int rc = dsv41_argmax_sliced_rows(
            d_v[0], seg, 0, seg, d_out[0], d_pk[0], nullptr, t.d_stg, t.d_rdy, t.ep[0],
            t.stg[0], t.rdy[0], 1, 0, trial_rows, stride_bytes, 0);
        if (rc != 0) {
            printf("    FAIL: the epoch arm rows=%d returned %d\n", trial_rows, rc);
            ++g_fails;
        } else if (t.epoch_of(0) != 1) {
            printf("    FAIL: rows=%d advanced the epoch to %u — the batching contract is\n"
                   "          EXACTLY ONE round per block (m rounds is the AR-v5 deadlock)\n",
                   trial_rows, t.epoch_of(0));
            ++g_fails;
        }
        t.free_all();
    }

    one.free_all();
    for (int r = 0; r < world; r++) {
        cudaFree(d_v[r]);
        cudaFree(d_pk[r]);
        cudaFree(d_out[r]);
    }
    s.free_all();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--quick") == 0) quick = true;

    printf("dsv41 argmax_sliced_rows self-test (slice + ONE-round exchange == full-vocab)\n");
    // The AR payload is `dim` floats (dim = 5120 -> 20480 bytes); the verify head
    // is gated on `VERIFY_ROWS * 8 <= bytes`, so this is the slot the head really
    // runs in.
    const long prod_stride = 5120L * 4;

    // Small shapes: every tail/parity case stays cheap to read on a failure.
    case_sliced(8, 16, kRows, false, false, prod_stride, false);
    case_sliced(8, 5, kRows, false, false, prod_stride, false);
    case_sliced(4, 7, 3, false, false, prod_stride, false);
    case_sliced(2, 3, 1, false, false, prod_stride, false);
    // The tie rules (cross-rank ties are the case world == 1 cannot see).
    case_sliced(8, 16, kRows, true, false, prod_stride, false);
    case_sliced(8, 16, kRows, false, true, prod_stride, false);
    // The slot BOUNDARY: exactly `rows * 8`, then one byte less (the decline arm).
    case_sliced(4, 16, kRows, false, false, (long)kRows * 8, false);
    case_sliced(8, 16, kRows, false, false, (long)kRows * 8 - 1, true);

    if (!quick) {
        // The production shape: vocab 129280 = 8 x 16160, 6 verify rows — the
        // buffer `verify_head_geom` actually hands the entry (logits_r's row pitch
        // is `seg`), so the addressing runs at its real width.
        case_sliced(8, 16160, kRows, false, false, prod_stride, false);
        case_sliced(8, 16160, kRows, false, true, prod_stride, false);
    }

    if (g_fails == 0) {
        printf("all cases passed\n");
        return 0;
    }
    printf("%d FAILURES\n", g_fails);
    return 1;
}
