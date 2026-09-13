// tests_dsv41_draft_parity.cu — the bs = 5 PARITY SUITE for every "BIT-IDENTICAL"
// fold on the draft (MTP / DSpark) chain.
//
// WHY THIS FILE EXISTS. The draft chain carries a stack of folds that all make
// the SAME claim — "the folded launch produces the same raw f32 bits as the
// launch sequence it replaces" — and NOT ONE of them has a measured receipt on
// the REAL draft shape (bs = 5). The draft-gate-diff verdict calls them the
// "unmeasured items":
//
//   P3A  a1..a4  (docs/agent/draft-p3-fusion.md §5)   launch-form folds
//   P3LITE l1..l4 (docs/agent/draft-p3lite-segment-fusion.md §2.3/§2.4 and
//                  docs/agent/p3lite-segment-b-k2-verify-manual.md §5)  the
//                  attention half's same-program fusions
//   DRAFT_MOE_MROWS (DSV41_DRAFT_MOE_MROWS)            routed experts rows = bs
//   DRAFT_HEAD_FOLD v1 (DSV41_DRAFT_HEAD_FOLD)         head row-fold
//
// P3B is NOT here: it was condemned as part of a four-way arm (never isolated
// by a single-variable cut, see draft-p3lite-segment-fusion.md §3.1). The one
// P3B item that still has a live env name — `DSV41_DRAFT_MOE_MROWS` (b2) — IS
// covered, because the routed-experts row fold is a pure geometry claim
// (rows only shift base pointers) and it is exactly the class of claim the P3B
// post-mortem said was never measured.
//
// WHAT EACH ARM MEASURES. Every arm runs BOTH paths IN ONE PROCESS (the
// `tests_dsv41_gemm_mrows.cu` pattern: the choice of kernel is made by the TEST,
// not by an env, so the two runs are two call sites on the same operands) and
// compares the outputs as RAW f32/i8 BITS — memcmp over uint32_t / uint8_t, so
// ±0.0, NaN payloads and a single last-bit flip all count as differences. This
// is exact equality, NOT a tolerance. A real fold is "bit-identical or it is
// not the fold it claims to be".
//
// A full miss on this suite means the gate may be defaulted ON. A single diff
// names the arm, the element index and both bit patterns.
//
// ---------------------------------------------------------------------------
// Build (needs nvcc, NO GPU to compile/link; the GPU run is the caller's):
//
//   # dsv41_kernels.cu is the INCLUDED TU (it carries the file-scope consts the
//   # skip logic reads); dsv41_glue.cu / ferrite_kernels.cu /
//   # dsv41_experts_mxf4.cu are LINKED as separate TUs, exactly as their own
//   # test files do (see tests_dsv41_head_mrows.cu, tests_dsv41_experts_mrows.cu).
//   CU=~/.local/lib/python3.10/site-packages/nvidia/cu13
//   /tmp/nvccx/nvidia/cu13/bin/nvcc -I$CU/include -L$CU/lib \
//        -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
//        -o /tmp/t_draft_parity kernels/cuda/tests_dsv41_draft_parity.cu \
//        kernels/cuda/dsv41_glue.cu kernels/cuda/ferrite_kernels.cu \
//        kernels/cuda/dsv41_experts_mxf4.cu
//
// Run (needs ONE free GPU; peak allocation ~250 MB — the MoE arm carries the
// 8-expert fp4 pools, the head arm a 4096x5120 bf16 weight):
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_draft_parity              # every arm
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_draft_parity --quick      # skips the
//                                three heavy arms (moe / head / k2)
// Exit status: 0 = every arm bit-identical, 1 = at least one differed.
//
// ENV THAT CHANGES WHAT IS COVERED (read once per process, before main):
//   DSV41_GEMV_FP8_MODE  must be >= 3 for the l4 / K2 arm (the mrows launchers
//                        decline mode 0/1, which reorder a lane's elements).
//                        Default 4. The arm reports SKIP when declined.
//   DSV41_NO_GEMV_FP8    set => the m=1 GEMV is off, so l4 reports SKIP.
//   DSV41_ATTN_SEQ / DSV41_ATTN_PF=0  make the l3 orope path decline (the plain
//                        launcher picks a different kernel); the arm reports SKIP.
//   DSV41_SPARSE_SPLIT / DSV41_ATTN_SPLIT  change the key-split count. Both arms
//                        read the SAME resolver in one process, so the parity is
//                        invariant — worth running under a second value.
//
// The dspark geometry mirrored here (crates/ferrite-models/src/dsv41/config.rs::
// production()): dim 5120 / hd 512 / nh 64 / q_lora 1280 / o_groups 8 / win 128 /
// hc 4 / moe_inter 2304 / n_activated 6 / bs 5 / rope_head_dim 64.
//
// AUTHOR'S NOTE ON SCOPE. The a2 / a3 items are NOT kernel folds: they retire a
// `memcpy_d2d` by aliasing pointers (a2: the two `hc_post`s write into each
// other's buffer; a3: the premix ping-pongs). Their arms therefore compare the
// BUFFER STATE the fold produces (the residual stream `h` after both sub-blocks,
// and the premix the next `hc_collapse` consumes) rather than a kernel's output.
// That is the whole claim for those two, and it is cheap to pin.
#include "dsv41_kernels.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// --------------------------------------------------------------------------
// The LINKED TUs' entry points (dsv41_glue.cu / ferrite_kernels.cu /
// dsv41_experts_mxf4.cu). Declared here because those files are inputs on the
// nvcc command line, not #included.
// --------------------------------------------------------------------------
extern "C" int dsv41_hc_collapse(const float* x, const float* pre, float* out, int rows, int hc,
                                 int dim, cudaStream_t s);
extern "C" int dsv41_gemv_bf16(const void* w, const float* x, float* out, int n, int k,
                               cudaStream_t s);
extern "C" int dsv41_gemv_bf16_v1_mrows(const void* w, const float* x, float* out, int m, int n,
                                        int k, cudaStream_t s);
extern "C" int dsv41_swiglu_limit_batched(float* gate_up, int rows, int inter, float limit,
                                          long slot_stride, int slots, cudaStream_t s);
extern "C" int dsv41_expert_gate_up_fp4_batched(
    const uint8_t* a, const float* a_scale, float* out, long out_slot_stride, int rows, int dim,
    int inter, float limit, int slots, const uint8_t* w1_base, long w1_stride,
    const uint8_t* w1s_base, long w1s_stride, const uint8_t* w3_base, long w3_stride,
    const uint8_t* w3s_base, long w3s_stride, const int* ids, int ilv, int act_e4m3,
    cudaStream_t stream);
extern "C" int dsv41_expert_down_reduce_fp4_batched(
    const float* act_base, long act_stride, float* out, int rows, int dim, int inter,
    const float* row_weight, long rw_stride, int slots, const uint8_t* w2_base, long w2_stride,
    const uint8_t* w2s_base, long w2s_stride, const int* ids, cudaStream_t stream);
extern "C" cudaError_t ferrite_rmsnorm(const float* x, const float* w, float* out, int n, int dim,
                                       float eps, cudaStream_t s);
extern "C" cudaError_t ferrite_hc_post(const float* x, const float* res, const float* post,
                                       const float* comb, float* out, int s, int n, int h,
                                       cudaStream_t stream);

namespace {

// ============================================================ test utilities
int g_fails = 0;
int g_skips = 0;

#define DP_CHECK(expr, fmt, ...)                                                       \
    do {                                                                               \
        if (!(expr)) {                                                                 \
            printf("    FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);     \
            ++g_fails;                                                                 \
        }                                                                              \
    } while (0)

uint32_t g_rng = 20260913u;
uint32_t dp_xr() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
int dp_irand(int n) { return (int)(dp_xr() % (uint32_t)n); }
float dp_frand(float lo, float hi) {
    return lo + (hi - lo) * (float)(dp_xr() >> 8) / (float)(1u << 24);
}
void dp_fill(std::vector<float>& v, float lo, float hi) {
    for (auto& x : v) x = dp_frand(lo, hi);
}

// Raw-bits compare. Returns true when identical; on a diff prints the first
// index (and the count of differing elements) so a log names the arm and the
// element without a second run.
bool dp_cmp_f32(const float* a, const float* b, size_t n, const char* tag) {
    size_t first = (size_t)-1, ndiff = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t x, y;
        std::memcpy(&x, a + i, 4);
        std::memcpy(&y, b + i, 4);
        if (x != y) {
            if (first == (size_t)-1) first = i;
            ++ndiff;
        }
    }
    if (ndiff == 0) return true;
    uint32_t x, y;
    std::memcpy(&x, a + first, 4);
    std::memcpy(&y, b + first, 4);
    printf("    FAIL [%s] %zu/%zu element(s) differ; first at %zu: A 0x%08x (%g)  B 0x%08x (%g)\n",
           tag, ndiff, n, first, x, (double)a[first], y, (double)b[first]);
    ++g_fails;
    return false;
}

bool dp_cmp_u8(const uint8_t* a, const uint8_t* b, size_t n, const char* tag) {
    size_t first = (size_t)-1, ndiff = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (first == (size_t)-1) first = i;
            ++ndiff;
        }
    }
    if (ndiff == 0) return true;
    printf("    FAIL [%s] %zu/%zu byte(s) differ; first at %zu: A 0x%02x  B 0x%02x\n", tag, ndiff, n,
           first, a[first], b[first]);
    ++g_fails;
    return false;
}

// ---- device-memory plumbing -------------------------------------------------
struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
    bool alloc(size_t n) {
        bytes = n ? n : 4;
        return cudaMalloc(&p, bytes) == cudaSuccess;
    }
    void free() {
        if (p) cudaFree(p);
        p = nullptr;
    }
    void* add(size_t off) const { return (void*)((char*)p + off); }
};

bool dp_upload(Buf& d, const void* h, size_t n) {
    return d.alloc(n) && cudaMemcpy(d.p, h, n, cudaMemcpyHostToDevice) == cudaSuccess;
}
template <typename T>
bool dp_download(std::vector<T>& h, const Buf& d, size_t bytes) {
    h.resize(bytes / sizeof(T));
    return cudaMemcpy(h.data(), d.p, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

// Enough free device memory? (do not fight a co-tenant serve: SKIP, not FAIL)
bool dp_have_free(size_t need) {
    size_t fb = 0, tb = 0;
    if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) return true;
    if (fb < need) {
        printf("    SKIP: needs ~%.1f MB free, %.1f MB available\n", (double)need / 1048576.0,
               (double)fb / 1048576.0);
        ++g_skips;
        return false;
    }
    return true;
}

// ========================================================= the draft geometry
constexpr int kBm = 5;        // bs  (the draft block size)
constexpr int kDim = 5120;    // dim
constexpr int kHd = 512;      // head_dim
constexpr int kNh = 64;       // n_heads
constexpr int kRd = 64;       // rope_head_dim
constexpr int kHalf = kRd / 2;
constexpr int kQl = 1280;     // q_lora_rank
constexpr int kHc = 4;        // hc
constexpr int kInter = 2304;  // moe_inter_dim
constexpr int kSlots = 6;     // n_activated_experts (routed top-k)
constexpr int kWin = 128;     // window_size
constexpr float kEps = 1e-6f;
// Positions: keep pos_dev == rope_pos (the l1..l4 sites' invariant) and the
// table large enough for every t = pos + r / kv_pos + r / P0 + r.
constexpr int kPos = 200;   // the anchor / rope base
constexpr int kP0 = 100;    // the a4 / l4 pos_rows base
constexpr int kTabRows = 512;
constexpr size_t kTabN = (size_t)kTabRows * kHalf;

// Deterministic rotation tables (both arms read the same values; the numbers
// only have to be finite and bounded).
void dp_rope_tables(std::vector<float>& cos, std::vector<float>& sin) {
    cos.resize(kTabN);
    sin.resize(kTabN);
    for (int t = 0; t < kTabRows; ++t)
        for (int i = 0; i < kHalf; ++i) {
            const double th = 0.013 * (double)t + 0.11 * (double)i;
            cos[(size_t)t * kHalf + i] = (float)std::cos(th);
            sin[(size_t)t * kHalf + i] = (float)std::sin(th);
        }
}

// =====================================================================
// Arm P3A a1 — `dsv41_hc_collapse_norm`  vs  `hc_collapse` + `rmsnorm`
//
// ON : hc_collapse_norm(h, pre, w, out)                (one launch)
// OFF: dsv41_hc_collapse(h, pre, collapsed) ; ferrite_rmsnorm(collapsed, w, out)
//
// `h` is read-only on both paths (the fused kernel normalises in its OUT
// buffer), so one copy of the input feeds both arms. The comparison is over
// `out` = [bs, dim].
// =====================================================================
int dp_arm_p3a_a1() {
    const char* tag = "P3A-a1";
    const int rows = kBm, hc = kHc, dim = kDim;
    std::vector<float> hx((size_t)rows * hc * dim), hpre((size_t)rows * hc), hw(dim);
    dp_fill(hx, -1.5f, 1.5f);
    dp_fill(hpre, -1.0f, 1.0f);
    dp_fill(hw, 0.4f, 1.6f);

    Buf dx, dpre, dw, dout_on, dout_off, dcol;
    bool ok = dp_upload(dx, hx.data(), hx.size() * 4) &&
              dp_upload(dpre, hpre.data(), hpre.size() * 4) &&
              dp_upload(dw, hw.data(), hw.size() * 4) && dout_on.alloc((size_t)rows * dim * 4) &&
              dout_off.alloc((size_t)rows * dim * 4) && dcol.alloc((size_t)rows * dim * 4);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;

    // ON
    const int rc_on = dsv41_hc_collapse_norm((float*)dx.p, (const float*)dpre.p, (const float*)dw.p,
                                             (float*)dout_on.p, rows, hc, dim, kEps, 0, nullptr);
    DP_CHECK(rc_on == 0, "[%s] hc_collapse_norm rc=%d (%s)", tag, rc_on,
             cudaGetErrorString((cudaError_t)rc_on));
    // OFF
    const int rc_c = dsv41_hc_collapse((const float*)dx.p, (const float*)dpre.p, (float*)dcol.p,
                                       rows, hc, dim, nullptr);
    DP_CHECK(rc_c == 0, "[%s] hc_collapse rc=%d", tag, rc_c);
    const cudaError_t rc_n = ferrite_rmsnorm((const float*)dcol.p, (const float*)dw.p,
                                             (float*)dout_off.p, rows, dim, kEps, nullptr);
    DP_CHECK(rc_n == cudaSuccess, "[%s] ferrite_rmsnorm: %s", tag, cudaGetErrorString(rc_n));

    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> on, off;
    dp_download(on, dout_on, (size_t)rows * dim * 4);
    dp_download(off, dout_off, (size_t)rows * dim * 4);
    const bool same = dp_cmp_f32(on.data(), off.data(), (size_t)rows * dim, tag);
    if (same) printf("  [%s] OK  hc_collapse_norm == hc_collapse + rmsnorm (bs=%d dim=%d)\n", tag, rows, dim);

    dx.free(); dpre.free(); dw.free(); dout_on.free(); dout_off.free(); dcol.free();
    return same ? 0 : 1;
}

// =====================================================================
// Arm P3A a2 — the two `hc_post` destination swaps (DSV41_P3A_HCPOST_SWAP)
//
// This is NOT a kernel fold: the two `hc_post`s write into each other's buffer
// instead of staging in `h_out` + `memcpy_d2d(h <- h_out)`. The claim is over
// the BUFFER STATE: after both sub-blocks, `h` holds the same bits.
//
// OFF: attn: hc_post(o, h, post, comb -> h_out); memcpy(h <- h_out)
//      ffn : hc_post(moe, h, post, comb -> h_out); memcpy(h <- h_out)
// ON : attn: hc_post(o, h, post, comb -> h_out)
//      ffn : hc_post(moe, h_out, post, comb -> h)     (no copy)
// =====================================================================
int dp_arm_p3a_a2() {
    const char* tag = "P3A-a2";
    const int rows = kBm, hc = kHc, dim = kDim;
    const size_t rsz = (size_t)rows * hc * dim * 4;
    std::vector<float> vx((size_t)rows * hc * dim), vo((size_t)rows * hc * dim),
        vmoe((size_t)rows * hc * dim), vpost((size_t)rows * hc), vcomb((size_t)rows * hc * hc);
    dp_fill(vx, -1.0f, 1.0f);
    dp_fill(vo, -1.0f, 1.0f);
    dp_fill(vmoe, -1.0f, 1.0f);
    dp_fill(vpost, 0.2f, 1.2f);
    dp_fill(vcomb, -0.3f, 0.3f);

    Buf d_o, d_moe, d_post, d_comb;
    Buf h_off, hout_off, h_on, hout_on;
    bool ok = dp_upload(d_o, vo.data(), vo.size() * 4) &&
              dp_upload(d_moe, vmoe.data(), vmoe.size() * 4) &&
              dp_upload(d_post, vpost.data(), vpost.size() * 4) &&
              dp_upload(d_comb, vcomb.data(), vcomb.size() * 4) &&
              dp_upload(h_off, vx.data(), rsz) && dp_upload(h_on, vx.data(), rsz) &&
              hout_off.alloc(rsz) && hout_on.alloc(rsz);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;

    // ---- OFF ----
    cudaError_t e = ferrite_hc_post((const float*)d_o.p, (const float*)h_off.p,
                                    (const float*)d_post.p, (const float*)d_comb.p,
                                    (float*)hout_off.p, rows, hc, dim, nullptr);
    DP_CHECK(e == cudaSuccess, "[%s] ferrite_hc_post(attn,off): %s", tag, cudaGetErrorString(e));
    cudaMemcpyAsync(h_off.p, hout_off.p, rsz, cudaMemcpyDeviceToDevice, nullptr);
    e = ferrite_hc_post((const float*)d_moe.p, (const float*)h_off.p, (const float*)d_post.p,
                        (const float*)d_comb.p, (float*)hout_off.p, rows, hc, dim, nullptr);
    DP_CHECK(e == cudaSuccess, "[%s] ferrite_hc_post(ffn,off): %s", tag, cudaGetErrorString(e));
    cudaMemcpyAsync(h_off.p, hout_off.p, rsz, cudaMemcpyDeviceToDevice, nullptr);

    // ---- ON (the a2 pointer swap) ----
    e = ferrite_hc_post((const float*)d_o.p, (const float*)h_on.p, (const float*)d_post.p,
                        (const float*)d_comb.p, (float*)hout_on.p, rows, hc, dim, nullptr);
    DP_CHECK(e == cudaSuccess, "[%s] ferrite_hc_post(attn,on): %s", tag, cudaGetErrorString(e));
    e = ferrite_hc_post((const float*)d_moe.p, (const float*)hout_on.p, (const float*)d_post.p,
                        (const float*)d_comb.p, (float*)h_on.p, rows, hc, dim, nullptr);
    DP_CHECK(e == cudaSuccess, "[%s] ferrite_hc_post(ffn,on): %s", tag, cudaGetErrorString(e));

    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> a, b;
    dp_download(a, h_off, rsz);
    dp_download(b, h_on, rsz);
    const bool same = dp_cmp_f32(a.data(), b.data(), a.size(), tag);
    if (same)
        printf("  [%s] OK  residual `h` after both sub-blocks identical (bs=%d hc=%d dim=%d)\n", tag,
               rows, hc, dim);

    d_o.free(); d_moe.free(); d_post.free(); d_comb.free();
    h_off.free(); hout_off.free(); h_on.free(); hout_on.free();
    return same ? 0 : 1;
}

// =====================================================================
// Arm P3A a3 — the premix ping-pong (DSV41_P3A_PREMIX_PP)
//
// Also a pointer fold, not a kernel: the next block's incoming premix is the
// previous block's own `pre_ffn` slot instead of a `memcpy_d2d(pre_in <- pre_ffn)`.
// The claim is that the premix a `hc_collapse` consumes is the same bits either
// way.
//
// OFF: hc_mixes -> pre_ffn ; memcpy(pre_in <- pre_ffn) ; hc_collapse(h, pre_in)
// ON : hc_mixes -> pre_ffn ;                          hc_collapse(h, pre_ffn)
// =====================================================================
int dp_arm_p3a_a3() {
    const char* tag = "P3A-a3";
    const int rows = kBm, hc = kHc, dim = kDim;
    const int hc_dim = hc * dim;               // 20480 (the whole-row reduction)
    const int mix = hc * (2 + hc);             // 24
    std::vector<float> vx((size_t)rows * hc * dim), vfn((size_t)mix * hc_dim), vsc(mix), vbase(mix);
    std::vector<float> hh((size_t)rows * hc * dim);   // the residual `h` the collapse reads
    dp_fill(vx, -1.0f, 1.0f);
    dp_fill(vfn, -0.02f, 0.02f);
    dp_fill(vsc, 0.5f, 1.5f);
    dp_fill(vbase, -0.5f, 0.5f);
    dp_fill(hh, -1.0f, 1.0f);

    Buf dx, dfn, dsc, dbase, dh, dpre_ffn, dpre_in, dpost, dcomb, dcol_on, dcol_off;
    bool ok = dp_upload(dx, vx.data(), vx.size() * 4) && dp_upload(dfn, vfn.data(), vfn.size() * 4) &&
              dp_upload(dsc, vsc.data(), vsc.size() * 4) &&
              dp_upload(dbase, vbase.data(), vbase.size() * 4) &&
              dp_upload(dh, hh.data(), hh.size() * 4) &&
              dpre_ffn.alloc((size_t)rows * hc * 4) && dpre_in.alloc((size_t)rows * hc * 4) &&
              dpost.alloc((size_t)rows * hc * 4) && dcomb.alloc((size_t)rows * hc * hc * 4) &&
              dcol_on.alloc((size_t)rows * dim * 4) && dcol_off.alloc((size_t)rows * dim * 4);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;

    const int rc_m = dsv41_hc_mixes((const float*)dx.p, (const float*)dfn.p, (const float*)dsc.p,
                                    (const float*)dbase.p, (float*)dpre_ffn.p, (float*)dpost.p,
                                    (float*)dcomb.p, rows, hc_dim, hc, /*sinkhorn=*/3, 1e-6f,
                                    nullptr);
    DP_CHECK(rc_m == 0, "[%s] hc_mixes rc=%d", tag, rc_m);

    // OFF: copy the premix into `pre_in`, then collapse with `pre_in`.
    cudaMemcpyAsync(dpre_in.p, dpre_ffn.p, (size_t)rows * hc * 4, cudaMemcpyDeviceToDevice, nullptr);
    const int rc_co = dsv41_hc_collapse((const float*)dh.p, (const float*)dpre_in.p,
                                        (float*)dcol_off.p, rows, hc, dim, nullptr);
    DP_CHECK(rc_co == 0, "[%s] hc_collapse(off) rc=%d", tag, rc_co);
    // ON: collapse straight off `pre_ffn`.
    const int rc_cn = dsv41_hc_collapse((const float*)dh.p, (const float*)dpre_ffn.p,
                                        (float*)dcol_on.p, rows, hc, dim, nullptr);
    DP_CHECK(rc_cn == 0, "[%s] hc_collapse(on) rc=%d", tag, rc_cn);

    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    // The premix itself (the copied slot == the slot) and the collapse it feeds.
    std::vector<float> pf, pi, co_on, co_off;
    dp_download(pf, dpre_ffn, (size_t)rows * hc * 4);
    dp_download(pi, dpre_in, (size_t)rows * hc * 4);
    dp_download(co_on, dcol_on, (size_t)rows * dim * 4);
    dp_download(co_off, dcol_off, (size_t)rows * dim * 4);
    const bool same = dp_cmp_f32(pf.data(), pi.data(), pf.size(), "P3A-a3/premix") &
                      dp_cmp_f32(co_on.data(), co_off.data(), co_on.size(), "P3A-a3/collapse");
    if (same)
        printf("  [%s] OK  premix ping-pong == copy+read (bs=%d hc=%d dim=%d)\n", tag, rows, hc, dim);

    dx.free(); dfn.free(); dsc.free(); dbase.free(); dh.free();
    dpre_ffn.free(); dpre_in.free(); dpost.free(); dcomb.free(); dcol_on.free(); dcol_off.free();
    return same ? 0 : 1;
}

// =====================================================================
// Arm P3A a4 — `dsv41_apply_rope_mrows`  vs  `bs` per-row `apply_rope`
//
// ON : apply_rope_mrows(x, ..., m=bs, rows=nh, row_stride=rstride, t=pos_rows[r])
// OFF: for r: apply_rope(x + r*rstride, ..., rows=nh, base, mul=1, off=P0+r, step=0)
// Both rotate the trailing `rd` columns of every head row; the position is the
// same integer (pos_rows[r] = P0 + r). Forward and inverse arms.
// =====================================================================
int dp_arm_p3a_a4() {
    const char* tag = "P3A-a4";
    const int m = kBm, rows = kNh, rstride = kNh * kHd, row_len = kHd;
    const size_t xn = (size_t)m * rstride;
    std::vector<float> cos, sin;
    dp_rope_tables(cos, sin);
    std::vector<float> vx(xn);
    dp_fill(vx, -2.0f, 2.0f);
    std::vector<int> vpos(m);
    for (int r = 0; r < m; ++r) vpos[r] = kP0 + r;

    Buf dx_on, dx_off, dcos, dsin, dpos, dbase;
    std::vector<float> hbase(1, (float)kP0);
    bool ok = dp_upload(dx_on, vx.data(), xn * 4) && dp_upload(dx_off, vx.data(), xn * 4) &&
              dp_upload(dcos, cos.data(), cos.size() * 4) &&
              dp_upload(dsin, sin.data(), sin.size() * 4) && dp_upload(dpos, vpos.data(), m * 4) &&
              dp_upload(dbase, hbase.data(), 4);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;

    int fails = 0;
    for (int inv = 0; inv <= 1; ++inv) {
        // restore both inputs (the arms rope in place)
        cudaMemcpy(dx_on.p, vx.data(), xn * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dx_off.p, vx.data(), xn * 4, cudaMemcpyHostToDevice);

        const int rc = dsv41_apply_rope_mrows((float*)dx_on.p, (const float*)dcos.p,
                                              (const float*)dsin.p, m, rows, rstride, row_len, kRd,
                                              kHalf, (const int*)dpos.p, inv, nullptr);
        DP_CHECK(rc == 0, "[%s/inv=%d] apply_rope_mrows rc=%d", tag, inv, rc);
        for (int r = 0; r < m; ++r) {
            const int rc1 = dsv41_apply_rope((float*)dx_off.p + (size_t)r * rstride,
                                             (const float*)dcos.p, (const float*)dsin.p, rows,
                                             row_len, kRd, kHalf, (const int*)dbase.p, 1, kP0 + r, 0,
                                             inv, nullptr);
            DP_CHECK(rc1 == 0, "[%s/inv=%d] apply_rope(r=%d) rc=%d", tag, inv, r, rc1);
        }
        cudaError_t se = cudaDeviceSynchronize();
        DP_CHECK(se == cudaSuccess, "[%s/inv=%d] sync: %s", tag, inv, cudaGetErrorString(se));

        std::vector<float> a, b;
        dp_download(a, dx_on, xn * 4);
        dp_download(b, dx_off, xn * 4);
        char t[96];
        std::snprintf(t, sizeof t, "%s/inv=%d", tag, inv);
        if (!dp_cmp_f32(a.data(), b.data(), xn, t)) ++fails;
        else
            printf("  [%s/inv=%d] OK  apply_rope_mrows == %d x apply_rope (m=%d rows=%d rd=%d)\n", tag,
                   inv, m, m, rows, kRd);
    }
    dx_on.free(); dx_off.free(); dcos.free(); dsin.free(); dpos.free(); dbase.free();
    return fails;
}

// =====================================================================
// Arm P3LITE l1 (seed) / l2 (kv) — `dsv41_rmsnorm_rope` vs `rmsnorm` + `apply_rope`
//
// ON : dsv41_rmsnorm_rope(x, w, out, cos, sin, n, dim, rope_len, half, base, 1,
//                         off, step, inverse, eps)
// OFF: ferrite_rmsnorm(x, w, out, n, dim, eps)
//      dsv41_apply_rope(out, cos, sin, n, dim, rd, half, base, 1, off, step, inv)
//
// l1 is the seed (n = 1, step = 1, off = pos - pos_dev), l2 the kv block
// (n = bs, step = 1, off = kv_pos - pos_dev, base = pos_dev). Both are IN PLACE
// in production (out == x), so two identical copies are roped and compared.
// =====================================================================
int dp_arm_p3lite_norm_rope(const char* tag, int n, int off, int step) {
    const int dim = kHd;
    std::vector<float> cos, sin;
    dp_rope_tables(cos, sin);
    std::vector<float> vx((size_t)n * dim), vw(dim);
    dp_fill(vx, -2.0f, 2.0f);
    dp_fill(vw, 0.4f, 1.6f);

    Buf dx_on, dx_off, dw, dcos, dsin, dbase;
    std::vector<float> hbase(1, (float)kPos);   // base = pos_dev
    bool ok = dp_upload(dx_on, vx.data(), vx.size() * 4) && dp_upload(dx_off, vx.data(), vx.size() * 4) &&
              dp_upload(dw, vw.data(), vw.size() * 4) && dp_upload(dcos, cos.data(), cos.size() * 4) &&
              dp_upload(dsin, sin.data(), sin.size() * 4) && dp_upload(dbase, hbase.data(), 4);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;

    // ON (in place, as the draft calls it)
    const int rc = dsv41_rmsnorm_rope((const float*)dx_on.p, (const float*)dw.p, (float*)dx_on.p,
                                      (const float*)dcos.p, (const float*)dsin.p, n, dim, kRd, kHalf,
                                      (const int*)dbase.p, 1, off, step, 0, kEps, nullptr);
    DP_CHECK(rc == 0, "[%s] rmsnorm_rope rc=%d", tag, rc);
    // OFF
    const cudaError_t rn = ferrite_rmsnorm((const float*)dx_off.p, (const float*)dw.p,
                                           (float*)dx_off.p, n, dim, kEps, nullptr);
    DP_CHECK(rn == cudaSuccess, "[%s] ferrite_rmsnorm: %s", tag, cudaGetErrorString(rn));
    const int rr = dsv41_apply_rope((float*)dx_off.p, (const float*)dcos.p, (const float*)dsin.p, n,
                                    dim, kRd, kHalf, (const int*)dbase.p, 1, off, step, 0, nullptr);
    DP_CHECK(rr == 0, "[%s] apply_rope rc=%d", tag, rr);

    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> a, b;
    dp_download(a, dx_on, vx.size() * 4);
    dp_download(b, dx_off, vx.size() * 4);
    const bool same = dp_cmp_f32(a.data(), b.data(), a.size(), tag);
    if (same)
        printf("  [%s] OK  rmsnorm_rope == rmsnorm + apply_rope (n=%d dim=%d rd=%d off=%d step=%d)\n",
               tag, n, dim, kRd, off, step);

    dx_on.free(); dx_off.free(); dw.free(); dcos.free(); dsin.free(); dbase.free();
    return same ? 0 : 1;
}

// =====================================================================
// Arm P3LITE l4 (segment B / K2) — `dsv41_gemm_fp8_mrows_rope_norm`
//
// ON : dsv41_gemm_fp8_mrows_rope_norm(qr_raw, qr_w, eps, qr_norm_out == qr_raw,
//          wq_b, wsq_b, null, q, m=bs, n=nh*hd, k=ql, out_stride=nh*hd,
//          cos, sin, pos_rows, rope_rd=64, rope_hd=hd, forward=0)
// OFF: rmsnorm_rows(qr) ; quant_fp8(qr) ; gemm_fp8_mrows(wq_b) ; per-row rope
//
// R1: the reference uses `gemm_fp8_mrows`, which is the program the draft runs
// with DSV41_ATTN_PROJ_ALIGN=1 (the unaligned draft's `gemm_fp8_mx@m=bs` is the
// 16-row TILE MMA — a DIFFERENT program, and folding across it is exactly the
// P3B-class claim rule R1 forbids). The arm therefore measures the SEGMENT
// FUSION, not a program swap; run it with ATTN_PROJ_ALIGN on.
// =====================================================================
int dp_arm_p3lite_l4() {
    const char* tag = "P3LITE-l4/K2";
    const int m = kBm, n = kNh * kHd, k = kQl, out_stride = kNh * kHd, nb_k = k / 32;
    if (!dp_have_free((size_t)n * k + 8u * 1024u * 1024u)) return 0;
    if (g_gemv_fp8_mode < 3 || getenv("DSV41_NO_GEMV_FP8") != nullptr) {
        printf("  [%s] SKIP: the m=1 GEMV program is off (mode=%d, no_gemv=%d)\n", tag,
               g_gemv_fp8_mode, (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr));
        ++g_skips;
        return 0;
    }

    std::vector<float> cos, sin;
    dp_rope_tables(cos, sin);
    std::vector<float> vqr((size_t)m * k), vqrw(k), vout((size_t)m * n);
    std::vector<uint8_t> vw((size_t)n * k), vws((size_t)(n / 32) * nb_k);
    dp_fill(vqr, -1.5f, 1.5f);
    dp_fill(vqrw, 0.4f, 1.6f);
    dp_fill(vout, -1.0f, 1.0f);
    for (auto& b : vw) b = (uint8_t)(dp_xr() & 0xFFu);
    for (auto& b : vws) b = (uint8_t)(124 + dp_irand(9));
    std::vector<int> vpos(m);
    for (int r = 0; r < m; ++r) vpos[r] = kP0 + r;

    Buf dqr_on, dqr_off, dqrw, dw, dws, dq_on, dq_off, dxq, dxsc, dcos, dsin, dpos, dbase;
    std::vector<float> hbase(1, (float)kP0);
    const size_t xq_bytes = (size_t)m * k;                 // fp8, one byte per value
    const size_t xsc_bytes = (size_t)m * nb_k * 4;
    bool ok = dp_upload(dqr_on, vqr.data(), vqr.size() * 4) &&
              dp_upload(dqr_off, vqr.data(), vqr.size() * 4) &&
              dp_upload(dqrw, vqrw.data(), vqrw.size() * 4) && dp_upload(dw, vw.data(), vw.size()) &&
              dp_upload(dws, vws.data(), vws.size()) && dp_upload(dq_on, vout.data(), vout.size() * 4) &&
              dp_upload(dq_off, vout.data(), vout.size() * 4) &&
              dxq.alloc(xq_bytes) && dxsc.alloc(xsc_bytes) &&
              dp_upload(dcos, cos.data(), cos.size() * 4) &&
              dp_upload(dsin, sin.data(), sin.size() * 4) && dp_upload(dpos, vpos.data(), m * 4) &&
              dp_upload(dbase, hbase.data(), 4);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;

    // ON
    const int rc = dsv41_gemm_fp8_mrows_rope_norm(
        (const float*)dqr_on.p, (const float*)dqrw.p, kEps, (float*)dqr_on.p, (const uint8_t*)dw.p,
        (const uint8_t*)dws.p, nullptr, (float*)dq_on.p, m, n, k, out_stride, (const float*)dcos.p,
        (const float*)dsin.p, (const int*)dpos.p, kRd, kHd, /*forward=*/0, nullptr);
    if (rc == 2) {
        printf("  [%s] SKIP: gemm_fp8_mrows_rope_norm declined (mode=%d)\n", tag, g_gemv_fp8_mode);
        ++g_skips;
        dqr_on.free(); dqr_off.free(); dqrw.free(); dw.free(); dws.free(); dq_on.free();
        dq_off.free(); dxq.free(); dxsc.free(); dcos.free(); dsin.free(); dpos.free(); dbase.free();
        return 0;
    }
    DP_CHECK(rc == 0, "[%s] mrows_rope_norm rc=%d (%s)", tag, rc, cudaGetErrorString((cudaError_t)rc));

    // OFF: rmsnorm_rows + quant_fp8 + gemm_fp8_mrows + per-row rope.
    const int rn = dsv41_rmsnorm_rows((const float*)dqr_off.p, (const float*)dqrw.p,
                                      (float*)dqr_off.p, m, k, kEps, nullptr);
    DP_CHECK(rn == 0, "[%s] rmsnorm_rows rc=%d", tag, rn);
    const int qn = dsv41_quant_fp8((const float*)dqr_off.p, (uint8_t*)dxq.p, (float*)dxsc.p, m, k,
                                   32, 1, nullptr);
    DP_CHECK(qn == 0, "[%s] quant_fp8 rc=%d", tag, qn);
    const int gm = dsv41_gemm_fp8_mrows((const uint8_t*)dxq.p, (const float*)dxsc.p,
                                        (const uint8_t*)dw.p, (const uint8_t*)dws.p, nullptr,
                                        (float*)dq_off.p, m, n, k, out_stride, nullptr);
    DP_CHECK(gm == 0, "[%s] gemm_fp8_mrows rc=%d", tag, gm);
    for (int r = 0; r < m; ++r) {
        const int rr = dsv41_apply_rope((float*)dq_off.p + (size_t)r * out_stride,
                                        (const float*)dcos.p, (const float*)dsin.p, kNh, kHd, kRd,
                                        kHalf, (const int*)dbase.p, 1, kP0 + r, 0, 0, nullptr);
        DP_CHECK(rr == 0, "[%s] apply_rope(r=%d) rc=%d", tag, r, rr);
    }

    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> qn_on, qn_off, q_on, q_off;
    dp_download(qn_on, dqr_on, (size_t)m * k * 4);
    dp_download(qn_off, dqr_off, (size_t)m * k * 4);
    dp_download(q_on, dq_on, (size_t)m * n * 4);
    dp_download(q_off, dq_off, (size_t)m * n * 4);
    bool same = dp_cmp_f32(qn_on.data(), qn_off.data(), qn_on.size(), "P3LITE-l4/qr_norm");
    same &= dp_cmp_f32(q_on.data(), q_off.data(), q_on.size(), "P3LITE-l4/q");
    if (same)
        printf("  [%s] OK  K2 == rmsnorm + quant + gemm(wq_b) + rope (m=%d n=%d k=%d)\n", tag, m, n, k);

    dqr_on.free(); dqr_off.free(); dqrw.free(); dw.free(); dws.free(); dq_on.free();
    dq_off.free(); dxq.free(); dxsc.free(); dcos.free(); dsin.free(); dpos.free(); dbase.free();
    return same ? 0 : 1;
}

// =====================================================================
// Arm P3LITE l3 — `dsv41_sparse_attn_orope` vs `sparse_attn` + per-row inv rope
// + `quant_fp8`
//
// ON : sparse_attn_orope(..., row_step=1, inverse=1, xq, xsc)
// OFF: sparse_attn(...) ; per-row apply_rope(inverse=1) ; quant_fp8(o)
// The draft's `b*m = 5 <= kAttnMaxBM = 8` selects the SPLIT arm, so this also
// pins the `row_step` term the merge kernel gained (its absence would rope rows
// 1..4 at row 0's position, silently).
// =====================================================================
int dp_arm_p3lite_l3() {
    const char* tag = "P3LITE-l3/orope";
    const int b = 1, m = kBm, h = kNh, d = kHd;
    const int window = kWin, cl = m, index_topk = m;
    const int n = window + cl;                    // kv rows
    const int topk = window + ((cl < index_topk) ? cl : index_topk);
    // The merge's `tt` = base*1 + off + hh*0 + mm*row_step; base holds pos_dev,
    // off = rope_pos - pos_dev with rope_pos == pos_dev == kPos, so tt = kPos + mm.
    const int off = 0;

    std::vector<float> cos, sin;
    dp_rope_tables(cos, sin);
    std::vector<float> q((size_t)b * m * h * d), kv((size_t)b * n * d), sink(h),
        out_on((size_t)b * m * h * d), out_off((size_t)b * m * h * d);
    dp_fill(q, -1.5f, 1.5f);
    dp_fill(kv, -1.5f, 1.5f);
    dp_fill(sink, -1.0f, 1.0f);
    std::vector<int32_t> idxs((size_t)b * m * topk);
    for (int r = 0; r < b * m; ++r) {
        for (int t = 0; t < topk; ++t) idxs[(size_t)r * topk + t] = dp_irand(n);
        idxs[(size_t)r * topk + topk - 1] = -1;   // exercise the skip path (row spares a slot)
    }
    const size_t xq_bytes = (size_t)b * m * h * d;
    const size_t xsc_bytes = ((size_t)b * m * h * d) / 32;

    Buf dq, dkv, dsink, didxs, dou_on, dou_off, dclen, dcos, dsin, dbase, dxq, dxsc;
    std::vector<float> hbase(1, (float)kPos);
    std::vector<int> hclen(1, cl);
    bool ok = dp_upload(dq, q.data(), q.size() * 4) && dp_upload(dkv, kv.data(), kv.size() * 4) &&
              dp_upload(dsink, sink.data(), sink.size() * 4) &&
              dp_upload(didxs, idxs.data(), idxs.size() * 4) &&
              dp_upload(dou_on, out_on.data(), out_on.size() * 4) &&
              dp_upload(dou_off, out_off.data(), out_off.size() * 4) &&
              dp_upload(dclen, hclen.data(), 4) && dp_upload(dcos, cos.data(), cos.size() * 4) &&
              dp_upload(dsin, sin.data(), sin.size() * 4) && dp_upload(dbase, hbase.data(), 4) &&
              dxq.alloc(xq_bytes) && dxsc.alloc(xsc_bytes);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;
    Buf dxq_off, dxsc_off;
    ok = dxq_off.alloc(xq_bytes) && dxsc_off.alloc(xsc_bytes);
    DP_CHECK(ok, "[%s] alloc(ref)", tag);
    if (!ok) return 1;

    const float scale = (float)std::pow((double)d, -0.5);

    // ON
    const int rc = dsv41_sparse_attn_orope(
        (const float*)dq.p, (const float*)dkv.p, (const float*)dsink.p, (const int32_t*)didxs.p,
        (float*)dou_on.p, b, m, h, d, (const int*)dclen.p, window, index_topk, scale,
        (const float*)dcos.p, (const float*)dsin.p, (const int*)dbase.p, kRd, kHalf, 1, off, 0, 1,
        (uint8_t*)dxq.p, (float*)dxsc.p, nullptr, 0, 1, nullptr);
    if (rc != 0) {
        printf("  [%s] SKIP: sparse_attn_orope declined rc=%d (ATTN_SEQ / ATTN_PF / split config)\n",
               tag, rc);
        ++g_skips;
        dq.free(); dkv.free(); dsink.free(); didxs.free(); dou_on.free(); dou_off.free();
        dclen.free(); dcos.free(); dsin.free(); dbase.free(); dxq.free(); dxsc.free();
        dxq_off.free(); dxsc_off.free();
        return 0;
    }
    // OFF: plain sparse_attn, then the per-row inverse rope and the fp8 pair.
    const int rs = dsv41_sparse_attn((const float*)dq.p, (const float*)dkv.p, (const float*)dsink.p,
                                     (const int32_t*)didxs.p, (float*)dou_off.p, b, m, h, d,
                                     (const int*)dclen.p, window, index_topk, scale, nullptr, 0,
                                     nullptr);
    DP_CHECK(rs == 0, "[%s] sparse_attn rc=%d", tag, rs);
    for (int r = 0; r < b * m; ++r) {
        const int rr = dsv41_apply_rope((float*)dou_off.p + (size_t)r * h * d, (const float*)dcos.p,
                                        (const float*)dsin.p, h, d, kRd, kHalf, (const int*)dbase.p, 1,
                                        kPos + r, 0, 1, nullptr);
        DP_CHECK(rr == 0, "[%s] apply_rope(r=%d) rc=%d", tag, r, rr);
    }
    const int qn = dsv41_quant_fp8((const float*)dou_off.p, (uint8_t*)dxq_off.p,
                                   (float*)dxsc_off.p, 1, (int)((size_t)b * m * h * d), 32, 1,
                                   nullptr);
    DP_CHECK(qn == 0, "[%s] quant_fp8 rc=%d", tag, qn);

    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> o_on, o_off, sc_on, sc_off;
    std::vector<uint8_t> xq_on, xq_off;
    dp_download(o_on, dou_on, out_on.size() * 4);
    dp_download(o_off, dou_off, out_off.size() * 4);
    dp_download(xq_on, dxq, xq_bytes);
    dp_download(xq_off, dxq_off, xq_bytes);
    dp_download(sc_on, dxsc, xsc_bytes);
    dp_download(sc_off, dxsc_off, xsc_bytes);
    bool same = dp_cmp_f32(o_on.data(), o_off.data(), o_on.size(), "P3LITE-l3/o");
    same &= dp_cmp_u8(xq_on.data(), xq_off.data(), xq_bytes, "P3LITE-l3/xq");
    same &= dp_cmp_f32(sc_on.data(), sc_off.data(), sc_on.size(), "P3LITE-l3/xsc");
    if (same)
        printf("  [%s] OK  orope == sparse_attn + inv rope x %d + quant (b=%d m=%d h=%d d=%d)\n", tag,
               b * m, b, m, h, d);

    dq.free(); dkv.free(); dsink.free(); didxs.free(); dou_on.free(); dou_off.free();
    dclen.free(); dcos.free(); dsin.free(); dbase.free(); dxq.free(); dxsc.free();
    dxq_off.free(); dxsc_off.free();
    return same ? 0 : 1;
}

// =====================================================================
// Arm DRAFT_MOE_MROWS — routed experts rows = bs  vs  bs launches of rows = 1
//
// The whole chain gate/up -> swiglu -> down+reduce, once at rows = bs and once
// per row at rows = 1. Row r of the batched launch must be BIT-IDENTICAL to the
// rows = 1 call on row r (only base pointers move between rows — see the
// launchers' ROW INDEPENDENCE block). The RAW gate|up pair layout is used
// (act_slot = 2*inter) so the comparison does not depend on the gate/up fusion
// env; the swiglu pass is run explicitly on both sides.
// =====================================================================
int dp_arm_moe_mrows() {
    const char* tag = "MOE-MROWS";
    const int rows = kBm, dim = kDim, inter = kInter, slots = kSlots, ne = 8;
    const float limit = 3.0f;
    const long act_slot = (long)2 * inter;               // raw pair (unfused)
    const int gup_bytes = inter * (dim / 2);
    const int gsc_bytes = inter * (dim / 32);
    const int w2_bytes = dim * (inter / 2);
    const int w2s_bytes = dim * (inter / 32);
    const size_t gu_n = (size_t)rows * slots * act_slot;
    const size_t dn_n = (size_t)rows * dim;
    const size_t need = (size_t)ne * (2 * gup_bytes + 2 * gsc_bytes + w2_bytes + w2s_bytes) +
                        4 * gu_n * 4 + 4 * dn_n * 4 + (64u << 20);
    if (!dp_have_free(need)) return 0;

    std::vector<uint8_t> w1p((size_t)ne * gup_bytes), w1s((size_t)ne * gsc_bytes);
    std::vector<uint8_t> w3p((size_t)ne * gup_bytes), w3s((size_t)ne * gsc_bytes);
    std::vector<uint8_t> w2p((size_t)ne * w2_bytes), w2s((size_t)ne * w2s_bytes);
    for (auto& v : {&w1p, &w3p, &w2p})
        for (auto& x : *v) x = (uint8_t)(dp_xr() & 0xFFu);
    for (auto* v : {&w1s, &w3s, &w2s})
        for (auto& x : *v) x = (uint8_t)(124 + dp_irand(9));
    std::vector<uint8_t> ap((size_t)rows * (dim / 2));
    std::vector<float> asc((size_t)rows * (dim / 32));
    for (auto& x : ap) x = (uint8_t)(dp_xr() & 0xFFu);
    for (auto& x : asc) x = std::ldexp(1.0f, -2 + dp_irand(4));
    std::vector<int> ids((size_t)rows * slots);
    for (auto& x : ids) x = dp_irand(ne);
    std::vector<float> rw((size_t)rows * slots);
    for (auto& x : rw) x = (float)(1 + dp_irand(3)) * 0.25f;

    Buf d_w1p, d_w1s, d_w3p, d_w3s, d_w2p, d_w2s, d_ap, d_asc, d_ids, d_rw;
    Buf d_gu_mr, d_gu_1r, d_dn_mr, d_dn_1r;
    bool ok = dp_upload(d_w1p, w1p.data(), w1p.size()) && dp_upload(d_w1s, w1s.data(), w1s.size()) &&
              dp_upload(d_w3p, w3p.data(), w3p.size()) && dp_upload(d_w3s, w3s.data(), w3s.size()) &&
              dp_upload(d_w2p, w2p.data(), w2p.size()) && dp_upload(d_w2s, w2s.data(), w2s.size()) &&
              dp_upload(d_ap, ap.data(), ap.size()) && dp_upload(d_asc, asc.data(), asc.size() * 4) &&
              dp_upload(d_ids, ids.data(), ids.size() * 4) && dp_upload(d_rw, rw.data(), rw.size() * 4) &&
              d_gu_mr.alloc(gu_n * 4) && d_gu_1r.alloc(gu_n * 4) && d_dn_mr.alloc(dn_n * 4) &&
              d_dn_1r.alloc(dn_n * 4);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;
    cudaMemset(d_gu_mr.p, 0xFF, gu_n * 4);
    cudaMemset(d_dn_mr.p, 0xFF, dn_n * 4);

    // ---- rows = bs (the fold) ----
    int rc = dsv41_expert_gate_up_fp4_batched(
        (const uint8_t*)d_ap.p, (const float*)d_asc.p, (float*)d_gu_mr.p, act_slot, rows, dim, inter,
        limit, slots, (const uint8_t*)d_w1p.p, gup_bytes, (const uint8_t*)d_w1s.p, gsc_bytes,
        (const uint8_t*)d_w3p.p, gup_bytes, (const uint8_t*)d_w3s.p, gsc_bytes, (const int*)d_ids.p,
        0, 0, nullptr);
    DP_CHECK(rc == 0, "[%s] gate_up(rows=%d) rc=%d", tag, rows, rc);
    rc = dsv41_swiglu_limit_batched((float*)d_gu_mr.p, rows, inter, limit, act_slot, slots, nullptr);
    DP_CHECK(rc == 0, "[%s] swiglu(rows=%d) rc=%d", tag, rows, rc);
    rc = dsv41_expert_down_reduce_fp4_batched(
        (const float*)d_gu_mr.p, act_slot, (float*)d_dn_mr.p, rows, dim, inter, (const float*)d_rw.p,
        1, slots, (const uint8_t*)d_w2p.p, w2_bytes, (const uint8_t*)d_w2s.p, w2s_bytes,
        (const int*)d_ids.p, nullptr);
    DP_CHECK(rc == 0, "[%s] down_reduce(rows=%d) rc=%d", tag, rows, rc);

    // ---- rows = 1, one launch per row ----
    for (int r = 0; r < rows; ++r) {
        const uint8_t* a_r = (const uint8_t*)d_ap.p + (size_t)r * (dim / 2);
        const float* asc_r = (const float*)d_asc.p + (size_t)r * (dim / 32);
        float* gu_r = (float*)d_gu_1r.p + (size_t)r * slots * act_slot;
        const int* ids_r = (const int*)d_ids.p + (size_t)r * slots;
        const float* rw_r = (const float*)d_rw.p + (size_t)r * slots;
        float* dn_r = (float*)d_dn_1r.p + (size_t)r * dim;
        int rc1 = dsv41_expert_gate_up_fp4_batched(
            a_r, asc_r, gu_r, act_slot, 1, dim, inter, limit, slots, (const uint8_t*)d_w1p.p,
            gup_bytes, (const uint8_t*)d_w1s.p, gsc_bytes, (const uint8_t*)d_w3p.p, gup_bytes,
            (const uint8_t*)d_w3s.p, gsc_bytes, ids_r, 0, 0, nullptr);
        DP_CHECK(rc1 == 0, "[%s] gate_up(rows=1,r=%d) rc=%d", tag, r, rc1);
        rc1 = dsv41_swiglu_limit_batched(gu_r, 1, inter, limit, act_slot, slots, nullptr);
        DP_CHECK(rc1 == 0, "[%s] swiglu(rows=1,r=%d) rc=%d", tag, r, rc1);
        rc1 = dsv41_expert_down_reduce_fp4_batched(gu_r, act_slot, dn_r, 1, dim, inter, rw_r, 1,
                                                   slots, (const uint8_t*)d_w2p.p, w2_bytes,
                                                   (const uint8_t*)d_w2s.p, w2s_bytes, ids_r, nullptr);
        DP_CHECK(rc1 == 0, "[%s] down_reduce(rows=1,r=%d) rc=%d", tag, r, rc1);
    }

    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> gu_mr, gu_1r, dn_mr, dn_1r;
    dp_download(gu_mr, d_gu_mr, gu_n * 4);
    dp_download(gu_1r, d_gu_1r, gu_n * 4);
    dp_download(dn_mr, d_dn_mr, dn_n * 4);
    dp_download(dn_1r, d_dn_1r, dn_n * 4);
    bool same = dp_cmp_f32(gu_mr.data(), gu_1r.data(), gu_n, "MOE-MROWS/gate_up");
    same &= dp_cmp_f32(dn_mr.data(), dn_1r.data(), dn_n, "MOE-MROWS/down");
    if (same)
        printf("  [%s] OK  rows=%d == %d x rows=1 (dim=%d inter=%d slots=%d ne=%d)\n", tag, rows,
               rows, dim, inter, slots, ne);

    d_w1p.free(); d_w1s.free(); d_w3p.free(); d_w3s.free(); d_w2p.free(); d_w2s.free();
    d_ap.free(); d_asc.free(); d_ids.free(); d_rw.free();
    d_gu_mr.free(); d_gu_1r.free(); d_dn_mr.free(); d_dn_1r.free();
    return same ? 0 : 1;
}

// =====================================================================
// Arm DRAFT_HEAD_FOLD v1 — `dsv41_gemv_bf16_v1_mrows` vs per-row `dsv41_gemv_bf16`
//
// ON : gemv_bf16_v1_mrows(w, x, out, m=bs, n, k)     (one weight pass, bs rows)
// OFF: for r: gemv_bf16(w, x + r*k, out + r*n, n, k)
// The v1 fold is the one the draft actually ships (`head_gemv_bf16_mrows`'s v2
// order is a KNOWN numerical change — 33% echo on the verify head — and is NOT
// tested here on purpose; it is the v2 that flips argmaxes).
// =====================================================================
int dp_arm_head_fold() {
    const char* tag = "HEAD-FOLD-v1";
    const int m = kBm, n = 4096, k = kDim;
    if (!dp_have_free((size_t)n * k * 2 + (64u << 20))) return 0;
    std::vector<uint16_t> hw((size_t)n * k);
    for (auto& b : hw) {
        const uint16_t e = (uint16_t)(122 + (dp_xr() % 9));
        b = (uint16_t)((dp_xr() & 0x8000u) | (e << 7) | (dp_xr() & 0x7Fu));
    }
    std::vector<float> hx((size_t)m * k);
    dp_fill(hx, -1.0f, 1.0f);

    Buf dw, dx, dout_on, dout_off;
    bool ok = dp_upload(dw, hw.data(), hw.size() * 2) && dp_upload(dx, hx.data(), hx.size() * 4) &&
              dout_on.alloc((size_t)m * n * 4) && dout_off.alloc((size_t)m * n * 4);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;

    const int rc = dsv41_gemv_bf16_v1_mrows((const void*)dw.p, (const float*)dx.p, (float*)dout_on.p,
                                            m, n, k, nullptr);
    DP_CHECK(rc == 0, "[%s] v1_mrows rc=%d", tag, rc);
    for (int r = 0; r < m; ++r) {
        const int rc1 = dsv41_gemv_bf16((const void*)dw.p, (const float*)dx.p + (size_t)r * k,
                                        (float*)dout_off.p + (size_t)r * n, n, k, nullptr);
        DP_CHECK(rc1 == 0, "[%s] gemv_bf16(r=%d) rc=%d", tag, r, rc1);
    }
    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));

    std::vector<float> a, b;
    dp_download(a, dout_on, (size_t)m * n * 4);
    dp_download(b, dout_off, (size_t)m * n * 4);
    const bool same = dp_cmp_f32(a.data(), b.data(), a.size(), tag);
    if (same)
        printf("  [%s] OK  v1_mrows == %d x gemv_bf16 (m=%d n=%d k=%d)\n", tag, m, m, n, k);

    dw.free(); dx.free(); dout_on.free(); dout_off.free();
    return same ? 0 : 1;
}

// =====================================================================
// Control — the tree the norm folds rest on
//
// `dsv41_rmsnorm_rows` and `ferrite_rmsnorm` must agree bit for bit (the
// l1/l2 arms' OFF side runs `ferrite_rmsnorm` while the fused kernels'
// in-kernel tree is a transcription of it). A diff here means the environment
// or a tree changed, NOT that a fold is wrong.
// =====================================================================
int dp_ctl_norm_tree() {
    const char* tag = "CTL/norm-tree";
    const int rows = kBm, dim = kQl;
    std::vector<float> hx((size_t)rows * dim), hw(dim);
    dp_fill(hx, -1.5f, 1.5f);
    dp_fill(hw, 0.4f, 1.6f);
    Buf dx, dw, dout_a, dout_b;
    bool ok = dp_upload(dx, hx.data(), hx.size() * 4) && dp_upload(dw, hw.data(), hw.size() * 4) &&
              dout_a.alloc((size_t)rows * dim * 4) && dout_b.alloc((size_t)rows * dim * 4);
    DP_CHECK(ok, "[%s] alloc/upload", tag);
    if (!ok) return 1;
    const cudaError_t rc_a = ferrite_rmsnorm((const float*)dx.p, (const float*)dw.p,
                                             (float*)dout_a.p, rows, dim, kEps, nullptr);
    DP_CHECK(rc_a == cudaSuccess, "[%s] ferrite_rmsnorm: %s", tag, cudaGetErrorString(rc_a));
    const int rc_b = dsv41_rmsnorm_rows((const float*)dx.p, (const float*)dw.p, (float*)dout_b.p,
                                        rows, dim, kEps, nullptr);
    DP_CHECK(rc_b == 0, "[%s] rmsnorm_rows rc=%d", tag, rc_b);
    cudaError_t se = cudaDeviceSynchronize();
    DP_CHECK(se == cudaSuccess, "[%s] sync: %s", tag, cudaGetErrorString(se));
    std::vector<float> a, b;
    dp_download(a, dout_a, (size_t)rows * dim * 4);
    dp_download(b, dout_b, (size_t)rows * dim * 4);
    const bool same = dp_cmp_f32(a.data(), b.data(), a.size(), tag);
    if (same) printf("  [%s] OK  ferrite_rmsnorm == dsv41_rmsnorm_rows (rows=%d dim=%d)\n", tag, rows, dim);
    dx.free(); dw.free(); dout_a.free(); dout_b.free();
    return same ? 0 : 1;
}

}  // namespace

// ============================================================== the suite
struct Arm {
    const char* name;
    int (*fn)();
};

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;

    printf("== dsv41 DRAFT-chain fold parity (bs = %d, dim = %d) ==\n", kBm, kDim);
    printf("   DSV41_GEMV_FP8_MODE=%d  DSV41_NO_GEMV_FP8=%d  sparse_split_c=%d  ATTN_SEQ=%d  ATTN_PF=%s\n",
           g_gemv_fp8_mode, (int)(getenv("DSV41_NO_GEMV_FP8") != nullptr),
           dsv41_resolve_sparse_split_c(), (int)(getenv("DSV41_ATTN_SEQ") != nullptr),
           getenv("DSV41_ATTN_PF") == nullptr ? "<unset>" : getenv("DSV41_ATTN_PF"));
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("   device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

    // Light arms always run; the three memory-heavy ones are skipped by --quick
    // (they need ~250 MB / ~40 MB of device memory).
    g_fails += dp_ctl_norm_tree();
    g_fails += dp_arm_p3a_a1();
    g_fails += dp_arm_p3a_a2();
    g_fails += dp_arm_p3a_a3();
    g_fails += dp_arm_p3a_a4();
    g_fails += dp_arm_p3lite_norm_rope("P3LITE-l1/seed", /*n=*/1, /*off=*/kPos - kPos, /*step=*/1);
    g_fails += dp_arm_p3lite_norm_rope("P3LITE-l2/kv", /*n=*/kBm, /*off=*/kPos - kPos, /*step=*/1);
    g_fails += dp_arm_p3lite_l3();
    if (!quick) {
        g_fails += dp_arm_p3lite_l4();
        g_fails += dp_arm_moe_mrows();
        g_fails += dp_arm_head_fold();
    } else {
        printf("  (--quick: skipping P3LITE-l4/K2, MOE-MROWS, HEAD-FOLD-v1)\n");
    }

    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        printf("  sticky CUDA error after the suite: %s\n", cudaGetErrorString(ce));
        ++g_fails;
    }
    if (g_skips)
        printf("RESULT: %d check(s) FAILED, %d arm(s) SKIPPED\n", g_fails, g_skips);
    else
        printf(g_fails ? "RESULT: %d check(s) FAILED\n" : "RESULT: all draft folds bit-identical\n",
               g_fails);
    return g_fails ? 1 : 0;
}
