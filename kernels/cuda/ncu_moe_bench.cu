// ncu micro-bench: MoE fp8 kernels (act W8A8 mma + down W8A16) + hc_pre_split
// in ISOLATION with real GLM-5.3-Flash TP4 decode shapes — ncu-safe (no serve,
// no weight load; random weights; AGENTS.md: ncu on the whole serve times out,
// single-kernel repros are the supported path).
//
// Build (remote b300):  nvcc -O3 -arch=sm_103a -o /tmp/ncu_moe_bench \
//   ncu_moe_bench.cu -L. -lferrite_kernels -lcudart
// Wall-clock:           /tmp/ncu_moe_bench [iters]
// ncu (SOL + occupancy):
//   sudo /usr/local/cuda-13.2/bin/ncu --set full --launch-count 3 \
//     -k "regex:moe_fused|hc_pre" /tmp/ncu_moe_bench 3
//
// Shapes = TP4 per-rank decode (n=1): hidden 4096, inter 512 (experts),
// inter_shared 512, topk 8, e_local 288 experts (all on rank, TP slice),
// expert_start 0. hc: s=1, n=4, h=4096, mix=24, iters=4.
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

extern "C" {
cudaError_t ferrite_moe_fused_act_fp8_mma(
    const float* x, const float* ids_f,
    const void* const* gate_w8_ptrs, const void* const* gate_scale_ptrs,
    const void* const* up_w8_ptrs, const void* const* up_scale_ptrs,
    const void* shared_gate_w8, const void* shared_gate_scale,
    const void* shared_up_w8, const void* shared_up_scale,
    float* act, int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, float limit, cudaStream_t s);
cudaError_t ferrite_moe_down_mma(
    const float* ids_f, const float* probs,
    const void* const* down_w8_ptrs, const void* const* down_scale_ptrs,
    const void* shared_down_w8, const void* shared_down_scale,
    const float* act, float* out,
    int expert_start, int e_local, int hidden, int inter, int inter_shared,
    int topk, int dscols, int n, cudaStream_t s);
cudaError_t ferrite_moe_fused_down_sum_fp8(
    const float* ids_f, const float* probs,
    const void* const* down_w8_ptrs, const void* const* down_scale_ptrs,
    const void* shared_down_w8, const void* shared_down_scale,
    const float* act, float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, int dscols, cudaStream_t s);
// bf16 comparison kernels (the pre-fp8 baseline)
cudaError_t ferrite_moe_fused_act(
    const float* x, const float* ids_f,
    const void* const* gate_ptrs, const void* const* up_ptrs,
    const void* shared_gate, const void* shared_up,
    float* act, int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, float limit, cudaStream_t s);
cudaError_t ferrite_moe_fused_down_sum(
    const float* ids_f, const float* probs,
    const void* const* down_ptrs, const void* shared_down,
    const float* act, float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, cudaStream_t s);
// hc chain (mix_split phase 1 + rest phase 2 — the real decode pair)
cudaError_t ferrite_hc_pre_split(
    const float* res, const float* fw,
    const float* scale, const float* base,
    const float* nw,
    float* li, float* post, float* comb,
    float* mx_scratch,
    int s, int n, int h, int mix,
    float rms_eps, float hc_eps, int iters,
    cudaStream_t stream);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static float bench_kernel(const char* name, int iters, void (*launch)(void*), void* ctx) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch(ctx); CK(cudaDeviceSynchronize()); // warmup 1
    launch(ctx); CK(cudaDeviceSynchronize()); // warmup 2 (l2 warm)
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch(ctx);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    float us = ms * 1000.f / iters;
    printf("%-34s %8.2f us/call  (%d iters)\n", name, us, iters);
    return us;
}

// ---------------- MoE fp8 context ----------------
struct MoeCtx {
    // fp8 expert tables (device): [e] pointer arrays
    unsigned char** gate_w8; float** gate_sc;
    unsigned char** up_w8;   float** up_sc;
    unsigned char** down_w8;  float** down_sc;
    // shared expert (fp8)
    unsigned char* sg_w8; float* sg_sc;
    unsigned char* su_w8; float* su_sc;
    unsigned char* sd_w8; float* sd_sc;
    // bf16 tables (comparison)
    unsigned char** gate16; unsigned char** up16; unsigned char** down16;
    unsigned char* sg16; unsigned char* su16; unsigned char* sd16;
    // io
    float* x;      // [n, hidden]
    float* ids_f;  // [n, topk] f32-encoded expert ids
    float* probs;  // [n, topk]
    float* act;    // [n, topk*inter + inter_shared]
    float* out;    // [n, hidden]
    int e_local, hidden, inter, inter_shared, topk, n;
    float limit;
    cudaStream_t s;
};

static void launch_act_fp8(void* v) {
    MoeCtx* c = (MoeCtx*)v;
    CK(ferrite_moe_fused_act_fp8_mma(c->x, c->ids_f,
        (const void* const*)c->gate_w8, (const void* const*)c->gate_sc,
        (const void* const*)c->up_w8, (const void* const*)c->up_sc,
        c->sg_w8, c->sg_sc, c->su_w8, c->su_sc,
        c->act, 0, c->e_local, c->hidden, c->inter, c->inter_shared,
        c->topk, c->n, c->limit, c->s));
}
static void launch_down_fp8(void* v) {
    MoeCtx* c = (MoeCtx*)v;
    CK(ferrite_moe_fused_down_sum_fp8(c->ids_f, c->probs,
        (const void* const*)c->down_w8, (const void* const*)c->down_sc,
        c->sd_w8, c->sd_sc, c->act, c->out,
        0, c->e_local, c->hidden, c->inter, c->inter_shared,
        c->topk, c->n, (c->inter + 127) / 128, c->s));
}
static void launch_down_mma(void* v) {
    MoeCtx* c = (MoeCtx*)v;
    CK(ferrite_moe_down_mma(c->ids_f, c->probs,
        (const void* const*)c->down_w8, (const void* const*)c->down_sc,
        c->sd_w8, c->sd_sc, c->act, c->out,
        0, c->e_local, c->hidden, c->inter, c->inter_shared,
        c->topk, c->n, (c->inter + 127) / 128, c->s));
}
static void launch_act_bf16(void* v) {
    MoeCtx* c = (MoeCtx*)v;
    CK(ferrite_moe_fused_act(c->x, c->ids_f,
        (const void* const*)c->gate16, (const void* const*)c->up16,
        c->sg16, c->su16, c->act, 0, c->e_local, c->hidden, c->inter,
        c->inter_shared, c->topk, c->n, c->limit, c->s));
}
static void launch_down_bf16(void* v) {
    MoeCtx* c = (MoeCtx*)v;
    CK(ferrite_moe_fused_down_sum(c->ids_f, c->probs,
        (const void* const*)c->down16, c->sd16, c->act, c->out,
        0, c->e_local, c->hidden, c->inter, c->inter_shared,
        c->topk, c->n, c->s));
}

// ---------------- hc context ----------------
struct HcCtx {
    float *res, *fw, *scale, *base, *nw;
    float *li, *post, *comb, *mx_scratch;
    int s, n, h, mix, iters;
    float rms_eps, hc_eps;
    cudaStream_t s_;
};
static void launch_hc_pre(void* v) {
    HcCtx* c = (HcCtx*)v;
    CK(ferrite_hc_pre_split(c->res, c->fw, c->scale, c->base, c->nw,
        c->li, c->post, c->comb, c->mx_scratch,
        c->s, c->n, c->h, c->mix, c->rms_eps, c->hc_eps, c->iters, c->s_));
}

// random fp8 e4m3 bytes (valid range: avoid NaN 0x7F/0xFF)
static void rand_fp8(unsigned char* p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)(rand() % 254); // 0..253
}

int main(int argc, char** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 100;
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    srand(42);

    // ===== MoE (TP4 decode shapes) =====
    // N / I parameterizable: the real B=16 TP8 decode is n=16, inter=256
    // (TP4 was I=512); n=1/I=512 is the old MTP-verify-ish shape.
    const int E = 288, H = 4096, TOPK = 8;
    const int N = argc > 2 ? atoi(argv[2]) : 1;
    const int I = argc > 3 ? atoi(argv[3]) : 512;
    const int IS = 512;
    MoeCtx mc = {};
    mc.e_local = E; mc.hidden = H; mc.inter = I; mc.inter_shared = IS;
    mc.topk = TOPK; mc.n = N; mc.limit = 7.0f; mc.s = s;

    // host staging then H2D (pointer tables need device pointers)
    size_t w8_per = (size_t)I * H;              // gate/up fp8 [I,H]
    size_t dw8_per = (size_t)H * I;             // down fp8 [H,I]
    size_t sc_per = (size_t)((I + 127) / 128) * ((H + 127) / 128) * 4; // gate/up scales
    size_t dsc_per = (size_t)((H + 127) / 128) * ((I + 127) / 128) * 4; // down scales
    unsigned char* h_w8_gate = (unsigned char*)malloc(w8_per);
    unsigned char* h_w8_up = (unsigned char*)malloc(w8_per);
    unsigned char* h_w8_down = (unsigned char*)malloc(dw8_per);
    float* h_sc_gate = (float*)malloc(sc_per);
    float* h_sc_up = (float*)malloc(sc_per);
    float* h_sc_down = (float*)malloc(dsc_per);
    if (!h_w8_gate || !h_w8_up || !h_w8_down) { fprintf(stderr, "host alloc\n"); return 1; }
    rand_fp8(h_w8_gate, w8_per); rand_fp8(h_w8_up, w8_per); rand_fp8(h_w8_down, dw8_per);
    for (size_t i = 0; i < sc_per / 4; i++) { h_sc_gate[i] = 0.001f; h_sc_up[i] = 0.001f; h_sc_down[i] = 0.001f; }

    // device per-expert allocations
    unsigned char** h_gate_ptrs = (unsigned char**)malloc(E * sizeof(void*));
    unsigned char** h_up_ptrs = (unsigned char**)malloc(E * sizeof(void*));
    unsigned char** h_down_ptrs = (unsigned char**)malloc(E * sizeof(void*));
    float** h_gsc_ptrs = (float**)malloc(E * sizeof(void*));
    float** h_usc_ptrs = (float**)malloc(E * sizeof(void*));
    float** h_dsc_ptrs = (float**)malloc(E * sizeof(void*));
    unsigned char** h_gate16_ptrs = (unsigned char**)malloc(E * sizeof(void*));
    unsigned char** h_up16_ptrs = (unsigned char**)malloc(E * sizeof(void*));
    unsigned char** h_down16_ptrs = (unsigned char**)malloc(E * sizeof(void*));
    for (int e = 0; e < E; e++) {
        unsigned char *g, *u, *d; float *gs, *us, *ds;
        unsigned char *g16, *u16, *d16;
        CK(cudaMalloc(&g, w8_per)); CK(cudaMalloc(&u, w8_per)); CK(cudaMalloc(&d, dw8_per));
        CK(cudaMalloc(&gs, sc_per)); CK(cudaMalloc(&us, sc_per)); CK(cudaMalloc(&ds, dsc_per));
        CK(cudaMalloc(&g16, w8_per * 2)); CK(cudaMalloc(&u16, w8_per * 2)); CK(cudaMalloc(&d16, dw8_per * 2));
        CK(cudaMemcpy(g, h_w8_gate, w8_per, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(u, h_w8_up, w8_per, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d, h_w8_down, dw8_per, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(gs, h_sc_gate, sc_per, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(us, h_sc_up, sc_per, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds, h_sc_down, dsc_per, cudaMemcpyHostToDevice));
        // bf16 "weights": fp8 bytes widened on the HOST (one bulk H2D — the
        // per-element loop was 288×3×2M tiny memcpys, hours; here: 3 copies)
        static unsigned short* h16 = nullptr;
        if (!h16) { h16 = (unsigned short*)malloc(w8_per * 2); }
        for (size_t i = 0; i < w8_per; i++) h16[i] = (unsigned short)h_w8_gate[i] << 8;
        CK(cudaMemcpy(g16, h16, w8_per * 2, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < w8_per; i++) h16[i] = (unsigned short)h_w8_up[i] << 8;
        CK(cudaMemcpy(u16, h16, w8_per * 2, cudaMemcpyHostToDevice));
        if (!h16) { h16 = (unsigned short*)malloc(dw8_per * 2); }
        for (size_t i = 0; i < dw8_per; i++) h16[i] = (unsigned short)h_w8_down[i] << 8;
        CK(cudaMemcpy(d16, h16, dw8_per * 2, cudaMemcpyHostToDevice));
        h_gate_ptrs[e] = g; h_up_ptrs[e] = u; h_down_ptrs[e] = d;
        h_gsc_ptrs[e] = gs; h_usc_ptrs[e] = us; h_dsc_ptrs[e] = ds;
        h_gate16_ptrs[e] = g16; h_up16_ptrs[e] = u16; h_down16_ptrs[e] = d16;
    }
    // shared expert
    CK(cudaMalloc(&mc.sg_w8, w8_per)); CK(cudaMalloc(&mc.su_w8, w8_per)); CK(cudaMalloc(&mc.sd_w8, dw8_per));
    CK(cudaMalloc(&mc.sg_sc, sc_per)); CK(cudaMalloc(&mc.su_sc, sc_per)); CK(cudaMalloc(&mc.sd_sc, dsc_per));
    CK(cudaMalloc(&mc.sg16, w8_per * 2)); CK(cudaMalloc(&mc.su16, w8_per * 2)); CK(cudaMalloc(&mc.sd16, dw8_per * 2));
    CK(cudaMemcpy(mc.sg_w8, h_w8_gate, w8_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.su_w8, h_w8_up, w8_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.sd_w8, h_w8_down, dw8_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.sg_sc, h_sc_gate, sc_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.su_sc, h_sc_up, sc_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.sd_sc, h_sc_down, dsc_per, cudaMemcpyHostToDevice));

    // pointer tables → device
    CK(cudaMalloc(&mc.gate_w8, E * sizeof(void*))); CK(cudaMalloc(&mc.gate_sc, E * sizeof(void*)));
    CK(cudaMalloc(&mc.up_w8, E * sizeof(void*)));   CK(cudaMalloc(&mc.up_sc, E * sizeof(void*)));
    CK(cudaMalloc(&mc.down_w8, E * sizeof(void*))); CK(cudaMalloc(&mc.down_sc, E * sizeof(void*)));
    CK(cudaMalloc(&mc.gate16, E * sizeof(void*)));  CK(cudaMalloc(&mc.up16, E * sizeof(void*)));
    CK(cudaMalloc(&mc.down16, E * sizeof(void*)));
    CK(cudaMemcpy(mc.gate_w8, h_gate_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.up_w8, h_up_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.down_w8, h_down_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.gate_sc, h_gsc_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.up_sc, h_usc_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.down_sc, h_dsc_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.gate16, h_gate16_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.up16, h_up16_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mc.down16, h_down16_ptrs, E * sizeof(void*), cudaMemcpyHostToDevice));

    // io
    float* h_x = (float*)malloc(H * 4);
    for (int i = 0; i < H; i++) h_x[i] = 0.01f * (rand() % 200 - 100);
    float* h_ids = (float*)malloc(TOPK * 4);
    float* h_probs = (float*)malloc(TOPK * 4);
    for (int k = 0; k < TOPK; k++) { h_ids[k] = (float)((k * 37 + 11) % E); h_probs[k] = 0.12f; }
    CK(cudaMalloc(&mc.x, H * 4)); CK(cudaMemcpy(mc.x, h_x, H * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&mc.ids_f, TOPK * 4)); CK(cudaMemcpy(mc.ids_f, h_ids, TOPK * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&mc.probs, TOPK * 4)); CK(cudaMemcpy(mc.probs, h_probs, TOPK * 4, cudaMemcpyHostToDevice));
    size_t act_len = N * (TOPK * I + IS);
    CK(cudaMalloc(&mc.act, act_len * 4)); CK(cudaMemset(mc.act, 0, act_len * 4));
    CK(cudaMalloc(&mc.out, N * H * 4)); CK(cudaMemset(mc.out, 0, N * H * 4));

    printf("=== MoE fp8/bf16 kernels (TP4 decode: E=%d H=%d I=%d IS=%d topk=%d) ===\n", E, H, I, IS, TOPK);
    printf("bytes/call: act fp8 %.1f MB (bf16 %.1f MB) | down fp8 %.1f MB (bf16 %.1f MB)\n",
        (TOPK * 2.0 * I * H + 2.0 * IS * H) / 1e6, (TOPK * 2.0 * I * H + 2.0 * IS * H) * 2 / 1e6,
        (TOPK * 1.0 * H * I + 1.0 * H * IS) / 1e6, (TOPK * 1.0 * H * I + 1.0 * H * IS) * 2 / 1e6);
    float t_act8 = bench_kernel("moe_fused_act_fp8_mma", iters, launch_act_fp8, &mc);
    float t_down8 = bench_kernel("moe_fused_down_sum_fp8", iters, launch_down_fp8, &mc);
    // ===== correctness: MMA down vs SIMT down on the SAME act/ids/probs =====
    {
        size_t olen = (size_t)mc.n * mc.hidden;
        std::vector<float> a(olen), b(olen);
        CK(cudaMemset(mc.out, 0, olen * 4));
        launch_down_fp8(&mc);
        CK(cudaMemcpy(a.data(), mc.out, olen * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemset(mc.out, 0, olen * 4));
        launch_down_mma(&mc);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(b.data(), mc.out, olen * 4, cudaMemcpyDeviceToHost));
        double mx = 0, mxv = 0; int bad = 0;
        for (size_t i = 0; i < olen; i++) {
            double d = fabs((double)a[i] - (double)b[i]);
            double r = d / fmax(fabs((double)a[i]), 1e-3);
            if (r > mx) mx = r;
            if (fabs((double)a[i]) > mxv) mxv = fabs((double)a[i]);
            if (r > 5e-2) bad++;
        }
        printf("MMA-down vs SIMT-down: maxrel=%.3e bad=%d/%zu (max|ref|=%.4g)\n", mx, bad, olen, mxv);
        for (int t = 0; t < 2 && t < mc.n; t++) {
            printf("  tok%d: ref[0..3]=%.4g %.4g %.4g %.4g | mma[0..3]=%.4g %.4g %.4g %.4g\n", t,
                   a[(size_t)t*mc.hidden+0], a[(size_t)t*mc.hidden+1], a[(size_t)t*mc.hidden+2], a[(size_t)t*mc.hidden+3],
                   b[(size_t)t*mc.hidden+0], b[(size_t)t*mc.hidden+1], b[(size_t)t*mc.hidden+2], b[(size_t)t*mc.hidden+3]);
        }
    }
    float t_act16 = bench_kernel("moe_fused_act (bf16 cmp)", iters, launch_act_bf16, &mc);
    float t_down16 = bench_kernel("moe_fused_down_sum (bf16 cmp)", iters, launch_down_bf16, &mc);
    printf("fp8 vs bf16: act %.1f%% down %.1f%%\n",
        t_act8 / t_act16 * 100, t_down8 / t_down16 * 100);

    // ===== hc_pre (decode: s=1, n=4, h=4096, mix=24) =====
    HcCtx hc = {};
    hc.s = 1; hc.n = 4; hc.h = 4096; hc.mix = 24; hc.iters = 4;
    hc.rms_eps = 1e-5f; hc.hc_eps = 1e-4f; hc.s_ = s;
    int nh = hc.n * hc.h;           // 16384
    int mx_mix = hc.mix;             // 24 (fn_w rows)
    CK(cudaMalloc(&hc.res, nh * 4)); CK(cudaMemset(hc.res, 0, nh * 4));
    float* h_res = (float*)malloc(nh * 4);
    for (int i = 0; i < nh; i++) h_res[i] = 0.001f * (rand() % 100 - 50);
    CK(cudaMemcpy(hc.res, h_res, nh * 4, cudaMemcpyHostToDevice));
    float* h_fw = (float*)malloc((size_t)mx_mix * nh * 4);
    for (int i = 0; i < mx_mix * nh; i++) h_fw[i] = 0.0001f * (rand() % 100 - 50);
    CK(cudaMalloc(&hc.fw, (size_t)mx_mix * nh * 4));
    CK(cudaMemcpy(hc.fw, h_fw, (size_t)mx_mix * nh * 4, cudaMemcpyHostToDevice));
    float h_scale[3] = {0.07f, 0.07f, 0.07f}, h_base[24];
    for (int i = 0; i < 24; i++) h_base[i] = -7.6f + 0.1f * i;
    CK(cudaMalloc(&hc.scale, 3 * 4)); CK(cudaMemcpy(hc.scale, h_scale, 3 * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&hc.base, 24 * 4)); CK(cudaMemcpy(hc.base, h_base, 24 * 4, cudaMemcpyHostToDevice));
    float* h_nw = (float*)malloc(4096 * 4);
    for (int i = 0; i < 4096; i++) h_nw[i] = 1.45f;
    CK(cudaMalloc(&hc.nw, 4096 * 4)); CK(cudaMemcpy(hc.nw, h_nw, 4096 * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&hc.li, 4096 * 4)); CK(cudaMalloc(&hc.post, 4 * 4)); CK(cudaMalloc(&hc.comb, 16 * 4));
    size_t mx_len = (size_t)hc.s * (mx_mix * 8 + 8 + 1 + hc.n);
    CK(cudaMalloc(&hc.mx_scratch, mx_len * 4)); CK(cudaMemset(hc.mx_scratch, 0, mx_len * 4));
    printf("\n=== hc_pre_split (decode: s=1 n=4 h=4096 mix=24 iters=4) ===\n");
    bench_kernel("ferrite_hc_pre_split (mix+rest)", iters, launch_hc_pre, &hc);

    printf("\nncu hint: sudo /usr/local/cuda-13.2/bin/ncu --set full --launch-count 3 -k \"regex:moe_fused|hc_pre\" /tmp/ncu_moe_bench 3\n");
    return 0;
}
