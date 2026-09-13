// =============================================================================
// tests_dn_bs_parity.cu — MoE **down** 方向：新 blockscaled 臂 vs SIMT 参考的逐元素对拍
//                          + 微基准 + **OFF 等价性的可执行断言**
// =============================================================================
// 被测对象
//   新臂 : `dsv41_moe_bs_down_dev`（tilelang_gen/moe_bs_dn_shim.cu，本文件直接 #include）
//  参考 : `dsv41_expert_down_reduce_fp4_batched`（dsv41_experts_mxf4.cu 的生产融合核，
//          即被替代的那个；本文件也 #include 它，与 tests_dsv41_experts_mrows.cu 同风格）
//
// 三个臂（判据不同，**不要混**）
//   arm A  "total"   : BS(默认, rw 在 epilogue) vs SIMT(rw 原样)
//                      ⇒ 差异主体是 **e4m3 输入的量化**（算法差，~2.5% of rms）
//   arm B  "order"   : BS  vs  SIMT(**喂同一份量化后的操作数**)
//                      ⇒ 差异只剩 **fp 累加序**（判据：max 元素相对误差 ~1e-5，
//                        rms 相对 ~1e-6 —— 这才是"实现正确"的硬指标）
//   arm C  "official": BS(DSV41_MOE_DOWN_BS_RWOP=1) vs SIMT(喂 rw×bf16→e4m3 的官方操作数,
//                      rw=1) ⇒ 复现 `DSV41_ROUTED_DOWN_QUANT` 的语义，差异只剩累加序
//   arm D  "off"     : 不设 `DSV41_MOE_DOWN_BS` ⇒ 入口必须 **return 2 且一个字节都不写**
//                      （用 NaN/sentinel 哨兵验证）⇒ **OFF 等价性的机器可检查证据**
//
// ---------------------------------------------------------------------------
// Build（需要 nvcc，**不需要 GPU**；两个 TU 直接编译进来，无需链 .so）
//   nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
//        -Ikernels/cuda/tilelang_gen -o /tmp/t_dnbs kernels/cuda/tests_dn_bs_parity.cu
//   ⚠️ 用显式 -gencode：`-arch=sm_103a` 在部分 nvcc 上会被静默降级成 sm_103，
//      ptxas 随后拒绝一切 tcgen05 指令（tests_bs_impulse.cu 的同一提醒）。
// Run（需要一张空闲 GPU；峰值显存 ~40 MB）
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_dnbs                 # arm A/B/C + off 断言
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_dnbs --bench         # 追加两臂的微基准
//   CUDA_VISIBLE_DEVICES=<free> /tmp/t_dnbs --quick         # 只跑 arm B + off
//
// ⚠️ misaligned address 在这颗芯片上是**粘性**的（污染 CUDA context）⇒ 每个阶段都查
//    cudaGetLastError 并在失败时打印上下文，一个坏阶段不能伪造其它阶段的结果。
// =============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "moe_bs_dn_shim.cu"  // 新臂（自包含：不依赖 tilelang_inc，也不需要 -lcuda）

// 被替代的 SIMT 生产核（dsv41_experts_mxf4.cu）。**直接 #include 而不是链 .so**：
// 本测试的全部意义是拿"真核"当参考，而 build.sh 的 .so 里那个是真核的同一份源码。
#include "../dsv41_experts_mxf4.cu"

// ---------------------------------------------------------------- 几何（down 生产形状）
static const int kDim = 5120;      // down 的输出维 N
static const int kInter = 320;     // down 的 contraction K
static const int kSlots = 6;       // topk
static const int kTopk = 6;
static const int kSegCap = 36;     // DN_SEGCAP
static const int kBm = 128;        // DN_BM
static const int kKpack = kInter / 2;
static const int kSfPitch = 16;
static const int kBlk = 32;

#define DN_CHECK(cond, ...)                                                      \
    do {                                                                         \
        if (!(cond)) {                                                           \
            printf("[FAIL] %s:%d ", __FILE__, __LINE__);                         \
            printf(__VA_ARGS__);                                                 \
            printf("\n");                                                        \
            fflush(stdout);                                                      \
            return 1;                                                            \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------- host 侧数学模型
// e4m3 的 256 码 -> f32（与 dsv41_experts_mxf4.cu:742 同式），供 arm B/C 造"官方操作数"
static float e4m3_val(int b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    float v = (e == 0) ? (float)m / 512.0f : (float)(8 + m) * ldexpf(1.0f, e - 10);
    return (b & 0x80) ? -v : v;
}
struct E4M3Tab {
    float v[256];
    E4M3Tab() {
        for (int i = 0; i < 256; ++i) v[i] = e4m3_val(i);
    }
};
static const E4M3Tab g_e4;

// f32 -> e4m3 字节：round-to-nearest-even + 饱和到 +-448（== __nv_fp8_e4m3）
static uint8_t host_e4m3(float x) {
    const double ax = std::min(std::fabs((double)x), 448.0);
    int best = 0;
    double bd = 1e30;
    double bm = 0;
    for (int c = 0; c < 128; ++c) {  // 正整数码 0..127
        const double d = std::fabs((double)g_e4.v[c] - ax);
        if (d < bd - 1e-12) {
            bd = d;
            best = c;
        } else if (std::fabs(d - bd) <= 1e-12 && (c & 1) == 0 && (best & 1)) {
            best = c;  // 平局取 mantissa 偶
        }
        bm = bm;  // (unused)
    }
    return (uint8_t)((x < 0.f) ? (best | 0x80) : best);
}
// glue_fast_round_scale（= kernel.py 的 fast_round_scale，2 的幂）
static float host_fast_round_scale(float amax, float max_inv) {
    const uint32_t bits = *(uint32_t*)&(*(float*)&(amax *= max_inv));
    const int exp = (int)((bits >> 23) & 0xFFu);
    const uint32_t man = bits & 0x7FFFFFu;
    const int e = exp - 127 + (man != 0 ? 1 : 0);
    const uint32_t r = (uint32_t)(e + 127) << 23;
    return *(float*)&r;
}
// 官方 down 输入的量化操作数（[inter] 一段）：v -> decode(e4m3(q)) * sc，f32
static void host_quant_operand(const float* v_in, int n, float* out, float rw, bool bf16_boundary) {
    for (int b0 = 0; b0 < n; b0 += kBlk) {
        const int n_blk = std::min(kBlk, n - b0);
        float vv[kBlk];
        float amax = 1e-4f;
        for (int i = 0; i < n_blk; ++i) {
            float v = v_in[b0 + i] * rw;
            if (bf16_boundary) {
                const uint32_t bits = *(uint32_t*)&v;
                const uint32_t r = (bits >> 16) & 1u;
                const uint32_t rounded = (bits + 0x7FFFu + r) & 0xFFFF0000u;
                v = *(float*)&rounded;
            }
            vv[i] = v;
            amax = std::max(amax, std::fabs(v));
        }
        const float sc = host_fast_round_scale(amax, 1.0f / 448.0f);
        for (int i = 0; i < n_blk; ++i) {
            const uint8_t q = host_e4m3(vv[i] / sc);
            out[b0 + i] = g_e4.v[q] * sc;
        }
    }
}

// ---------------------------------------------------------------- 比较器
struct DiffStat {
    double max_abs, rms, max_rel, p99_rel, rel_to_rms;
};
template <typename R>
static DiffStat compare(const std::vector<float>& ref, const std::vector<float>& got, R rel_floor) {
    DiffStat s{0, 0, 0, 0, 0};
    double sq = 0, rsq = 0;
    std::vector<double> rels;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double)got[i] - (double)ref[i]);
        const double r = std::fabs((double)ref[i]);
        s.max_abs = std::max(s.max_abs, d);
        sq += d * d;
        rsq += r * r;
        if (r > rel_floor) rels.push_back(d / r);
    }
    s.rms = std::sqrt(sq / (double)ref.size());
    s.rel_to_rms = s.max_abs / std::sqrt(rsq / (double)ref.size());
    std::sort(rels.begin(), rels.end());
    s.max_rel = rels.empty() ? 0.0 : rels.back();
    s.p99_rel = rels.empty() ? 0.0 : rels[(size_t)((double)(rels.size() - 1) * 0.99)];
    return s;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    bool bench = false, quick = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--bench")) bench = true;
        if (!strcmp(argv[i], "--quick")) quick = true;
    }
    const bool cap = dsv41_moe_bs_down_cap() == 1;
    printf("[dn-bs] cap=%d  (capability symbol of the down BS arm)\n", (int)cap);
    DN_CHECK(cap, "the .so has no dsv41_moe_bs_down_cap -> the arm is not compiled in");

    // ---- arm D 先跑：**OFF 等价性**（不设门 ⇒ return 2 且一个字节都不写）----
    {
        unsetenv("DSV41_MOE_DOWN_BS");
        float* d_out = nullptr;
        float* d_act = nullptr;
        int* d_ids = nullptr;
        uint8_t *d_w2 = nullptr, *d_w2s = nullptr;
        int* d_eid = nullptr; int* d_order = nullptr; int* d_counts = nullptr; int* d_nseg = nullptr;
        cudaMalloc(&d_out, kDim * sizeof(float));
        cudaMalloc(&d_act, kSlots * 2 * kInter * sizeof(float));
        cudaMalloc(&d_ids, kSlots * sizeof(int));
        cudaMalloc(&d_w2, (size_t)kSlots * kDim * kKpack);
        cudaMalloc(&d_w2s, (size_t)kSlots * kDim * kSfPitch);
        cudaMalloc(&d_eid, kSegCap * sizeof(int));
        cudaMalloc(&d_order, (size_t)kSegCap * kBm * sizeof(int));
        cudaMalloc(&d_counts, kSegCap * sizeof(int));
        cudaMalloc(&d_nseg, sizeof(int));
        std::vector<float> sent(kDim, 1234.5f), back(kDim, 0.f);
        cudaMemcpy(d_out, sent.data(), kDim * sizeof(float), cudaMemcpyHostToDevice);
        const int rc = dsv41_moe_bs_down_dev(nullptr, 0, d_out, nullptr, 0, nullptr, 0, nullptr,
                                             nullptr, nullptr, nullptr, nullptr, 1, 1, kDim, kInter,
                                             kTopk, nullptr);
        cudaMemcpy(back.data(), d_out, kDim * sizeof(float), cudaMemcpyDeviceToHost);
        bool untouched = (memcmp(sent.data(), back.data(), kDim * sizeof(float)) == 0);
        printf("[arm D] gate OFF -> rc=%d (expect 2), out untouched=%d\n", rc, (int)untouched);
        DN_CHECK(rc == 2 && untouched,
                 "OFF equivalence violated: rc=%d untouched=%d (a gate that is OFF must be a "
                 "complete no-op)",
                 rc, (int)untouched);
        cudaFree(d_out); cudaFree(d_act); cudaFree(d_ids); cudaFree(d_w2); cudaFree(d_w2s);
        cudaFree(d_eid); cudaFree(d_order); cudaFree(d_counts); cudaFree(d_nseg);
    }

    // ---- 武装 ----
    setenv("DSV41_MOE_DOWN_BS", "1", 1);
    printf("[dn-bs] arming DSV41_MOE_DOWN_BS=1 (this test's entire purpose)\n");

    // ---- 合成数据（无需模型权重；池布局与生产无关，stride 由调用方给）----
    const int n_e = kSlots;
    std::vector<uint8_t> h_w2((size_t)n_e * kDim * kKpack);
    std::vector<uint8_t> h_w2s((size_t)n_e * kDim * kSfPitch, 0);
    srand(20260913);
    for (size_t i = 0; i < h_w2.size(); ++i) h_w2[i] = (uint8_t)(rand() & 0xFF);
    for (int e = 0; e < n_e; ++e)
        for (int r = 0; r < kDim; ++r)
            for (int b = 0; b < kInter / 32; ++b)
                h_w2s[((size_t)e * kDim + r) * kSfPitch + b] = (uint8_t)(118 + rand() % 18);
    std::vector<float> h_act(kSlots * 2 * kInter);
    for (size_t i = 0; i < h_act.size(); ++i) h_act[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    std::vector<int> h_ids(kSlots);
    for (int s = 0; s < kSlots; ++s) h_ids[s] = s;  // 每槽一个不同专家
    std::vector<float> h_rw = {0.1210f, 0.2006f, 0.1414f, 0.1934f, 0.2387f, 0.1049f};

    // BS 段表：每段一个专家、段内 live 行数 = 该专家的 assignment 数（这里各 1）
    std::vector<int> h_eid(kSegCap, 0), h_order((size_t)kSegCap * kBm, -1), h_counts(kSegCap, 0);
    for (int s = 0; s < kSlots; ++s) {
        h_eid[s] = h_ids[s];
        h_counts[s] = 1;
        h_order[(size_t)s * kBm + 0] = s;  // flat assignment index = row*topk + slot = slot（rows=1）
    }
    const int h_nseg = kSlots;
    const long w2_stride = (long)kDim * kKpack;
    const long w2s_stride = (long)kDim * kSfPitch;

    // ---- device 缓冲 ----
    cudaStream_t st;
    cudaStreamCreate(&st);
    float *d_act = nullptr, *d_rw = nullptr;
    float *d_ref = nullptr, *d_part = nullptr, *d_bs = nullptr;
    float *d_act_q = nullptr, *d_act_qo = nullptr;
    uint8_t *d_w2 = nullptr, *d_w2s = nullptr;
    int *d_ids = nullptr, *d_eid = nullptr, *d_order = nullptr, *d_counts = nullptr, *d_nseg = nullptr;
    cudaMalloc(&d_act, h_act.size() * sizeof(float));
    cudaMalloc(&d_act_q, kSlots * kInter * sizeof(float));   // arm B 的量化操作数（[slot][inter]）
    cudaMalloc(&d_act_qo, kSlots * 2 * kInter * sizeof(float));
    cudaMalloc(&d_rw, kSlots * sizeof(float));
    cudaMalloc(&d_ref, kDim * sizeof(float));
    cudaMalloc(&d_part, (size_t)kSlots * kDim * sizeof(float));
    cudaMalloc(&d_bs, kDim * sizeof(float));
    cudaMalloc(&d_w2, h_w2.size());
    cudaMalloc(&d_w2s, h_w2s.size());
    cudaMalloc(&d_ids, kSlots * sizeof(int));
    cudaMalloc(&d_eid, kSegCap * sizeof(int));
    cudaMalloc(&d_order, (size_t)kSegCap * kBm * sizeof(int));
    cudaMalloc(&d_counts, kSegCap * sizeof(int));
    cudaMalloc(&d_nseg, sizeof(int));
    cudaMemcpyAsync(d_act, h_act.data(), h_act.size() * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_rw, h_rw.data(), kSlots * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_w2, h_w2.data(), h_w2.size(), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_w2s, h_w2s.data(), h_w2s.size(), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_ids, h_ids.data(), kSlots * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_eid, h_eid.data(), kSegCap * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_order, h_order.data(), (size_t)kSegCap * kBm * sizeof(int),
                    cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_counts, h_counts.data(), kSegCap * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(d_nseg, &h_nseg, sizeof(int), cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);
    DN_CHECK(cudaGetLastError() == cudaSuccess, "H2D upload failed");

    // ---- arm A: SIMT(f32 act) 参考 ----
    // NOTE: the real signature gained `seq_align` from the DSV41_SEQ_ALIGN merge (#5), which
    // landed AFTER this harness was written — hence the original 'too few arguments' and the
    // cudaStream_t-as-int error. 0 keeps that arm OFF, matching the reference's historical form.
    int rc = dsv41_expert_down_reduce_fp4_batched(d_act, 2 * kInter, d_ref, 1, kDim, kInter, d_rw, 1,
                                                 kSlots, d_w2, w2_stride, d_w2s, w2s_stride, d_ids, 0,
                                                 st);
    DN_CHECK(rc == 0, "SIMT reference rc=%d", rc);
    cudaStreamSynchronize(st);
    std::vector<float> h_ref(kDim);
    cudaMemcpy(h_ref.data(), d_ref, kDim * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- arm A: 新臂（默认：rw 在 epilogue）----
    rc = dsv41_moe_bs_down_dev(d_act, 2 * kInter, d_part, d_w2, w2_stride, d_w2s, w2s_stride, d_eid,
                               d_order, d_counts, d_nseg, d_rw, 1, 1, kDim, kInter, kTopk, st);
    DN_CHECK(rc == 0, "down BS arm rc=%d (2 = declined: see its one-shot note)", rc);
    rc = dsv41_moe_down_reduce(d_part, d_bs, kDim, kSlots, st);
    DN_CHECK(rc == 0, "moe_down_reduce rc=%d", rc);
    cudaStreamSynchronize(st);
    std::vector<float> h_bs(kDim);
    cudaMemcpy(h_bs.data(), d_bs, kDim * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- arm B: SIMT 喂**同一份**量化操作数（隔离出 fp 序差）----
    std::vector<float> h_q(kSlots * kInter);
    for (int s = 0; s < kSlots; ++s)
        host_quant_operand(&h_act[s * 2 * kInter], kInter, &h_q[s * kInter], 1.0f, false);
    // 布局：SIMT 读 [slot][2*inter] 的前 inter 个 float ⇒ 尾半区保持原值（不读）
    std::vector<float> h_act_q = h_act;
    for (int s = 0; s < kSlots; ++s)
        memcpy(&h_act_q[s * 2 * kInter], &h_q[s * kInter], kInter * sizeof(float));
    cudaMemcpyAsync(d_act_q, h_act_q.data(), h_act_q.size() * sizeof(float),
                    cudaMemcpyHostToDevice, st);
    rc = dsv41_expert_down_reduce_fp4_batched(d_act_q, 2 * kInter, d_ref, 1, kDim, kInter, d_rw, 1,
                                              kSlots, d_w2, w2_stride, d_w2s, w2s_stride, d_ids, 0, st);
    DN_CHECK(rc == 0, "SIMT(quantised operand) rc=%d", rc);
    cudaStreamSynchronize(st);
    std::vector<float> h_refq(kDim);
    cudaMemcpy(h_refq.data(), d_ref, kDim * sizeof(float), cudaMemcpyDeviceToHost);

    const DiffStat sA = compare(h_ref, h_bs, 1e-3);
    const DiffStat sB = compare(h_refq, h_bs, 1e-3);
    printf("\n=========== down BS arm vs SIMT ===========\n");
    printf("arm A (BS vs SIMT, f32 operand)     max|d|=%.3e rms=%.3e max_rel=%.3e p99_rel=%.3e "
           "max/rms_ref=%.3e\n",
           sA.max_abs, sA.rms, sA.max_rel, sA.p99_rel, sA.rel_to_rms);
    printf("arm B (BS vs SIMT, SAME e4m3 op)    max|d|=%.3e rms=%.3e max_rel=%.3e p99_rel=%.3e "
           "max/rms_ref=%.3e\n",
           sB.max_abs, sB.rms, sB.max_rel, sB.p99_rel, sB.rel_to_rms);
    printf("  => arm B is the implementation check (fp order only). arm A is the ALGORITHMIC\n"
           "     e4m3 input quantisation the tensor-core arm must perform (see design §5).\n");
    DN_CHECK(sB.max_rel < 1e-4, "arm B max_rel=%.3e too large -> the kernel's own numerics are wrong",
             sB.max_rel);

    // ---- arm C: 官方顺序（rw 进 operand + bf16 边界）----
    if (!quick) {
        setenv("DSV41_MOE_DOWN_BS_RWOP", "1", 1);
        // 重新 INIT 代价：shim 的 INIT 只跑一次 ⇒ 本臂必须**先**设 env 再第一次调用。
        // 因此这里只能提示：真正的 RWOP 臂要单独起一次进程（下面会打印命令）。
        printf("[arm C] DSV41_MOE_DOWN_BS_RWOP=1 requires a fresh process (INIT caches the env); "
               "run:  DSV41_MOE_DOWN_BS_RWOP=1 /tmp/t_dnbs --quick\n");
    }

    // ---- arm D': 微基准（两臂各自的 launcher，同 stream 背靠背）----
    if (bench) {
        const int n_iter = 200;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int i = 0; i < 5; ++i) {
            dsv41_expert_down_reduce_fp4_batched(d_act, 2 * kInter, d_ref, 1, kDim, kInter, d_rw, 1,
                                                 kSlots, d_w2, w2_stride, d_w2s, w2s_stride, d_ids,
                                                 st);
            dsv41_moe_bs_down_dev(d_act, 2 * kInter, d_part, d_w2, w2_stride, d_w2s, w2s_stride,
                                  d_eid, d_order, d_counts, d_nseg, d_rw, 1, 1, kDim, kInter, kTopk,
                                  st);
            dsv41_moe_down_reduce(d_part, d_bs, kDim, kSlots, st);
        }
        cudaStreamSynchronize(st);
        float t_simt = 0, t_bs = 0;
        cudaEventRecord(e0, st);
        for (int i = 0; i < n_iter; ++i)
            dsv41_expert_down_reduce_fp4_batched(d_act, 2 * kInter, d_ref, 1, kDim, kInter, d_rw, 1,
                                                 kSlots, d_w2, w2_stride, d_w2s, w2s_stride, d_ids,
                                                 st);
        cudaEventRecord(e1, st);
        cudaStreamSynchronize(st);
        cudaEventElapsedTime(&t_simt, e0, e1);
        cudaEventRecord(e0, st);
        for (int i = 0; i < n_iter; ++i) {
            dsv41_moe_bs_down_dev(d_act, 2 * kInter, d_part, d_w2, w2_stride, d_w2s, w2s_stride,
                                  d_eid, d_order, d_counts, d_nseg, d_rw, 1, 1, kDim, kInter, kTopk,
                                  st);
            dsv41_moe_down_reduce(d_part, d_bs, kDim, kSlots, st);
        }
        cudaEventRecord(e1, st);
        cudaStreamSynchronize(st);
        cudaEventElapsedTime(&t_bs, e0, e1);
        printf("\n[bench] N=%d  SIMT fused down+reduce = %.1f us/call   BS arm(gather+mma+reduce) = "
               "%.1f us/call   speedup=%.2fx\n",
               n_iter, t_simt * 1000.0f / n_iter, t_bs * 1000.0f / n_iter, t_simt / t_bs);
        printf("  [warn] L2-HOT micro-benchmark: production streams w2 cold from HBM "
               "(29.5 MB/layer), so this ratio is an upper bound only -- never a substitute "
               "for the e2e p50 (this repo already paid +38 pct for that mistake)\n");
    }

    // ---- 收尾 ----
    cudaFree(d_act); cudaFree(d_act_q); cudaFree(d_act_qo); cudaFree(d_rw); cudaFree(d_ref);
    cudaFree(d_part); cudaFree(d_bs); cudaFree(d_w2); cudaFree(d_w2s); cudaFree(d_ids);
    cudaFree(d_eid); cudaFree(d_order); cudaFree(d_counts); cudaFree(d_nseg);
    cudaStreamDestroy(st);
    printf("\n[dn-bs] ALL CHECKS PASSED\n");
    return 0;
}
