// wq_a_shim.cu — TileLang 投影 GEMM 的 launcher shim（第二阶段：wq_a 形状）。
//
// 与第一阶段 `wkv_shim.cu` 同构（契约、三条硬约束、rc 语义、两次 launch + 常驻 scratch 的
// 理由都见该文件头与 docs/agent/tilelang-integration-design.md §3.3）。本文件只列与 wkv 的
// **生成差异**，全部来自 gen_proj_shapes_aot.py 的文件头：
//
//   * OUT ROW STRIDE IS `OS`, NOT `n`. 归约内核的输出声明是 `C: T.Tensor((MPAD, OS))`，
//     store 的行 stride 被烘成 OS —— 调用点真实的 `out_stride`。wq_a 的三个站点
//     （verify 的 proj_mrows / eager 的 lin / lin2）恰好都是 `out_stride == q_lora_rank == n`，
//     所以 OS = N = 1280；形状门据此要求 `out_stride == OS`。
//   * 运行期 m 谓词：激活 staging 带 `if i < m`（行 >= m 写 0，不越界读），归约 store 带
//     同样的谓词 ⇒ 直接写 ferrite 的 `out`，**不需要 16 行 pad / m 行回拷**。
//
// =============================================================================
// ABI（与 `dsv41_gemm_fp8_mrows` 同形）
// =============================================================================
//   extern "C" int dsv41_gemm_fp8_tilelang_wq_a(
//       const uint8_t* a, const float* a_scale,      // 激活 fp8 [m, k] + [m, k/32] f32
//       const uint8_t* w, const uint8_t* w_scale,    // 权重 fp8 [n, k] + [n/32, k/32] ue8m0
//       const float* bias,                           // 本阶段必须 null（见下方 decline）
//       float* out,                                  // f32 [m, n]，行 stride == OS
//       int m, int n, int k, int out_stride, cudaStream_t s);
//   -> 0 = 已发射；2 = DECLINED（调用方保持既有路径）；其它 = cuda 错误码（真失败）
//
// rc 语义照 `dsv41_gemm_fp8_mrows` / `wkv_shim.cu`：`2` = declined，**永不返回 1**
// （1 是 cudaErrorInvalidValue，与真实发射失败不可区分）。

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 生成物：改名后 include（两份 dump 都叫 main_kernel）。
#define main_kernel wq_a_tl_partial_kernel
#include "wq_a_partial_tl.cu"
#undef main_kernel
#define main_kernel wq_a_tl_reduce_kernel
#include "wq_a_reduce_tl.cu"
#undef main_kernel

namespace {

// ---------------------------------------------------------------------------
// 冻结几何 —— 逐项来自 tilelang_gen/proj_shapes_tl_config.txt（生成器写出，勿手改）。
// ---------------------------------------------------------------------------
constexpr int kTLN = 1280;          // 权重/输出的 n（wq_a: q_lora_rank）
constexpr int kTLK = 5120;          // 归约维度 k（wq_a: dim）
constexpr int kTLOS = 1280;         // 输出行 stride（烘进归约内核的 `OS` = q_lora_rank）
constexpr int kTLBN = 128;          // partial 的 N 向 tile
constexpr int kTLKS = 8;            // K-split 分片数
constexpr int kTLThreads = 128;     // partial block
constexpr int kTLSmem = 13824;      // partial 动态 smem（NS*(16*32 + 128*32)）
constexpr int kTLRedBN = 256;       // reduce 的 N 向 tile
constexpr int kTLRedThreads = 256;  // reduce block
constexpr int kTLMPad = 16;         // mma m16 的 M（激活的 pad 目标，生成物内部使用）
constexpr int kTLMRows = 8;         // 本阶段接受的 m 上界（与 mrows 族一致）

// 常驻 scratch：P[KS][MPAD][N] f32。INIT 期分配，进程生命周期内复用。
float* g_part = nullptr;

// INIT：只做一次。返回 false ⇒ 本 shim 永不发射（调用方保持老路径）。
bool tl_wq_a_init() {
    static int state = 0;  // 0 = 未初始化, 1 = 已就绪, -1 = 初始化失败
    if (state != 0) return state > 0;
    // INIT-TIME ONLY（见 wkv_shim.cu 约束 1）。smem 只有 13.8 KiB（< 48 KiB 默认上限），
    // 这里的 SetAttribute 是契约对齐，失败不致命。
    (void)cudaFuncSetAttribute(wq_a_tl_partial_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, kTLSmem);
    (void)cudaGetLastError();  // 吞掉 SetAttribute 的潜在错误，别污染后面的 launch 检查
    const size_t bytes = (size_t)kTLKS * (size_t)kTLMPad * (size_t)kTLN * sizeof(float);
    if (cudaMalloc(&g_part, bytes) != cudaSuccess) {
        g_part = nullptr;
        (void)cudaGetLastError();
        // P1 (no-latch-death): -1 is a PERMANENT latch, kept deliberately. Every
        // failure reachable here is deterministic (cudaMalloc OOM / SetAttribute
        // rejection are repeatable), and the transient "called inside a capture"
        // case is intercepted by the P0-2 guard at the entry point — INIT is never
        // entered while capturing. A retry could not rescue a latched failure, so
        // re-attempting each call would only re-pay a guaranteed-to-fail init.
        state = -1;
        return false;
    }
    state = 1;
    return true;
}

}  // namespace

// 返回 0（已发射）/ 2（DECLINED：调用方保持老路径）/ 其它 cuda 错误码。
extern "C" int dsv41_gemm_fp8_tilelang_wq_a(const uint8_t* a, const float* a_scale,
                                            const uint8_t* w, const uint8_t* w_scale,
                                            const float* bias, float* out, int m, int n, int k,
                                            int out_stride, cudaStream_t s) {
    // ---- 形状门（第二阶段 = 冻结的 wq_a 形状）----
    if (m < 1 || m > kTLMRows) return 2;
    if (n != kTLN || k != kTLK) return 2;
    // 归约 store 的行 stride 被烘成 OS（生成物）。行 stride 只在 m > 1 参与地址计算
    // （第 0 行恒在 offset 0）⇒ m == 1 时放宽（eager 单行站点按 `n_out` 约定传 stride，
    // 而 wq_b 的 OS = nh*hd ≠ n_out = nlh*hd；单行下两者等价）。m > 1 严格要求 == OS。
    if (m > 1 && out_stride != kTLOS) return 2;
    if (a == nullptr || a_scale == nullptr || w == nullptr || w_scale == nullptr || out == nullptr)
        return 2;
    // 本阶段的生成物没有 bias 通路 ⇒ 带 bias 的调用一律 decline（绝不静默丢 bias）。
    if (bias != nullptr) return 2;
    // 对齐：partial 的 A staging 是 4B 组的读写，W 走 cp.async 16B，reduce 的 store 是
    // 32B（tl::store_global_256）。基址不对齐在这些写法下是 err 716 或静默错位，一律 decline。
    if ((((uintptr_t)a & 0xF) != 0) || (((uintptr_t)a_scale & 0xF) != 0) ||
        (((uintptr_t)w & 0xF) != 0) || (((uintptr_t)w_scale & 0xF) != 0) ||
        (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit): NEVER run cudaMalloc inside a capture — it
    // invalidates the caller's capture BEFORE we could decline. Decline here.
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone) {
        return 2;  // decline without touching capture
    }

    if (!tl_wq_a_init()) {
        // 初始化失败（分配不到 scratch / 恰好发生在 capture 内）：decline，不发射。
        // 一次性提示，避免"armed 但每次静默测老路"（本项目 #1 测量偏置陷阱）。
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang] ARMED but INIT FAILED (scratch alloc) -> this run measures "
                    "the OLD path (m=%d n=%d k=%d)\n",
                    m, n, k);
        return 2;
    }

    // 一次性活性回执（照 [mrows-mtile] ARMED 的先例：kernel 名只在日志里出现，
    // 这是"ON 的臂真的跑了新程序"的唯一证据）。
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang] ARMED wq_a m=%d n=%d k=%d ks=%d os=%d -> grid=(%d,%d)x%d "
                    "smem=%d + reduce grid=%d x %d\n",
                    m, n, k, kTLKS, kTLOS, kTLN / kTLBN, kTLKS, kTLThreads, kTLSmem,
                    kTLN / kTLRedBN, kTLRedThreads);
    }

    // (1) K-split 分片：grid (N/BN, KS)，每块算一段 K 的 partial 写 P[kp]。
    // 生成物的 fp8 形参是 tl_templates 的 `fp8_e4_t` / `fp8_e8_t`，ferrite 的 ABI 是裸
    // `uint8_t*` ⇒ 这里 reinterpret_cast 对齐两种契约。
    wq_a_tl_partial_kernel<<<dim3((unsigned)(kTLN / kTLBN), (unsigned)kTLKS), kTLThreads,
                             kTLSmem, s>>>(reinterpret_cast<const fp8_e4_t*>(a), a_scale, g_part,
                                           reinterpret_cast<const fp8_e4_t*>(w),
                                           reinterpret_cast<const fp8_e8_t*>(w_scale), m);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    // (2) 确定性归约：按 kp 升序求和 ks 个 partial，只写前 m 行（行 stride = OS = out_stride）。
    wq_a_tl_reduce_kernel<<<dim3((unsigned)(kTLN / kTLRedBN)), kTLRedThreads, 0, s>>>(out, g_part, m);
    e = cudaGetLastError();
    return (int)e;
}
