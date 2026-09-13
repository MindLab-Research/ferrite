// wo_b_shim.cu — TileLang 投影 GEMM 的 launcher shim（第二阶段：wo_b 形状）。
//
// 与第一阶段 `wkv_shim.cu` 同构（契约、三条硬约束、rc 语义、两次 launch + 常驻 scratch 的
// 理由都见该文件头与 docs/agent/tilelang-integration-design.md §3.3）。本文件只列与 wkv 的
// **生成差异**，全部来自 gen_proj_shapes_aot.py 的文件头：
//
//   * OUT ROW STRIDE `OS` = dim = 5120 == n（wo_b 的 verify `proj_mrows` 与 eager `lin`
//     两个站点的 out_stride 都是 dim）。形状门因此要求 `out_stride == OS == n`。
//   * 运行期 m 谓词：同 wq_a（激活 staging + 归约 store）。
//
// =============================================================================
// ABI（与 `dsv41_gemm_fp8_mrows` 同形）
// =============================================================================
//   extern "C" int dsv41_gemm_fp8_tilelang_wo_b(
//       const uint8_t* a, const float* a_scale,      // 激活 fp8 [m, k] + [m, k/32] f32
//       const uint8_t* w, const uint8_t* w_scale,    // 权重 fp8 [n, k] + [n/32, k/32] ue8m0
//       const float* bias,                           // 本阶段必须 null
//       float* out,                                  // f32 [m, n]，行 stride == n
//       int m, int n, int k, int out_stride, cudaStream_t s);
//   -> 0 = 已发射；2 = DECLINED；其它 = cuda 错误码。
//
// rc 语义照 `dsv41_gemm_fp8_mrows` / `wkv_shim.cu`：`2` = declined，**永不返回 1**。

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 生成物：改名后 include（两份 dump 都叫 main_kernel）。
#define main_kernel wo_b_tl_partial_kernel
#include "wo_b_partial_tl.cu"
#undef main_kernel
#define main_kernel wo_b_tl_reduce_kernel
#include "wo_b_reduce_tl.cu"
#undef main_kernel

namespace {

// ---------------------------------------------------------------------------
// 冻结几何 —— 逐项来自 tilelang_gen/proj_shapes_tl_config.txt。
// ---------------------------------------------------------------------------
constexpr int kTLN = 5120;          // 权重/输出的 n（wo_b: dim）
constexpr int kTLK = 1024;          // 归约维度 k（wo_b: o_lora_rank）
constexpr int kTLOS = 5120;         // 输出行 stride（烘进归约内核的 `OS` = dim == n）
constexpr int kTLBN = 128;          // partial 的 N 向 tile
constexpr int kTLKS = 8;            // K-split 分片数
constexpr int kTLThreads = 128;     // partial block
constexpr int kTLSmem = 13824;      // partial 动态 smem（NS*(16*32 + 128*32)）
constexpr int kTLRedBN = 256;       // reduce 的 N 向 tile
constexpr int kTLRedThreads = 256;  // reduce block
constexpr int kTLMPad = 16;         // mma m16 的 M（激活的 pad 目标）
constexpr int kTLMRows = 8;         // 本阶段接受的 m 上界

// 常驻 scratch：P[KS][MPAD][N] f32。INIT 期分配，进程生命周期内复用。
float* g_part = nullptr;

// INIT：只做一次。返回 false ⇒ 本 shim 永不发射。
bool tl_wo_b_init() {
    static int state = 0;
    if (state != 0) return state > 0;
    (void)cudaFuncSetAttribute(wo_b_tl_partial_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, kTLSmem);
    (void)cudaGetLastError();
    const size_t bytes = (size_t)kTLKS * (size_t)kTLMPad * (size_t)kTLN * sizeof(float);
    if (cudaMalloc(&g_part, bytes) != cudaSuccess) {
        g_part = nullptr;
        (void)cudaGetLastError();
        state = -1;
        return false;
    }
    state = 1;
    return true;
}

}  // namespace

extern "C" int dsv41_gemm_fp8_tilelang_wo_b(const uint8_t* a, const float* a_scale,
                                            const uint8_t* w, const uint8_t* w_scale,
                                            const float* bias, float* out, int m, int n, int k,
                                            int out_stride, cudaStream_t s) {
    // ---- 形状门（第二阶段 = 冻结的 wo_b 形状）----
    if (m < 1 || m > kTLMRows) return 2;
    if (n != kTLN || k != kTLK) return 2;
    // 行 stride 只在 m > 1 参与地址计算（第 0 行恒在 offset 0）⇒ m == 1 放宽。
    if (m > 1 && out_stride != kTLOS) return 2;
    if (a == nullptr || a_scale == nullptr || w == nullptr || w_scale == nullptr || out == nullptr)
        return 2;
    if (bias != nullptr) return 2;
    if ((((uintptr_t)a & 0xF) != 0) || (((uintptr_t)a_scale & 0xF) != 0) ||
        (((uintptr_t)w & 0xF) != 0) || (((uintptr_t)w_scale & 0xF) != 0) ||
        (((uintptr_t)out & 0x1F) != 0))
        return 2;

    if (!tl_wo_b_init()) {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang] ARMED but INIT FAILED (scratch alloc) -> this run measures "
                    "the OLD path (m=%d n=%d k=%d)\n",
                    m, n, k);
        return 2;
    }

    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang] ARMED wo_b m=%d n=%d k=%d ks=%d os=%d -> grid=(%d,%d)x%d "
                    "smem=%d + reduce grid=%d x %d\n",
                    m, n, k, kTLKS, kTLOS, kTLN / kTLBN, kTLKS, kTLThreads, kTLSmem,
                    kTLN / kTLRedBN, kTLRedThreads);
    }

    wo_b_tl_partial_kernel<<<dim3((unsigned)(kTLN / kTLBN), (unsigned)kTLKS), kTLThreads,
                             kTLSmem, s>>>(reinterpret_cast<const fp8_e4_t*>(a), a_scale, g_part,
                                           reinterpret_cast<const fp8_e4_t*>(w),
                                           reinterpret_cast<const fp8_e8_t*>(w_scale), m);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    wo_b_tl_reduce_kernel<<<dim3((unsigned)(kTLN / kTLRedBN)), kTLRedThreads, 0, s>>>(out, g_part, m);
    e = cudaGetLastError();
    return (int)e;
}
