// wo_a_shim.cu — TileLang 分组输出投影 GEMM 的 launcher shim（第二阶段：wo_a 分组）。
//
// 与第一阶段 `wkv_shim.cu` 同构（契约、三条硬约束、rc 语义、两次 launch + 常驻 scratch 的
// 理由都见该文件头与 docs/agent/tilelang-integration-design.md §3.3）。与稠密四形状的差异
// 全部来自 gen_proj_shapes_aot.py 的文件头 + docs/agent/tilelang-proj-phase2.md §6：
//
//   * ABI = `dsv41_wo_a_grouped_fp8` 的 ABI（多 `groups` / `rows` / `a_stride` / `out_stride`）。
//     量化在 **shim 外**（调用点已经给了 fp8 `a` + f32 `a_scale`）—— phase2 §6.1 的结论：
//     shim 侧零新增集成。
//   * GROUP 维进 grid（`blockIdx.y` 语义，原型 GPU 已验证）：`T.Kernel(N/bN, G, ks)`。
//     每 block 单组、零跨组共享，与老 kernel `blockIdx.y = g` 同形。
//   * 激活是 2D `[MPAD, ASTRIDE]`，第 g 组段在 +g*K 字节（block-diagonal）；行距 ASTRIDE ≠ K
//     （world=8 时 nlg=1 退化为 ASTRIDE == K；world=1 时是真正的跨组行距）。
//     ⇒ ASTRIDE 是编译期常量（地址计算的一部分），**G=1(ASTRIDE=4096) 与 G=8(ASTRIDE=32768)
//     各一对 dump**，本文件按 (groups, a_stride) 派发。
//   * OUT ROW STRIDE `OS` = ol_total = groups*o_lora_rank = 8192, NOT G*n。wo_a 是
//     ColumnParallel：本 rank 的 nlg 个组写进全局宽行（列偏移 g*n），行距是 groups*o_lora。
//     生成物把 OS 烘进归约的输出声明，形状门据此要求 `out_stride == OS`。
//   * 运行期 m 谓词：激活 staging + 归约 store 都带 `if i < m`。
//
// =============================================================================
// ABI（与 `dsv41_wo_a_grouped_fp8` 同形）
// =============================================================================
//   extern "C" int dsv41_gemm_fp8_tilelang_wo_a(
//       const uint8_t* a, const float* a_scale,   // 激活 fp8 [rows, a_stride] + f32 [rows, a_stride/32]
//       const uint8_t* w, const uint8_t* w_scale, // 权重 fp8 [G, n, k] + ue8m0 [G, n/32, k/32]
//       const float* bias,                        // wo_a 站点恒 null（:8177），带 bias 一律 decline
//       float* out,                               // f32 [rows, G*n]，行 stride = out_stride
//       int groups, int rows, int n, int k, int a_stride, int out_stride, cudaStream_t s);
//   -> 0 = 已发射；2 = DECLINED（调用方保持 per-(group,row) 回退）；其它 = cuda 错误码。
//
// rc 语义照 `dsv41_wo_a_grouped_fp8` / `wkv_shim.cu`：`2` = declined，**永不返回 1**。

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 生成物：四个 dump 都叫 main_kernel ⇒ 逐个改名后 include。
#define main_kernel wo_a_g1_tl_partial_kernel
#include "wo_a_g1_partial_tl.cu"
#undef main_kernel
#define main_kernel wo_a_g1_tl_reduce_kernel
#include "wo_a_g1_reduce_tl.cu"
#undef main_kernel
#define main_kernel wo_a_g8_tl_partial_kernel
#include "wo_a_g8_partial_tl.cu"
#undef main_kernel
#define main_kernel wo_a_g8_tl_reduce_kernel
#include "wo_a_g8_reduce_tl.cu"
#undef main_kernel

namespace {

// ---------------------------------------------------------------------------
// 冻结几何 —— 逐项来自 tilelang_gen/proj_shapes_tl_config.txt。
// ---------------------------------------------------------------------------
constexpr int kTLN = 1024;          // 每组输出宽（wo_a: o_lora_rank）
constexpr int kTLK = 4096;          // 每组 K（wo_a: hpg*head_dim）
constexpr int kTLOS = 8192;         // 输出行 stride（烘进归约内核的 `OS` = groups*o_lora_rank）
constexpr int kTLBN = 128;          // partial 的 N 向 tile
constexpr int kTLKS = 8;            // K-split 分片数
constexpr int kTLThreads = 128;     // partial block
constexpr int kTLSmem = 13824;      // partial 动态 smem（NS*(16*32 + 128*32)）
constexpr int kTLRedBN = 256;       // reduce 的 N 向 tile
constexpr int kTLRedThreads = 256;  // reduce block
constexpr int kTLMPad = 16;         // mma m16 的 M（激活的 pad 目标）
constexpr int kTLMRows = 8;         // 本阶段接受的 rows 上界

// 两个变体的 (G, ASTRIDE)
constexpr int kTLG1 = 1;
constexpr int kTLAstride1 = 4096;   // nlg=1（verify@TP8）：ASTRIDE == K
constexpr int kTLG8 = 8;
constexpr int kTLAstride8 = 32768;  // nlg=8（TP1）：跨组行距

// 常驻 scratch：P[KS][G][MPAD][N] f32。G=8 是上界，两个变体复用同一块。
// **per-thread（per-rank）** 惰性分配（理由见 wkv_shim.cu）：并发 rank 线程各持一块。
thread_local float* g_part = nullptr;

// INIT：只做一次。返回 false ⇒ 本 shim 永不发射（调用方保持老路径）。
bool tl_wo_a_init() {
    // thread_local：每个 rank 线程各自 SetAttribute + cudaMalloc 一次（理由见 wkv_shim.cu）。
    static thread_local int state = 0;
    if (state != 0) return state > 0;
    (void)cudaFuncSetAttribute(wo_a_g1_tl_partial_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, kTLSmem);
    (void)cudaFuncSetAttribute(wo_a_g8_tl_partial_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, kTLSmem);
    (void)cudaGetLastError();  // 吞掉 SetAttribute 的潜在错误
    const size_t bytes = (size_t)kTLKS * (size_t)kTLG8 * (size_t)kTLMPad * (size_t)kTLN * sizeof(float);
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

// 返回 0（已发射）/ 2（DECLINED：调用方保持 per-(group,row) 回退）/ 其它 cuda 错误码。
extern "C" int dsv41_gemm_fp8_tilelang_wo_a(const uint8_t* a, const float* a_scale,
                                            const uint8_t* w, const uint8_t* w_scale,
                                            const float* bias, float* out, int groups, int rows,
                                            int n, int k, int a_stride, int out_stride,
                                            cudaStream_t s) {
    // ---- 形状门（第二阶段 = 冻结的 wo_a 分组形状；两个 (G,ASTRIDE) 变体）----
    if (rows < 1 || rows > kTLMRows) return 2;
    if (n != kTLN || k != kTLK) return 2;
    // OS = ol_total 是烘进生成物的行 stride（≠ G*n）。只在 rows > 1 参与地址计算
    // （第 0 行恒在 offset 0）⇒ rows == 1 时放宽（eager 单行站点按紧凑行传）。
    if (rows > 1 && out_stride != kTLOS) return 2;
    // 变体选择：(groups, a_stride) 必须恰好命中一对 dump 的编译期常量。
    const bool g1 = (groups == kTLG1 && a_stride == kTLAstride1);
    const bool g8 = (groups == kTLG8 && a_stride == kTLAstride8);
    if (!g1 && !g8) return 2;
    if (a == nullptr || a_scale == nullptr || w == nullptr || w_scale == nullptr || out == nullptr)
        return 2;
    // wo_a 站点 bias 恒为 null；带 bias 一律 decline（绝不静默丢 bias）。
    if (bias != nullptr) return 2;
    if ((((uintptr_t)a & 0xF) != 0) || (((uintptr_t)a_scale & 0xF) != 0) ||
        (((uintptr_t)w & 0xF) != 0) || (((uintptr_t)w_scale & 0xF) != 0) ||
        (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit, v2 state-gated): only decline when INIT hasn't
    // completed (g_part == nullptr) — once scratch is allocated, launches are
    // capture-safe and SHOULD enter the verify graph (the head_bf16 pattern).
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone
        && g_part == nullptr) {
        return 2;  // INIT hasn't run yet — decline without touching capture
    }

    if (!tl_wo_a_init()) {
        // 初始化失败（分配不到 scratch / 恰好发生在 capture 内）：decline，不发射。
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang] ARMED but INIT FAILED (scratch alloc) -> this run measures "
                    "the OLD path (G=%d n=%d k=%d)\n",
                    groups, n, k);
        return 2;
    }

    // 一次性活性回执（照 [mrows-mtile] ARMED 的先例）。
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang] ARMED wo_a G=%d rows=%d n=%d k=%d a_stride=%d os=%d ks=%d "
                    "-> grid=(%d,%d,%d)x%d + reduce grid=(%d,%d)x%d\n",
                    groups, rows, n, k, a_stride, kTLOS, kTLKS, kTLN / kTLBN, groups, kTLKS,
                    kTLThreads, kTLN / kTLRedBN, groups, kTLRedThreads);
    }

    // 一次性回执（tl-parity-vs-old #4：out_stride 的 row-0 免疫陷阱）。本形状的 OS =
    // groups*o_lora_rank = 8192 ≠ groups*n（ColumnParallel：本 rank 的每组写进全局宽行的
    // +g*n 列，行距是整行）。行 stride 只在 rows > 1 参与地址计算（第 0 行恒在 offset 0）
    // ⇒ 调用侧把 out_stride 传成一个「错的但非零」的值时，rows == 1 的逐位测试永远看不见
    // （row 0 正确、row >= 1 整行错位）—— 只有多行 kernel 才暴露。形状门只在 rows > 1 时
    // 强制 out_stride == OS，所以这里把**实际收到的** out_stride 与**烘死的** OS 各打一次；
    // **out 缓冲的真实行距 shim 看不到，调用方必须自行核对它 == baked_os（verify 的 wo_r
    // 行距就是 ol_total）**。
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang:wo_a] FIRST CALL out_stride=%d n=%d rows=%d G=%d a_stride=%d "
                    "baked_os=%d — caller MUST verify the out buffer's real row pitch == baked_os\n",
                    (int)out_stride, (int)n, (int)rows, (int)groups, (int)a_stride, kTLOS);
    }

    cudaError_t e;
    if (g1) {
        // (1) 分组 K-split 分片：grid (N/BN, G=1, KS)。
        wo_a_g1_tl_partial_kernel<<<dim3((unsigned)(kTLN / kTLBN), (unsigned)kTLG1, (unsigned)kTLKS),
                                    kTLThreads, kTLSmem, s>>>(
            reinterpret_cast<const fp8_e4_t*>(a), a_scale, g_part,
            reinterpret_cast<const fp8_e4_t*>(w), reinterpret_cast<const fp8_e8_t*>(w_scale),
            rows);
        e = cudaGetLastError();
        if (e != cudaSuccess) return (int)e;
        // (2) 分组归约：组 g 的 partial 归到 out[:, g*N + ...]，行 stride = OS。
        wo_a_g1_tl_reduce_kernel<<<dim3((unsigned)(kTLN / kTLRedBN), (unsigned)kTLG1),
                                   kTLRedThreads, 0, s>>>(out, g_part, rows);
        e = cudaGetLastError();
        return (int)e;
    }
    // G=8 变体
    wo_a_g8_tl_partial_kernel<<<dim3((unsigned)(kTLN / kTLBN), (unsigned)kTLG8, (unsigned)kTLKS),
                                kTLThreads, kTLSmem, s>>>(
        reinterpret_cast<const fp8_e4_t*>(a), a_scale, g_part,
        reinterpret_cast<const fp8_e4_t*>(w), reinterpret_cast<const fp8_e8_t*>(w_scale), rows);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    wo_a_g8_tl_reduce_kernel<<<dim3((unsigned)(kTLN / kTLRedBN), (unsigned)kTLG8),
                               kTLRedThreads, 0, s>>>(out, g_part, rows);
    e = cudaGetLastError();
    return (int)e;
}
