// head_bf16_shim.cu — TileLang head bf16 GEMM 的 launcher shim（ferrite 生产链接线件）。
//
// 这是 docs/agent/tilelang-attn-head.md（原型结论）+ kernels/cuda/tilelang_gen/
// wkv_shim.cu（投影族第一阶段接线模式）的落地件：生成物 `head_partial_tl.cu` /
// `head_reduce_tl.cu` 只含 `__global__` kernel（TileLang 固定名 `main_kernel`），
// 不含 host launcher；本文件就是那个 host launcher + 这个 TU 的编译单元
// （build.sh 的 `tilelang_gen/*_shim.cu` glob 自动纳入；生成物仍不作为独立 TU ——
//  两份都定义 `main_kernel`，直接编译会 duplicate symbol）。
//
// =============================================================================
// 它替代的是什么（为什么值得接）
// =============================================================================
// ferrite 的 head 是 `dsv41_head_gemv_bf16_mrows` 家族的 **v1 序** SIMT GEMV
// （`dsv41_gemv_bf16_v1_mrows`，dsv41_glue.cu:1624）：w 是 bf16、x 是 f32，逐元素
// f32 FMA。原型把 M 放进 mma 的 tile 维度（m16 一发覆盖 6 行）+ K-split 把块数填满
// 148 SM，实测 M6/M1 = **1.00**（老 v1_mrows 是 **3.23–3.81**），M=6 绝对值
// **2.93×**（slice 125.2→42.2µs；docs/agent/tilelang-attn-head.md §3）。
//
// ⚠️ 数值债（原型 §4，本 shim 不能替它还）：tensor-core mma 没有 bf16×f32 形态，
// A 必须降到 bf16 ⇒ 本 shim 在 host 侧把 f32 激活 cast 到 bf16。head 输出喂 argmax，
// 该 cast 的 max_rel ≈ 2.3e-2 远大于 `draft-head-fold-v2-argmax-verdict.md` 记的
// ~1e-3 翻面阈值 ⇒ **本臂默认 OFF，且它是 A/B 实验臂不是生产替换**（§6.3）。数值债
// 必须由 verify 的 acc 门（mean-k / Z_）来还，micro bench 证明不了。
//
// =============================================================================
// 导出符号（一个）
// =============================================================================
//   int dsv41_head_bf16_tilelang(const void* w, const float* x, float* out,
//                                int m, int n, int k, cudaStream_t s);
//     w   bf16 [n, k]（head.weight 的切片）
//     x   f32  [m, k]（verify 的 xn_r；**仅前 m 行有效**）
//     out f32  [m, n]（logits_r；行 stride 必须恰为 n —— 见下）
//
// ABI 是 `dsv41_gemv_bf16_v1_mrows` 的 **verbatim**（device.rs 的
// `head_gemv_bf16_v1_mrows` 同形），所以 Rust 侧就是它的 drop-in 臂。
//
// rc 契约与 wkv / moe shim / dsv41_proj_mma_skel.cu 完全一致：
//   * `0`  = 已发射；
//   * `2`  = DECLINED（形状/模式/capture/INIT 不接受）——**永不返回 1**，因为 1 是
//            `cudaErrorInvalidValue`，与真实发射失败不可区分；
//   * 其它非 0 = `cudaGetLastError()` 的真实错误码，由 Rust 侧 `kerr` 报出。
// ⇒ Rust 侧只在 `rc == 2` 时回退老路径，其它非 0 一律当错误。
//
// =============================================================================
// 三条硬约束 + 一条审计红线
// =============================================================================
//  1. `cudaFuncSetAttribute` / `cudaMalloc` 只在 INIT 期（capture 内调用会让
//     cudaStreamEndCapture 失败）——先例 dsv41_experts_mxf4.cu:4137「init-time,
//     never capture-time」。本 shim 的 INIT 是懒初始化：第一次调用时做一次。
//     ⚠️ SetAttribute **是必要条件**：partial 的动态 smem 55296 B > 48 KiB 默认上限，
//     不设就 launch 失败（err 1）。
//  2. **capture guard（verify-graph-tl-audit 的 P0-2，审计红线，必须包含）**：
//     `cudaStreamIsCapturing` 检查放在**形状门之后、INIT 之前**。审计的判决是：
//     C 侧「防御性 decline」若在 INIT 之后才检查，cudaMalloc 会**先**触发并把
//     capture 打废（wkv_shim.cu:107 的先例），链条以 state=-1 永久 disarm 收场 ——
//     即所谓 "capture-safe holds by LUCK not mechanism"。本 shim 的 guard 在
//     shape gate 之后立即检查：**capturing 且 INIT 尚未完成 ⇒ decline（绝不
//     cudaMalloc 在 capture 内）**；INIT 已完成（state==1）则三个 launch 本身是
//     capture-safe 的，**允许**进 capture（这才是 graph 能录到本臂的前提）。
//  3. 形状不合规 `return 2`（不是 1 / 负数）。
//  4. **P1（同一审计）：INIT 失败不永久闩死。** 失败分两类：确定性的
//     （`cudaFuncSetAttribute` 的 invalid value）闩到 state=-1；**瞬态的**
//     （`cudaMalloc` 的 `cudaErrorMemoryAllocation`）**留在 state=0 以便下次重试**
//     （accumulator 不是 one-shot）。
//
// =============================================================================
// 形状域（冻结 —— 生产 draft 切片）
// =============================================================================
// 生成物的 grid/索引表达式把 N/K/BN/KS 全部 bake 了。本 shim 是**形状专用**入口：
// 只接受 n==16160 && k==5120 && m∈[1,8]，其余一律 decline，由分派链回退到老
// `gemv_bf16` / `gemv_bf16_v1_mrows`。
//
//   ⚠️ 16160 = vocab(129280) / world(8)，即**生产 draft 切片**的 per-rank 行数
//   （原型 §0 已判任务书的 "256" 不是 head 的 K；K = dim = 5120）。world ≠ 8 时
//   seg ≠ 16160 ⇒ decline ⇒ 回退老路径。verify 的**未切片** head（n=129280）同理
//   decline（它是一个不同的冻结形状，见 PROVENANCE.md §5 的扩展路径）。
//
// =============================================================================
// 两次 TileLang launch + 一次 cast（f32 激活 → bf16 staging）
// =============================================================================
// 生成物是 **runtime-m 变体**（见 kernels/tilelang/gen_head_aot.py 文件头）：
//   * partial 的 X 是常驻 bf16 staging [MPAD=16, K]（由本文件的 cast kernel 填，
//     行 >= m 写 0）⇒ 不需要在 host 侧为每次调用 pad，也不会越界读；
//   * 归约的 store 带 `if i < m` + `if col < n` 谓词 ⇒ **直接写 ferrite 的 `out`**
//     （行 stride 必须 == n），不需要 [16, n] 输出 staging，也不需要 m 行回拷。
// ⇒ 每次调用 = 1 次 cast + 1 次 partial + 1 次 reduce，scratch 是常驻的
//   `P[KS][16][NPAD]` f32（8*16*16256*4 = 8.30 MiB）+ `X[16][K]` bf16（160 KiB）。
//
// K-split（KS=8）与投影族同源：把块数从 N/BN=127 抬到 127*8=1016（148 SM 的 6.9 倍），
// 且 partial 写 P、reduce 按 kp 升序求和 ⇒ 确定性（同一程序重复跑逐位相同）。
//
// =============================================================================
// ⚠️ 生成物尚未产出时的可编译性（__has_include 守卫）
// =============================================================================
// 本文件的 `head_partial_tl.cu` / `head_reduce_tl.cu` 由主 agent 在远端用
// `kernels/tilelang/gen_head_aot.py` AOT 生成（见交付文档的 GPU 手册）。在那之前它们
// **不存在**，若直接 `#include` 会让 `build.sh` 硬失败、连累同仓的其它 agent。⇒ 两个
// include 与**整个导出符号**都在 `__has_include` 守卫内：生成物缺席时本 TU 编译出一个
// 空单元（符号缺席 ⇒ Rust 侧 `supports_head_tilelang()` false ⇒ armed 会报
// "no symbol, rebuild"，走既有的 stale-.so 纪律，绝不静默测老路）。
// 生成物就位后重建 `.so`，符号自动出现。
#if defined(__has_include)
#if __has_include("head_partial_tl.cu") && __has_include("head_reduce_tl.cu")
#define FERRITE_HEAD_TL_GENERATED 1
#endif
#endif

#ifdef FERRITE_HEAD_TL_GENERATED

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 生成物：改名后 include（两份 dump 都叫 main_kernel）。
#define main_kernel head_tl_partial_kernel
#include "head_partial_tl.cu"
#undef main_kernel
#define main_kernel head_tl_reduce_kernel
#include "head_reduce_tl.cu"
#undef main_kernel

namespace {

// ---------------------------------------------------------------------------
// 冻结几何 —— 逐项来自 tilelang_gen/head_tl_config.txt（生成器写出，勿手改）。
// ---------------------------------------------------------------------------
constexpr int kTLN = 16160;      // 权重/输出的 n（head.weight 的生产切片行数）
constexpr int kTLK = 5120;       // 归约维度 k（模型 dim）
constexpr int kTLBN = 128;       // partial 的 N 向 tile
constexpr int kTLKS = 8;         // K-split 分片数
constexpr int kTLBK = 64;        // partial 的 K 向 tile
constexpr int kTLNS = 3;         // pipelined stages
constexpr int kTLThreads = 128;  // partial block
constexpr int kTLRedBN = 256;    // reduce 的 N 向 tile
constexpr int kTLRedThreads = 256;  // reduce block
constexpr int kTLMPad = 16;      // mma m16 的 M（激活的 pad 目标，生成物内部使用）
constexpr int kTLM = 8;          // 本阶段接受的 m 上界（与 v1_mrows 族一致）
// P 的 N 向容量：N 不整除 BN 时把最后一个 tile 补齐，使 partial 的 store 永远在界内
// （否则末块的 96 列会越过 P 的 N 宽 —— 依赖 TileLang 的 store 谓词，不如直接 pad）。
constexpr int kTLNPad = ((kTLN + kTLBN - 1) / kTLBN) * kTLBN;  // 16256
// partial 动态 smem：NS*(A_sh + W_sh) = 3*(16*64 + 128*64)*2 B = 55296（> 48 KiB，
// 所以 INIT 的 SetAttribute 是必要条件）。
constexpr size_t kTLSmem = (size_t)kTLNS * (size_t)(kTLMPad + kTLBN) * (size_t)kTLBK *
                           sizeof(__nv_bfloat16);

// ---- 常驻 scratch（INIT 期分配一次，进程生命周期内复用）--------------------
// P : [KS, MPAD, NPAD] f32 —— K-split 的 partial（8.30 MiB）
// Xb: [MPAD, K]        bf16 —— cast 后的激活 staging（160 KiB；行 >= m 写 0）
float* g_part = nullptr;
__nv_bfloat16* g_xb = nullptr;

// INIT 状态机（P1：瞬态不永久闩死）。
//   0 = 未初始化（含「上次瞬态失败，可重试」）
//   1 = 已就绪
//  -1 = **确定性**失败（闩死：再试也不会变）
int g_state = 0;

// capture guard 的探测点（P0-2）：INIT 是否已经跑完（只有 state==1 才允许进 capture）。
bool tl_head_ready() { return g_state == 1; }

// INIT：只做一次（成功时）。返回 false ⇒ 本次不发射（调用方保持老路径）。
bool tl_head_init() {
    if (g_state == 1) return true;
    if (g_state == -1) return false;
    // INIT-TIME ONLY（见文件头约束 1）。partial 的动态 smem 55296 B > 48 KiB 默认
    // 上限，这里**必须**成功，否则 launch 直接 err 1。
    cudaError_t es = cudaFuncSetAttribute(head_tl_partial_kernel,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int)kTLSmem);
    (void)cudaGetLastError();  // 吞掉 SetAttribute 的潜在错误，别污染后面的 launch 检查
    if (es != cudaSuccess) {
        // 确定性失败（例如属性名非法）：再试也不会变 ⇒ 闩死。
        g_state = -1;
        return false;
    }
    const size_t p_bytes = (size_t)kTLKS * (size_t)kTLMPad * (size_t)kTLNPad * sizeof(float);
    const size_t x_bytes = (size_t)kTLMPad * (size_t)kTLK * sizeof(__nv_bfloat16);
    cudaError_t ex = cudaMalloc(&g_xb, x_bytes);
    cudaError_t ep = cudaMalloc(&g_part, p_bytes);
    if (ex != cudaSuccess || ep != cudaSuccess) {
        // 清理半成功的分配，别泄漏（下次重试会重新分配）。
        if (ex == cudaSuccess) {
            (void)cudaFree(g_xb);
            g_xb = nullptr;
        }
        if (ep == cudaSuccess) {
            (void)cudaFree(g_part);
            g_part = nullptr;
        }
        (void)cudaGetLastError();
        // P1：`cudaErrorMemoryAllocation` 是**瞬态**（此刻显存紧张）⇒ 留在 state=0
        // 让后续调用重试；其它错误是确定性 ⇒ 闩死。绝不用一次 OOM 永久 disarm。
        const bool transient = (ex == cudaErrorMemoryAllocation) || (ep == cudaErrorMemoryAllocation);
        g_state = transient ? 0 : -1;
        return false;
    }
    g_state = 1;
    return true;
}

// 一次性「INIT 失败」提示：避免「armed 但每次静默测老路」（本项目 #1 测量偏置陷阱）。
void init_failed_note(int m, int n, int k) {
    static int reported = 0;
    if (reported++ == 0)
        fprintf(stderr,
                "[head-tilelang] ARMED but INIT FAILED (SetAttribute/scratch alloc) -> this run "
                "measures the OLD path (m=%d n=%d k=%d)\n",
                m, n, k);
}

// ---------------------------------------------------------------------------
// cast：f32 激活 [m, k] -> bf16 staging [MPAD, K]，行 >= m 写 0（pad）。
// 生成物的 partial 对 X 的读没有 m 谓词，所以 pad 行**必须**显式写 0（干净、
// 不依赖上一次调用的残留；虽然各行独立、脏 pad 行不影响被读的前 m 行，但显式清零
// 让「同一输入的重复跑逐位相同」不依赖历史）。
// ---------------------------------------------------------------------------
__global__ void tl_head_cast_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ xb,
                                    int m, int k) {
    const int r = blockIdx.x;
    __nv_bfloat16* dst = xb + (size_t)r * k;
    if (r >= m) {
        for (int c = threadIdx.x; c < k; c += blockDim.x) dst[c] = __float2bfloat16(0.f);
        return;
    }
    const float* src = x + (size_t)r * k;
    for (int c = threadIdx.x; c < k; c += blockDim.x) dst[c] = __float2bfloat16(src[c]);
}

}  // namespace

// 返回 0（已发射）/ 2（DECLINED：调用方保持老路径）/ 其它 cuda 错误码。
extern "C" int dsv41_head_bf16_tilelang(const void* w, const float* x, float* out, int m, int n,
                                        int k, cudaStream_t s) {
    // ---- 形状门（本阶段 = 冻结的生产切片形状）----
    if (w == nullptr || x == nullptr || out == nullptr) return 2;
    if (m < 1 || m > kTLM) return 2;
    if (n != kTLN || k != kTLK) return 2;
    // 对齐：W 走 cp.async（16B），cast 的 x 是 4B 组读，reduce 的 store 可能向量化
    // 到 16B。基址不满足一律 decline（照 wkv_shim.cu / dsv41_gemm_fp8_mrows_mma 的先例）。
    if ((((uintptr_t)w & 0xF) != 0) || (((uintptr_t)x & 0xF) != 0) || (((uintptr_t)out & 0xF) != 0))
        return 2;

    // ---- capture guard（verify-graph-tl-audit P0-2 —— 审计红线）----
    // cudaMalloc / cudaFuncSetAttribute 在 capture 内非法：malloc 会让
    // cudaStreamEndCapture 失败、graph 永不 engage（wkv_shim.cu:107 的教训）。
    // 形状门已过、INIT 尚未完成 ⇒ DECLINE（调用方保持老路径）；INIT 完成后
    // （state==1）下面的三个 launch 本身 capture-safe，**允许**进 capture。
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone &&
        !tl_head_ready())
        return 2;

    if (!tl_head_init()) {
        init_failed_note(m, n, k);
        return 2;
    }

    // 一次性活性回执（照 [mrows-mtile] / [proj-tilelang] ARMED 的先例：kernel 名只在
    // 日志里出现，这是"ON 的臂真的跑了新程序"的唯一证据）。
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[head-tilelang] ARMED m=%d n=%d k=%d ks=%d -> cast + partial grid=(%d,%d)x%d "
                    "smem=%zu + reduce grid=%d x %d\n",
                    m, n, k, kTLKS, kTLNPad / kTLBN, kTLKS, kTLThreads, (size_t)kTLSmem,
                    kTLNPad / kTLRedBN, kTLRedThreads);
    }

    // (0) cast：f32 激活 -> bf16 [MPAD, K]（行 >= m 写 0）。
    tl_head_cast_kernel<<<dim3((unsigned)kTLMPad), 256, 0, s>>>(x, g_xb, m, k);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (1) K-split 分片：grid (NPAD/BN, KS)，每块算一段 K 的 partial 写 P[kp]。
    // 生成物的形参序 = dump 的 (A, P, W)（TileLang 重排，见 head_tl_config.txt 的签名行）。
    head_tl_partial_kernel<<<dim3((unsigned)(kTLNPad / kTLBN), (unsigned)kTLKS), kTLThreads, kTLSmem,
                             s>>>(reinterpret_cast<const bfloat16_t*>(g_xb), g_part,
                                  reinterpret_cast<const bfloat16_t*>(w));
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (2) 确定性归约：按 kp 升序求和 ks 个 partial，只写前 m 行（行 stride = n = out 的
    // 行 stride —— verify 的两个臂都是 out_stride == n，见 chain_dev.rs 的 geom）。
    head_tl_reduce_kernel<<<dim3((unsigned)(kTLNPad / kTLRedBN)), kTLRedThreads, 0, s>>>(g_part, out,
                                                                                          m);
    return (int)cudaGetLastError();
}

#endif  // FERRITE_HEAD_TL_GENERATED
