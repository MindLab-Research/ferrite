// wkv_shim.cu — TileLang 投影 GEMM 的 launcher shim（第一阶段：wkv 一个形状）。
//
// 这是 docs/agent/tilelang-integration-design.md §3.3「launcher shim 契约」的落地件：
// 生成物（`wkv_partial_tl.cu` / `wkv_reduce_tl.cu`）只含 `__global__` kernel，不含
// host launcher；本文件就是那个 host launcher + 这个 TU 的编译单元。
//
// =============================================================================
// 为什么生成物是 #include 而不是各自一个 TU
// =============================================================================
// 两份 dump 都定义 `extern "C" __global__ void main_kernel(...)`（TileLang 的固定名）。
// 直接当两个 TU 编进同一个 `.so` 会 duplicate symbol。⇒ 用 `#define main_kernel ...`
// 把它们改名后 include 进来。**build.sh 只把本文件当 TU**（生成物是 include 的"头"）。
//
// =============================================================================
// ABI（与 `dsv41_gemm_fp8_mrows` / `dsv41_gemm_fp8_mrows_mma` 同形）
// =============================================================================
//   extern "C" int dsv41_gemm_fp8_tilelang_wkv(
//       const uint8_t* a, const float* a_scale,      // 激活 fp8 [m, k] + [m, k/32] f32
//       const uint8_t* w, const uint8_t* w_scale,    // 权重 fp8 [n, k] + [n/32, k/32] ue8m0
//       const float* bias,                           // 本阶段必须 null（见下方 decline）
//       float* out,                                  // f32 [m, n]（out_stride 必须 == n）
//       int m, int n, int k, int out_stride, cudaStream_t s);
//   -> 0 = 已发射；2 = DECLINED（调用方保持既有路径）；其它 = cuda 错误码（真失败）
//
// 与既有两族**完全相同**的 rc 语义（`dsv41_proj_mma_skel.cu:420` 的注释即先例）：
//   * `2` = declined（形状/模式不接受）——**永不返回 1**，因为 1 是
//     `cudaErrorInvalidValue`，与真实发射失败不可区分；
//   * 其它非 0 = `cudaGetLastError()` 的真实错误码，由 Rust 侧 `kerr` 报出。
// ⇒ Rust 侧的 `gemm_fp8_tilelang_wkv` 只在 `rc == 2` 时回退，其它非 0 一律当错误。
//
// =============================================================================
// 三条硬约束（每条都有树内先例）
// =============================================================================
//  1. cudaFuncSetAttribute / cudaMalloc 只在 INIT 期（capture 内调用会让
//     cudaStreamEndCapture 失败）——先例 dsv41_experts_mxf4.cu:4137「init-time, never
//     capture-time」。本 shim 的 INIT 是懒初始化：第一次调用时做一次。
//     ⚠️ INIT 失败（例如恰好发生在 capture 内）⇒ **decline（返回 2）**，调用方保持
//     老路径，绝不半发射。
//  2. `<<<..., s>>>` 用调用方传入的 CuStream —— 必须进 capture / 与主流同序。
//  3. 形状不合规 `return 2`（不是 1 / 负数）。
//
// =============================================================================
// 形状域（第一阶段 = 冻结的 wkv 形状）
// =============================================================================
// 生成物把 N/K/BN/KS 全部 bake 进了 grid 与索引表达式。所以本 shim 是一个**形状专用**
// 入口：只接受 n==512 && k==5120 && out_stride==n && m∈[1,8]，其余一律 decline，
// 由分派链回退到老 kernel（`gemm_fp8_mrows` / `gemm_fp8_mx`）。
// 第二个形状的扩展路径见 PROVENANCE.md §「第二形状」。
//
// =============================================================================
// 两次 launch + 常驻 scratch，为什么不 pad
// =============================================================================
// 生成物是 **runtime-m 变体**（见 kernels/tilelang/gen_wkv_aot.py 文件头）：
//   * 激活 staging 带 `if i < m` 谓词 ⇒ 只读 `a` 的前 m 行，**不需要 16 行 pad 缓冲**，
//     也不会越界读（行 >= m 写 0）；
//   * 归约的 store 带同样的谓词 ⇒ **直接写 ferrite 的 `out`**（行 stride 必须 == n），
//     **不需要 [16, n] 的输出 staging，也不需要 m 行回拷**。
// ⇒ 每次调用 = 1 次 partial launch + 1 次 reduce launch，scratch 只有常驻的
//   `P[KS][16][N]` f32（= 8*16*512*4 = 256 KiB）。
//
// K-split（KS=8）：wkv 的 n=512 在 148 SM 的机器上只有 N/BN = 4 个块，并行度严重不足
// （原型实测 M=1 达 32µs）。KS=8 把块数抬到 4*8 = 32。partial 写 P，reduce 按 kp 升序
// 求和 ⇒ 确定性（同一程序重复跑逐位相同，门 1 的"重复性"判据）。

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 生成物：改名后 include（两份 dump 都叫 main_kernel）。
#define main_kernel wkv_tl_partial_kernel
#include "wkv_partial_tl.cu"
#undef main_kernel
#define main_kernel wkv_tl_reduce_kernel
#include "wkv_reduce_tl.cu"
#undef main_kernel

namespace {

// ---------------------------------------------------------------------------
// 冻结几何 —— 逐项来自 tilelang_gen/wkv_tl_config.txt（生成器写出，勿手改）。
// ---------------------------------------------------------------------------
constexpr int kTLN = 512;           // 权重/输出的 n（wkv: hd）
constexpr int kTLK = 5120;          // 归约维度 k（wkv: dim）
constexpr int kTLBN = 128;          // partial 的 N 向 tile
constexpr int kTLKS = 8;            // K-split 分片数
constexpr int kTLThreads = 128;     // partial block
constexpr int kTLSmem = 13824;      // partial 动态 smem（NS*(16*32 + 128*32)）
constexpr int kTLRedBN = 256;       // reduce 的 N 向 tile
constexpr int kTLRedThreads = 256;  // reduce block
constexpr int kTLMPad = 16;         // mma m16 的 M（激活的 pad 目标，生成物内部使用）
constexpr int kTLMRows = 8;         // 本阶段接受的 m 上界（与 mrows 族一致）

// 常驻 scratch：P[KS][MPAD][N] f32。**per-thread（per-rank）** 惰性分配。
// TP8 的 ranks-are-threads 模型下，同一 .so 的多个 rank 线程会并发调用本 shim；进程级单例
// 会让他们互相竞写 P（静默数值损坏 = T 臂乱码根因 #4）。thread_local 把 scratch 绑到执行
// 线程：并发调用必在不同线程 ⇒ 必用不同缓冲区 ⇒ 竞写窗口消除（单线程进程行为不变）。
thread_local float* g_part = nullptr;

// INIT：只做一次。返回 false ⇒ 本 shim 永不发射（调用方保持老路径）。
bool tl_wkv_init() {
    // thread_local：每个 rank 线程各自 SetAttribute + cudaMalloc 一次。state 也线程本地 ⇒
    // 下方的永久闩锁（-1）变成「本线程」的失败记忆，其它线程仍可正常初始化。
    static thread_local int state = 0;  // 0 = 未初始化, 1 = 已就绪, -1 = 初始化失败
    if (state != 0) return state > 0;
    // INIT-TIME ONLY（见文件头约束 1）。smem 只有 13.8 KiB（< 48 KiB 默认上限），
    // 这里的 SetAttribute 是契约对齐（多形状/大 smem 时才是必要条件），失败不致命。
    (void)cudaFuncSetAttribute(wkv_tl_partial_kernel,
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
extern "C" int dsv41_gemm_fp8_tilelang_wkv(const uint8_t* a, const float* a_scale,
                                           const uint8_t* w, const uint8_t* w_scale,
                                           const float* bias, float* out, int m, int n, int k,
                                           int out_stride, cudaStream_t s) {
    // ---- 形状门（第一阶段 = 冻结的 wkv 形状）----
    if (m < 1 || m > kTLMRows) return 2;
    if (n != kTLN || k != kTLK) return 2;
    // reduce 的 store 以行 stride == n 写 `out`（生成物 bake 的），所以 out_stride
    // 必须恰为 n；否则会写错位置。不满足就 decline（调用方用自己的 kernel）。
    // [DIAG] one-shot call trace: is this shim even being reached? where does it decline?
    static int diag_calls = 0;
    if (diag_calls < 3) {
        diag_calls++;
        fprintf(stderr,
                "[tl-diag] wkv call#%d: m=%d n=%d k=%d out_stride=%d bias=%p | "
                "a=%p(al:%d) a_scale=%p(al:%d) w=%p(al:%d) w_scale=%p(al:%d) out=%p(al:%d)\n",
                diag_calls, m, n, k, out_stride, (void*)bias,
                (void*)a, (int)((uintptr_t)a & 0xF),
                (void*)a_scale, (int)((uintptr_t)a_scale & 0xF),
                (void*)w, (int)((uintptr_t)w & 0xF),
                (void*)w_scale, (int)((uintptr_t)w_scale & 0xF),
                (void*)out, (int)((uintptr_t)out & 0x1F));
    }
    if (out_stride != n) return 2;
    if (a == nullptr || a_scale == nullptr || w == nullptr || w_scale == nullptr || out == nullptr)
        return 2;
    // 本阶段的生成物没有 bias 通路 ⇒ 带 bias 的调用一律 decline（绝不静默丢 bias）。
    if (bias != nullptr) return 2;
    // 对齐：partial 的 A staging 是 4B 组的读写，W 走 cp.async 16B，reduce 的 store 是
    // 32B（tl::store_global_256）。基址不对齐在这些写法下是 err 716 或静默错位，
    // 一律 decline（照 dsv41_gemm_fp8_mrows_mma 的 16B 检查先例，这里取更严的 16B）。
    if ((((uintptr_t)a & 0xF) != 0) || (((uintptr_t)a_scale & 0xF) != 0) ||
        (((uintptr_t)w & 0xF) != 0) || (((uintptr_t)w_scale & 0xF) != 0) ||
        (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit, v2 state-gated): the ORIGINAL unconditional
    // decline was a DESIGN FLAW — it blocked ALL TileLang launches from ever
    // entering the verify graph (VERIFY_GRAPH=1 is the production config),
    // making the entire TileLang effort pointless (zero speedup on verify).
    // The correct semantics (matching head_bf16_shim's audited pattern):
    //   - capturing AND INIT not done (g_part == nullptr) → decline (can't
    //     cudaMalloc inside capture)
    //   - capturing AND INIT done (g_part != nullptr) → ALLOW the launch into
    //     the graph (launches are capture-safe once scratch is allocated)
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone
        && g_part == nullptr) {
        return 2;  // INIT hasn't run yet — decline without touching capture
    }

    if (!tl_wkv_init()) {
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
                    "[proj-tilelang] ARMED wkv m=%d n=%d k=%d ks=%d -> grid=(%d,%d)x%d "
                    "smem=%d + reduce grid=%d x %d\n",
                    m, n, k, kTLKS, kTLN / kTLBN, kTLKS, kTLThreads, kTLSmem,
                    kTLN / kTLRedBN, kTLRedThreads);
    }

    // 一次性回执（tl-parity-vs-old #4：out_stride 的 row-0 免疫陷阱）。行 stride 只在
    // m > 1 参与地址计算（第 0 行恒在 offset 0）⇒ 调用侧把 out_stride 传成一个「错的但
    // 非零」的值时，m == 1 的逐位测试永远看不见（row 0 正确、row >= 1 整行错位）——
    // 只有多行 kernel 才暴露。本 shim 的形状门只在 m > 1 时强制 out_stride == OS
    // （内核烘死的行 stride），所以这里把**实际收到的** out_stride 与**烘死的** OS 各打
    // 一次；**out 缓冲的真实行距 shim 看不到，调用方必须自行核对它 == baked_os**。
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang:wkv] FIRST CALL out_stride=%d n=%d m=%d baked_os=%d — "
                    "caller MUST verify the out buffer's real row pitch == baked_os\n",
                    (int)out_stride, (int)n, (int)m, kTLN);
    }

    // (1) K-split 分片：grid (N/BN, KS)，每块算一段 K 的 partial 写 P[kp]。
    // 生成物的 fp8 形参是 tl_templates 的 `fp8_e4_t` / `fp8_e8_t`（1 字节位模式类型），
    // ferrite 的 ABI 是裸 `uint8_t*` ⇒ 这里 reinterpret_cast 对齐两种契约。
    wkv_tl_partial_kernel<<<dim3((unsigned)(kTLN / kTLBN), (unsigned)kTLKS), kTLThreads,
                            kTLSmem, s>>>(reinterpret_cast<const fp8_e4_t*>(a), a_scale, g_part,
                                          reinterpret_cast<const fp8_e4_t*>(w),
                                          reinterpret_cast<const fp8_e8_t*>(w_scale), m);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    // (2) 确定性归约：按 kp 升序求和 ks 个 partial，只写前 m 行（行 stride = n = out_stride）。
    wkv_tl_reduce_kernel<<<dim3((unsigned)(kTLN / kTLRedBN)), kTLRedThreads, 0, s>>>(out, g_part, m);
    e = cudaGetLastError();
    return (int)e;
}
