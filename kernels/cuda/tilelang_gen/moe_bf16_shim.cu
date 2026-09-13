// moe_bf16_shim.cu — TileLang MoE grouped-GEMM 的 launcher shim（ferrite 生产链接线件）。
//
// 这是 docs/agent/tilelang-moe-grouped.md（原型结论）+ kernels/cuda/tilelang_gen/
// wkv_shim.cu（第一阶段接线模式）的落地件：生成物 `moe_up_tl.cu` / `moe_dn_tl.cu`
// 只含 `__global__` kernel（TileLang 固定名 `main_kernel`），不含 host launcher；
// 本文件就是那个 host launcher + 这个 TU 的编译单元（build.sh 的
// `tilelang_gen/*_shim.cu` glob 自动纳入；生成物仍不作为独立 TU —— 两份都定义
// `main_kernel`，直接编译会 duplicate symbol）。
//
// =============================================================================
// 它替代的是什么（为什么值得接）
// =============================================================================
// ferrite verify MoE 的老路径是 SIMT FMA-issue bound（36 sweep = 物理下限，
// 250µs/层）。本 shim 把「每段 expert 权重只读一次 + MMA 替掉 FMA 指令流」搬进
// 生产链：原型在 B300 上实测 bf16 臂 **70.0µs/层**（up 45.5 + dn 24.5）=
// SIMT 的 **28.0%**（判据 <40% 通过）。bf16 臂成立的前置是**专家权重在加载期
// 一次性 dequant 成 bf16 常驻**（见 DSV41_MOE_BF16_DEQUANT / `dsv41_moe_fp4_to_bf16`），
// 代价是显存 ×4（详细预算见 PROVENANCE.md §8，**需拥有者拍板**）；
// fp4 原地 dequant 已判死（去掉 ALU 仍 102.8µs > bf16 45.5µs），唯一替代是
// tcgen05 blockscaled 原生 fp4 路线。
//
// =============================================================================
// 三个导出符号
// =============================================================================
//   int dsv41_moe_tilelang_gate_up_bf16(...)   -- up（gate‖up）grouped GEMM + scatter
//   int dsv41_moe_tilelang_down_bf16(...)      -- down grouped GEMM + scatter
//   int dsv41_moe_fp4_to_bf16(...)             -- 加载期 fp4(e2m1+ue8m0) -> bf16 副本
//
// rc 契约与 wkv shim / dsv41_proj_mma_skel.cu 完全一致：
//   * `0`  = 已发射；
//   * `2`  = DECLINED（形状/模式不接受）——**永不返回 1**，因为 1 是
//            `cudaErrorInvalidValue`，与真实发射失败不可区分；
//   * 其它非 0 = `cudaGetLastError()` 的真实错误码，由 Rust 侧 `kerr` 报出。
// ⇒ Rust 侧只在 `rc == 2` 时回退老路径，其它非 0 一律当错误。
//
// =============================================================================
// 三条硬约束（照 wkv，每条都有树内先例）
// =============================================================================
//  1. `cudaFuncSetAttribute` / `cudaMalloc` 只在 INIT 期（capture 内调用会让
//     cudaStreamEndCapture 失败）。先例 dsv41_experts_mxf4.cu:4137「init-time,
//     never capture-time」。本 shim 的 INIT 是懒初始化：第一次调用时做一次。
//     ⚠️ **本 shim 的 SetAttribute 是必要条件，不是契约对齐**：up 动态 smem
//     104448 B、dn 135168 B，都远超 48 KiB 默认上限，不设就 launch 失败（err 1）。
//     INIT 失败（例如恰好发生在 capture 内）⇒ **decline（返回 2）**，绝不半发射。
//  2. `<<<..., s>>>` 用调用方传入的 CuStream —— 必须进 capture / 与主流同序。
//  3. 形状不合规 `return 2`（不是 1 / 负数）。
//
// =============================================================================
// 布局契约（生成物 bake 死的东西，host 必须对齐）
// =============================================================================
//   A  : [SEG_CAP*16, K] bf16   -- **已 gather + 每段 pad 到 BM=16** 的激活
//   W  : [E=384, N, K]   bf16   -- 权重，K 连续；up 的 N 前半 = gate(w1)，后半 = up(w3)
//   Eid: [SEG_CAP] int32        -- 每段的 expert id（查 W 的第三坐标）
//   C  : [SEG_CAP*16, N] f32    -- 段内顺序 = expert-grouped 顺序
// 内核**不做 gather / mask / atomic**（原型 §2.2）：gather/scatter 在本文件里完成。
//
// moe_align（host 侧，chain_dev.rs）产出被本 shim 消费的三张表（都是 HOST 数组，
// 由本文件上行到常驻 device scratch）：
//   order[SEG_CAP*16] : int32  -- 该 (段, 段内行) 对应的 assignment 下标（flat，
//                                 = row*topk + slot）；pad 段填 -1
//   counts[SEG_CAP]   : int32  -- 该段的 live 行数（1..2；pad 段 0）
//   eid[SEG_CAP]      : int32  -- 该段的 expert id（pad 段任意，取 0）
// 纯函数、无 atomic、不依赖 block 调度 ⇒ 同一路由表重复计算逐位相同。
//
// =============================================================================
// ⚠️ 这是 EAGER 臂（不是 capture 臂）—— 必须知道
// =============================================================================
// 前置的 moe_align 要在 HOST 上读路由表（`route_idx_r` 在 device 上，由 route_topk
// 写出），所以这个臂**天然需要一个 D2H 回读 + 三个小 H2D 上行**，在 CUDA-graph
// capture 内非法。原型 §8-3 已把「把 nseg 摊到 GPU 侧」列为后续项；在那之前：
//   * 调用方（chain_dev.rs）在 `dev.capturing()` 时**不派遣**本臂（decline 回老路径）；
//   * shim 自身也在 capture 内 decline（防御性——入口处的 P0-2 capture guard 在
//     tl_moe_init() 之前用 cudaStreamIsCapturing 拒绝，绝不把 cudaMalloc 带进 capture；
//     见两个入口 `dsv41_moe_tilelang_{gate_up,down}_bf16`）。
//
// =============================================================================
// 形状域（冻结）
// =============================================================================
// 生成物的 grid/索引表达式把 dim/inter/E/NSEG 全部 bake 了。本 shim 是**形状专用**
// 入口：只接受 dim==5120 && inter==320 && topk∈[1,6] && rows∈[1,6] && nseg∈[1,36]，
// 其余一律 decline，由调用方回退到 `expert_gate_up_fp4_batched` / down 那对。

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 生成物：改名后 include（两份 dump 都叫 main_kernel）。
#define main_kernel moe_up_tl_kernel
#include "moe_up_tl.cu"
#undef main_kernel
#define main_kernel moe_dn_tl_kernel
#include "moe_dn_tl.cu"
#undef main_kernel

namespace {

// ---------------------------------------------------------------------------
// 冻结几何 —— 逐项来自 tilelang_gen/moe_tl_config.txt（生成器写出，勿手改）。
// ---------------------------------------------------------------------------
constexpr int kTLSegCap = 36;    // SEG_CAP = VERIFY_ROWS(6) * TOPK_MAX(6)
constexpr int kTLBM = 16;        // MMA 的最小 M：每段 pad 到 16 行
constexpr int kTLDim = 5120;     // 模型维度 = up 的 K / dn 的 N
constexpr int kTLInter = 320;    // inter_local = up 的 N/2 / dn 的 K
constexpr int kTLE = 384;        // n_routed（生成物的 `e < 384` 守卫是 bake 的）
constexpr int kTLTopkMax = 6;    // topk 上界
constexpr int kTLRowsMax = 6;    // m 上界（VERIFY_ROWS）
constexpr int kTLUpN = 2 * kTLInter;  // 640 = gate ‖ up

// up: grid (3, 36) x 256 threads, smem 104448
constexpr int kUpGridX = 3;
constexpr int kUpThreads = 256;
constexpr size_t kUpSmem = 104448;
// dn: grid (10, 36) x 256 threads, smem 135168
constexpr int kDnGridX = 10;
constexpr int kDnThreads = 256;
constexpr size_t kDnSmem = 135168;

// gather/scatter 的 block（纯搬 + 一次 bf16 舍入，与 GEMM 无关）。
constexpr int kMovThreads = 256;

// ---- 常驻 scratch（INIT 期分配一次，进程生命周期内复用）--------------------
// A: [SEG_CAP*BM, K] bf16（up 的 K=dim，dn 的 K=inter）
// C: [SEG_CAP*BM, N] f32（up 的 N=2*inter，dn 的 N=dim）
// **全部 per-thread（per-rank）**：TP8 的 ranks-are-threads 模型下，同一 .so 的多个 rank
// 线程会并发调用本 shim；进程级单例 scratch 会让他们互相竞写（gather/GEMM/scatter 全链
// 静默数值损坏 = T 臂乱码根因 #4）。thread_local 把每个缓冲区绑到执行线程：并发调用必在
// 不同线程 ⇒ 必用不同缓冲区 ⇒ 竞写窗口消除（单线程进程行为不变）。
thread_local __nv_bfloat16* g_a_up = nullptr;  // 36*16*5120 bf16 = 5.90 MiB
thread_local __nv_bfloat16* g_a_dn = nullptr;  // 36*16*320  bf16 = 0.35 MiB
thread_local float* g_c_up = nullptr;          // 36*16*640  f32  = 1.47 MiB
thread_local float* g_c_dn = nullptr;          // 36*16*5120 f32  = 11.80 MiB
// 元数据（每调用上行一次；见「EAGER 臂」）
thread_local int* g_eid = nullptr;             // [SEG_CAP]
thread_local int* g_order = nullptr;           // [SEG_CAP*BM]
thread_local int* g_counts = nullptr;          // [SEG_CAP]
// `nseg` 的 device 副本：host-table 入口每调用上行一次（4 字节），
// device-table 入口不用它（直接传调用方的 `tl_nseg`）。mover kernel 的 `nseg`
// 形参因此统一成指针 —— 两条臂共用同一 launch 序列。
thread_local int* g_nseg = nullptr;            // [1]

bool tl_moe_init() {
    // thread_local：每个 rank 线程各自 SetAttribute + 7 次 cudaMalloc 一次。state 线程本地 ⇒
    // 下方两处永久闩锁（-1）都变成「本线程」的失败记忆，其它 rank 线程仍可正常初始化。
    static thread_local int state = 0;  // 0 = 未初始化, 1 = 已就绪, -1 = 初始化失败
    if (state != 0) return state > 0;
    // INIT-TIME ONLY（见文件头约束 1）。两个生成物的动态 smem 都 > 48 KiB 默认上限，
    // 这里**必须**成功，否则 launch 直接 err 1。
    bool ok = cudaFuncSetAttribute(moe_up_tl_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)kUpSmem) == cudaSuccess;
    ok = cudaFuncSetAttribute(moe_dn_tl_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)kDnSmem) == cudaSuccess &&
         ok;
    (void)cudaGetLastError();  // 吞掉 SetAttribute 的潜在错误，别污染后面的 launch 检查
    if (!ok) {
        // P1 (no-latch-death): -1 is a PERMANENT latch, kept deliberately. Every
        // failure reachable here is deterministic (cudaFuncSetAttribute rejection is
        // repeatable), and the transient "called inside a capture" case is
        // intercepted by the P0-2 guard at the entry point — INIT is never entered
        // while capturing. A retry could not rescue a latched failure, so
        // re-attempting each call would only re-pay a guaranteed-to-fail init.
        state = -1;
        return false;
    }
    const size_t a_up = (size_t)kTLSegCap * kTLBM * kTLDim * sizeof(__nv_bfloat16);
    const size_t a_dn = (size_t)kTLSegCap * kTLBM * kTLInter * sizeof(__nv_bfloat16);
    const size_t c_up = (size_t)kTLSegCap * kTLBM * kTLUpN * sizeof(float);
    const size_t c_dn = (size_t)kTLSegCap * kTLBM * kTLDim * sizeof(float);
    bool alloc_ok = cudaMalloc(&g_a_up, a_up) == cudaSuccess &&
                    cudaMalloc(&g_a_dn, a_dn) == cudaSuccess &&
                    cudaMalloc(&g_c_up, c_up) == cudaSuccess &&
                    cudaMalloc(&g_c_dn, c_dn) == cudaSuccess &&
                    cudaMalloc(&g_eid, kTLSegCap * sizeof(int)) == cudaSuccess &&
                    cudaMalloc(&g_order, kTLSegCap * kTLBM * sizeof(int)) == cudaSuccess &&
                    cudaMalloc(&g_counts, kTLSegCap * sizeof(int)) == cudaSuccess &&
                    cudaMalloc(&g_nseg, sizeof(int)) == cudaSuccess;
    if (!alloc_ok) {
        (void)cudaGetLastError();
        // P1 (no-latch-death): same permanent-latch rationale as the SetAttribute
        // branch above — cudaMalloc OOM is deterministic, and capture-time entry is
        // blocked by the P0-2 guard, so there is no transient case to retry.
        state = -1;
        return false;
    }
    state = 1;
    return true;
}

// 一次性「INIT 失败」提示：避免「armed 但每次静默测老路」（本项目 #1 测量偏置陷阱）。
void init_failed_note(int rows, int dim, int inter) {
    static int reported = 0;
    if (reported++ == 0)
        fprintf(stderr,
                "[moe-tilelang] ARMED but INIT FAILED (SetAttribute/scratch alloc) -> this run "
                "measures the OLD path (rows=%d dim=%d inter=%d)\n",
                rows, dim, inter);
}

// ---------------------------------------------------------------------------
// gather：f32 激活 -> bf16 的 [SEG_CAP*BM, K]，按 moe_align 的 order/ 分段 pad。
//   A[seg*BM + r][k] = (r < counts[seg]) ? bf16(act[src_row*rpitch + k]) : 0
// up 的 rpitch = dim，src_row = order[...] / topk（激活按 token 行索引）；
// dn 的 rpitch = inter，src_row = order[...]（激活已按 (row,slot) 展平）。
// pad 行写 0 —— 生成物对 A 的读没有 mask，pad 段必须真的是 0（否则脏数据进 MMA）。
//
// `nseg` 是**指针**：host-table 入口传常驻 scratch `g_nseg`（上行一次），
// device-table 入口直接传调用方由 `dsv41_moe_align_from_group` 写出的
// `tl_nseg`。两条臂因此共用同一个 kernel，只有表的来源不同（A/B 基线）。
// ---------------------------------------------------------------------------
__global__ void tl_moe_gather_kernel(const float* __restrict__ act, __nv_bfloat16* __restrict__ a,
                                     const int* __restrict__ order, const int* __restrict__ counts,
                                     int K, int rpitch, int row_div, const int* nseg) {
    const int seg = blockIdx.y;
    if (seg >= *nseg) return;
    const int r = blockIdx.x;
    __nv_bfloat16* dst = a + ((size_t)seg * kTLBM + r) * K;
    const int live = counts[seg];
    if (r >= live || order[seg * kTLBM + r] < 0) {
        for (int k = threadIdx.x; k < K; k += kMovThreads) dst[k] = __float2bfloat16(0.f);
        return;
    }
    const int src = order[seg * kTLBM + r] / row_div;
    const float* s = act + (size_t)src * rpitch;
    for (int k = threadIdx.x; k < K; k += kMovThreads) dst[k] = __float2bfloat16(s[k]);
}

// ---------------------------------------------------------------------------
// scatter：把 C[seg*BM + r][0..N) 按 order 写回调用方的 out。
// up：out[(row*topk + slot)*2*inter + n]（(row,slot) = order[...] 的分解）——RAW gate‖up，
//     swiglu 仍由既有 `dsv41_swiglu_limit_batched` 做（最小改动：本臂只换 gate/up 那一步）。
// dn：out[order[...] * dim + n]（激活已按 (row,slot) 展平，直接是 per-slot partial）。
// `nseg` 是指针，与 gather 同约定（见上）。
// ---------------------------------------------------------------------------
__global__ void tl_moe_scatter_kernel(const float* __restrict__ c, float* __restrict__ out,
                                      const int* __restrict__ order, const int* __restrict__ counts,
                                      int N, int out_pitch, int split, const int* nseg) {
    const int seg = blockIdx.y;
    if (seg >= *nseg) return;
    const int r = blockIdx.x;
    const int live = counts[seg];
    const int idx = order[seg * kTLBM + r];
    if (r >= live || idx < 0) return;
    // up 的 out_pitch = topk*2*inter，分页 = (row*topk + slot)*2*inter；dn 直接 idx*dim。
    const long dst = (split > 0) ? ((long)(idx / split) * out_pitch + (long)(idx % split) * N)
                                 : (long)idx * out_pitch;
    const float* s = c + ((size_t)seg * kTLBM + r) * N;
    for (int n = threadIdx.x; n < N; n += kMovThreads) out[dst + n] = s[n];
}

// ---------------------------------------------------------------------------
// 加载期 dequant：fp4(e2m1 + ue8m0) -> bf16 副本。**与原型 fp4_dequant_ref 位级等价**：
//   nibble -> e2m1 幅值（bf16 精确表示），scale = 2^(b-127) 是纯 2 的幂 ⇒ bf16 位模式
//   相加（指数域）即精确乘法，无舍入（幅值 0 单独处理）。low nibble = 偶数 k。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float tl_e2m1_mag(unsigned idx) {
    // e2m1 幅值表（原型 moe_grouped_proto.py::LUT）：e*2+m -> {0,0.5,1,1.5,2,3,4,6}
    switch (idx & 7u) {
        case 0: return 0.0f;
        case 1: return 0.5f;
        case 2: return 1.0f;
        case 3: return 1.5f;
        case 4: return 2.0f;
        case 5: return 3.0f;
        case 6: return 4.0f;
        default: return 6.0f;
    }
}

__global__ void tl_moe_fp4_to_bf16_kernel(const uint8_t* __restrict__ wq,
                                          const uint8_t* __restrict__ ws,
                                          uint8_t* __restrict__ out, int n, int k, int wq_pitch,
                                          int ws_pitch) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const uint8_t* q = wq + (size_t)row * (size_t)wq_pitch;
    const uint8_t* sc = ws + (size_t)row * (size_t)ws_pitch;
    unsigned short* o = reinterpret_cast<unsigned short*>(out + (size_t)row * k * sizeof(__nv_bfloat16));
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
        const unsigned nb = (q[kk >> 1] >> ((kk & 1) * 4)) & 0xFu;
        // 幅值（e2m1，bf16 精确）× 2^(s-127)（2 的幂，f32 精确）⇒ 一次 f32->bf16 舍入，
        // 与原型 fp4_dequant_ref 的 `val * 2^(Ws-127)` 位级等价。
        const float v = tl_e2m1_mag(nb >> 1) * exp2f((float)sc[kk >> 5] - 127.0f);
        unsigned short bits = __bfloat16_as_ushort(__float2bfloat16((nb & 8u) ? -v : v));
        o[kk] = bits;
    }
}

}  // namespace

// ===========================================================================
// up（gate‖up）：grouped GEMM + scatter。返回 0 / 2 / cuda 错误码。
// ===========================================================================
extern "C" int dsv41_moe_tilelang_gate_up_bf16(
    const float* act,        // [rows, dim] f32（已 rmsnorm 的激活，一行一 token）
    float* out,              // [rows][topk][2*inter] f32（row pitch = topk*2*inter）
    const void* w_up,        // bf16 [E=384, 2*inter, dim]（N 前半 gate / 后半 up）
    const int* eid,          // [SEG_CAP] i32 —— HOST 数组
    const int* order,        // [SEG_CAP*BM] i32 —— HOST 数组（pad = -1）
    const int* counts,       // [SEG_CAP] i32 —— HOST 数组
    int nseg, int rows, int dim, int inter, int topk, cudaStream_t s) {
    if (act == nullptr || out == nullptr || w_up == nullptr || eid == nullptr || order == nullptr ||
        counts == nullptr)
        return 2;
    if (dim != kTLDim || inter != kTLInter) return 2;
    if (topk < 1 || topk > kTLTopkMax) return 2;
    if (rows < 1 || rows > kTLRowsMax) return 2;
    if (nseg < 1 || nseg > kTLSegCap) return 2;
    // 16B 对齐：A 走 cp.async 8/16B，W 走 cp.async 16B，C 是 float2 store。
    if ((((uintptr_t)act & 0xF) != 0) || (((uintptr_t)w_up & 0xF) != 0) ||
        (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit, v2 state-gated): only decline when INIT hasn't
    // completed — once scratch is allocated, launches are capture-safe and SHOULD
    // enter the verify graph (the head_bf16 pattern).
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone
        && g_a_up == nullptr) {
        return 2;  // INIT hasn't run yet — decline without touching capture
    }

    if (!tl_moe_init()) {
        init_failed_note(rows, dim, inter);
        return 2;
    }

    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[moe-tilelang] ARMED gate_up rows=%d dim=%d inter=%d topk=%d nseg=%d -> "
                    "grid=(%d,%d)x%d smem=%zu + gather/scatter\n",
                    rows, dim, inter, topk, nseg, kUpGridX, kTLSegCap, kUpThreads, kUpSmem);
    }

    // 一次性回执（tl-parity-vs-old #4：out_stride 的 row-0 免疫陷阱的 MoE 形态）。
    // 本入口没有 `out_stride` 参数 —— scatter 的行距是**烘死的**：assignment
    // idx = row*topk + slot 落在 out + idx*2*inter，token 行距 = topk*2*inter。调用方把
    // out 缓冲的行距/槽距摆错时，只有 rows > 1（或多 topk 槽）才暴露。这里把烘死的行距
    // 打一次；**out 缓冲的真实行距 shim 看不到，调用方必须自行核对**。
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang:moe_up] FIRST CALL rows=%d topk=%d n=%d inter=%d — caller "
                    "MUST verify out token pitch == topk*2*inter=%d and slot pitch == 2*inter=%d\n",
                    (int)rows, (int)topk, (int)kTLUpN, (int)inter, (int)(topk * kTLUpN),
                    (int)kTLUpN);
    }

    // (0) 元数据上行（小数组；见文件头「EAGER 臂」）。⚠️ H2D 上行在 CUDA-graph
    // capture 内非法 —— 这正是本入口只能走 eager 的原因，也是 device-table 入口
    // （`*_dev`，表已在 device 上）存在的理由。本入口保留为 A/B 基线。
    cudaError_t e = cudaMemcpyAsync(g_eid, eid, (size_t)nseg * sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpyAsync(g_order, order, (size_t)kTLSegCap * kTLBM * sizeof(int),
                        cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpyAsync(g_counts, counts, (size_t)nseg * sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpyAsync(g_nseg, &nseg, sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;

    // (1) gather：f32 激活 -> bf16 [SEG_CAP*BM, dim]，每段 pad 到 16 行。
    tl_moe_gather_kernel<<<dim3((unsigned)kTLBM, (unsigned)kTLSegCap), kMovThreads, 0, s>>>(
        act, g_a_up, g_order, g_counts, kTLDim, kTLDim, topk, g_nseg);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (2) grouped GEMM（生成物；参数序 = dump 的 (A, C, Eid, W)）。
    moe_up_tl_kernel<<<dim3((unsigned)kUpGridX, (unsigned)kTLSegCap), kUpThreads, kUpSmem, s>>>(
        reinterpret_cast<const bfloat16_t*>(g_a_up), g_c_up, g_eid,
        reinterpret_cast<const bfloat16_t*>(w_up));
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (3) scatter：RAW gate‖up 写回 out[row][slot][2*inter]（swiglu 由既有 kernel 做）。
    tl_moe_scatter_kernel<<<dim3((unsigned)kTLBM, (unsigned)kTLSegCap), kMovThreads, 0, s>>>(
        g_c_up, out, g_order, g_counts, kTLUpN, topk * kTLUpN, topk, g_nseg);
    e = cudaGetLastError();
    return (int)e;
}

// ===========================================================================
// down：grouped GEMM + scatter（per-slot partial 写 ex_down_r，reduce 仍走既有 kernel）。
// ===========================================================================
extern "C" int dsv41_moe_tilelang_down_bf16(
    const float* act,        // [rows*topk][inter] f32，**按 (row,slot) 展平且槽距 = act_pitch**
    float* out,              // [rows*topk][dim] f32（per-slot partial）
    const void* w_dn,        // bf16 [E=384, dim, inter]
    const int* eid, const int* order, const int* counts,
    int nseg, int rows, int dim, int inter, int topk, int act_pitch, cudaStream_t s) {
    if (act == nullptr || out == nullptr || w_dn == nullptr || eid == nullptr || order == nullptr ||
        counts == nullptr)
        return 2;
    if (dim != kTLDim || inter != kTLInter) return 2;
    // act_pitch 是调用方 `ex_act_r` 的**槽距**：swiglu 就地写后仍是 2*inter（非融合布局），
    // 融合布局则是 inter。两者都合法，但必须显式给出（默认的 inter 假设会读错位置）。
    if (act_pitch < kTLInter) return 2;
    if (topk < 1 || topk > kTLTopkMax) return 2;
    if (rows < 1 || rows > kTLRowsMax) return 2;
    if (nseg < 1 || nseg > kTLSegCap) return 2;
    if ((((uintptr_t)act & 0xF) != 0) || (((uintptr_t)w_dn & 0xF) != 0) ||
        (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit, v2 state-gated): only decline when INIT hasn't
    // completed — once scratch is allocated, launches are capture-safe and SHOULD
    // enter the verify graph (the head_bf16 pattern).
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone
        && g_a_up == nullptr) {
        return 2;  // INIT hasn't run yet — decline without touching capture
    }

    if (!tl_moe_init()) {
        init_failed_note(rows, dim, inter);
        return 2;
    }

    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[moe-tilelang] ARMED down rows=%d dim=%d inter=%d topk=%d nseg=%d -> "
                    "grid=(%d,%d)x%d smem=%zu + gather/scatter\n",
                    rows, dim, inter, topk, nseg, kDnGridX, kTLSegCap, kDnThreads, kDnSmem);
    }

    // 一次性回执（tl-parity-vs-old #4：out_stride 的 row-0 免疫陷阱的 MoE 形态）。
    // 本入口的 scatter 走 split == 0（dst = idx*dim，idx = row*topk + slot）：assignment
    // 行距 = dim。调用方把 out（`ex_down_r` 的 per-slot partial）的行距摆错时，只有
    // rows > 1 才暴露。这里把烘死的行距打一次；**out 缓冲的真实行距 shim 看不到，调用方
    // 必须自行核对它 == dim**。
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang:moe_dn] FIRST CALL rows=%d topk=%d n=%d — caller MUST "
                    "verify out per-slot pitch == dim=%d\n",
                    (int)rows, (int)topk, (int)dim, (int)kTLDim);
    }

    // 元数据上行（见 up 入口 / 文件头「EAGER 臂」；本入口同样只能走 eager）。
    cudaError_t e = cudaMemcpyAsync(g_eid, eid, (size_t)nseg * sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpyAsync(g_order, order, (size_t)kTLSegCap * kTLBM * sizeof(int),
                        cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpyAsync(g_counts, counts, (size_t)nseg * sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpyAsync(g_nseg, &nseg, sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;

    // gather：激活按 assignment 展平（row_div = 1），槽距 = act_pitch。
    tl_moe_gather_kernel<<<dim3((unsigned)kTLBM, (unsigned)kTLSegCap), kMovThreads, 0, s>>>(
        act, g_a_dn, g_order, g_counts, kTLInter, act_pitch, 1, g_nseg);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    moe_dn_tl_kernel<<<dim3((unsigned)kDnGridX, (unsigned)kTLSegCap), kDnThreads, kDnSmem, s>>>(
        reinterpret_cast<const bfloat16_t*>(g_a_dn), g_c_dn, g_eid,
        reinterpret_cast<const bfloat16_t*>(w_dn));
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // scatter：split == 0 ⇒ dst = idx * dim（assignment 直接索引 per-slot partial）。
    tl_moe_scatter_kernel<<<dim3((unsigned)kTLBM, (unsigned)kTLSegCap), kMovThreads, 0, s>>>(
        g_c_dn, out, g_order, g_counts, kTLDim, kTLDim, 0, g_nseg);
    e = cudaGetLastError();
    return (int)e;
}

// ===========================================================================
//  DEVICE-TABLE 入口（up / down）—— 让本臂进 CUDA-graph capture
// ===========================================================================
// 与上面两个 host-table 入口**逐行相同**，只有一个本质区别：三张表 + nseg
// **已经在 device 上**，由 `dsv41_moe_align_from_group`（kernels/cuda/
// dsv41_moe_align.cu）从 `dsv41_route_group` 的输出投影出来 —— 与 host 侧
// `moe_align_host` 的表逐位相同（等价性论证见该文件的头注释）。因此：
//   * **没有 D2H 回读、没有 H2D 上行**：整条链在 capture 内合法，这正是本入口
//     存在的理由（host-table 入口的 `cudaMemcpyAsync(..., HostToDevice, s)` 是
//     capture 里的非法操作，所以它被 `moe_tilelang_ready()` 的 `!capturing()`
//     挡在 graph 外 —— 那 −7.1ms 就压在这一点上）；
//   * `nseg` 是 DEVICE 指针：没有 host 形状检查（设备上的值 host 读不到），
//     边界由 `dsv41_moe_align_from_group` 的 `min(n_active, SEG_CAP)` 和两个
//     mover 的 `seg >= *nseg` 守卫保证；`grid.y = SEG_CAP` 恒定。
// 旧入口保留为 A/B 基线：同一 `ARMED` 回执格式，便于逐行对比两条臂。
// ===========================================================================
extern "C" int dsv41_moe_tilelang_gate_up_bf16_dev(
    const float* act,        // [rows, dim] f32（已 rmsnorm 的激活，一行一 token）
    float* out,              // [rows][topk][2*inter] f32（row pitch = topk*2*inter）
    const void* w_up,        // bf16 [E=384, 2*inter, dim]（N 前半 gate / 后半 up）
    const int* eid,          // [SEG_CAP] i32 —— **DEVICE**
    const int* order,        // [SEG_CAP*BM] i32 —— **DEVICE**（pad = -1）
    const int* counts,       // [SEG_CAP] i32 —— **DEVICE**
    const int* nseg_dev,     // [1] i32 —— **DEVICE**（dsv41_moe_align_from_group 的输出）
    int rows, int dim, int inter, int topk, cudaStream_t s) {
    if (act == nullptr || out == nullptr || w_up == nullptr || eid == nullptr || order == nullptr ||
        counts == nullptr || nseg_dev == nullptr)
        return 2;
    if (dim != kTLDim || inter != kTLInter) return 2;
    if (topk < 1 || topk > kTLTopkMax) return 2;
    if (rows < 1 || rows > kTLRowsMax) return 2;
    // nseg 的形状检查在这里**不存在**（device 上的值 host 读不到）——见上方文件头。
    if ((((uintptr_t)act & 0xF) != 0) || (((uintptr_t)w_up & 0xF) != 0) ||
        (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit, v2 state-gated): only decline when INIT hasn't
    // completed — once scratch is allocated, launches are capture-safe and SHOULD
    // enter the verify graph (the head_bf16 pattern). This is the guard that makes
    // `cudaMalloc` unreachable from inside a capture.
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone
        && g_a_up == nullptr) {
        return 2;  // INIT hasn't run yet — decline without touching capture
    }

    if (!tl_moe_init()) {
        init_failed_note(rows, dim, inter);
        return 2;
    }

    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[moe-tilelang] ARMED gate_up (device tables) rows=%d dim=%d inter=%d topk=%d "
                    "nseg_dev=<device> -> grid=(%d,%d)x%d smem=%zu + gather/scatter\n",
                    rows, dim, inter, topk, kUpGridX, kTLSegCap, kUpThreads, kUpSmem);
    }

    // 一次性回执（与 host-table 入口同一行距契约，见那里的长注释）。
    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang:moe_up_dev] FIRST CALL rows=%d topk=%d n=%d inter=%d — caller "
                    "MUST verify out token pitch == topk*2*inter=%d and slot pitch == 2*inter=%d\n",
                    (int)rows, (int)topk, (int)kTLUpN, (int)inter, (int)(topk * kTLUpN),
                    (int)kTLUpN);
    }

    // 表已在 device 上 ⇒ 没有 (0) 元数据上行这一步。

    // (1) gather：f32 激活 -> bf16 [SEG_CAP*BM, dim]，每段 pad 到 16 行。
    tl_moe_gather_kernel<<<dim3((unsigned)kTLBM, (unsigned)kTLSegCap), kMovThreads, 0, s>>>(
        act, g_a_up, order, counts, kTLDim, kTLDim, topk, nseg_dev);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (2) grouped GEMM（生成物；参数序 = dump 的 (A, C, Eid, W)）。
    moe_up_tl_kernel<<<dim3((unsigned)kUpGridX, (unsigned)kTLSegCap), kUpThreads, kUpSmem, s>>>(
        reinterpret_cast<const bfloat16_t*>(g_a_up), g_c_up, eid,
        reinterpret_cast<const bfloat16_t*>(w_up));
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (3) scatter：RAW gate‖up 写回 out[row][slot][2*inter]（swiglu 由既有 kernel 做）。
    tl_moe_scatter_kernel<<<dim3((unsigned)kTLBM, (unsigned)kTLSegCap), kMovThreads, 0, s>>>(
        g_c_up, out, order, counts, kTLUpN, topk * kTLUpN, topk, nseg_dev);
    e = cudaGetLastError();
    return (int)e;
}

// down 的 device-table 孪生体（语义见上）。
extern "C" int dsv41_moe_tilelang_down_bf16_dev(
    const float* act,        // [rows*topk][inter] f32，**按 (row,slot) 展平且槽距 = act_pitch**
    float* out,              // [rows*topk][dim] f32（per-slot partial）
    const void* w_dn,        // bf16 [E=384, dim, inter]
    const int* eid, const int* order, const int* counts, const int* nseg_dev,
    int rows, int dim, int inter, int topk, int act_pitch, cudaStream_t s) {
    if (act == nullptr || out == nullptr || w_dn == nullptr || eid == nullptr || order == nullptr ||
        counts == nullptr || nseg_dev == nullptr)
        return 2;
    if (dim != kTLDim || inter != kTLInter) return 2;
    if (act_pitch < kTLInter) return 2;
    if (topk < 1 || topk > kTLTopkMax) return 2;
    if (rows < 1 || rows > kTLRowsMax) return 2;
    if ((((uintptr_t)act & 0xF) != 0) || (((uintptr_t)w_dn & 0xF) != 0) ||
        (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2（见 up 的 device-table 入口）。
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone
        && g_a_up == nullptr) {
        return 2;
    }

    if (!tl_moe_init()) {
        init_failed_note(rows, dim, inter);
        return 2;
    }

    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[moe-tilelang] ARMED down (device tables) rows=%d dim=%d inter=%d topk=%d "
                    "nseg_dev=<device> -> grid=(%d,%d)x%d smem=%zu + gather/scatter\n",
                    rows, dim, inter, topk, kDnGridX, kTLSegCap, kDnThreads, kDnSmem);
    }

    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[proj-tilelang:moe_dn_dev] FIRST CALL rows=%d topk=%d n=%d — caller MUST "
                    "verify out per-slot pitch == dim=%d\n",
                    (int)rows, (int)topk, (int)dim, (int)kTLDim);
    }

    // gather：激活按 assignment 展平（row_div = 1），槽距 = act_pitch。
    tl_moe_gather_kernel<<<dim3((unsigned)kTLBM, (unsigned)kTLSegCap), kMovThreads, 0, s>>>(
        act, g_a_dn, order, counts, kTLInter, act_pitch, 1, nseg_dev);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    moe_dn_tl_kernel<<<dim3((unsigned)kDnGridX, (unsigned)kTLSegCap), kDnThreads, kDnSmem, s>>>(
        reinterpret_cast<const bfloat16_t*>(g_a_dn), g_c_dn, eid,
        reinterpret_cast<const bfloat16_t*>(w_dn));
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // scatter：split == 0 ⇒ dst = idx * dim（assignment 直接索引 per-slot partial）。
    tl_moe_scatter_kernel<<<dim3((unsigned)kTLBM, (unsigned)kTLSegCap), kMovThreads, 0, s>>>(
        g_c_dn, out, order, counts, kTLDim, kTLDim, 0, nseg_dev);
    e = cudaGetLastError();
    return (int)e;
}

// ===========================================================================
// 加载期 dequant（DSV41_MOE_BF16_DEQUANT）：fp4(e2m1+ue8m0) -> bf16 副本。
// 逐平面（w1/w3/w2）调用；n/k 是**本 rank 切片**的行/列。rc 语义同上（0 / 2 / 错误）。
// ===========================================================================
extern "C" int dsv41_moe_fp4_to_bf16(const void* wq, const void* ws, void* out_bf16, int n, int k,
                                     int wq_pitch, int ws_pitch, cudaStream_t s) {
    if (wq == nullptr || ws == nullptr || out_bf16 == nullptr) return 2;
    if (n <= 0 || k <= 0) return 2;
    if ((k % 32) != 0) return 2;  // ue8m0 是 32 元素一块
    // 行距：<= 0 表示用自然行距（k/2 与 k/32）。ferrite 的 DSV41_SF_STRIDE_PAD（默认 ON）
    // 会把 scale 平面的物理行距补到 16B 整数倍（w2.scale 10B -> 16B），所以调用方必须
    // 显式给出；否则会静默读错 scale（权重错、不报错）。
    if (wq_pitch <= 0) wq_pitch = k / 2;
    if (ws_pitch <= 0) ws_pitch = k / 32;
    if (wq_pitch < k / 2 || ws_pitch < k / 32) return 2;
    if ((((uintptr_t)wq & 0xF) != 0) || (((uintptr_t)ws & 0xF) != 0) ||
        (((uintptr_t)out_bf16 & 0xF) != 0))
        return 2;
    tl_moe_fp4_to_bf16_kernel<<<dim3((unsigned)n), 256, 0, s>>>(
        reinterpret_cast<const uint8_t*>(wq), reinterpret_cast<const uint8_t*>(ws),
        reinterpret_cast<uint8_t*>(out_bf16), n, k, wq_pitch, ws_pitch);
    return (int)cudaGetLastError();
}
