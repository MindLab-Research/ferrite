// moe_bs_dn_shim.cu — block-scaled fp4 MoE **DOWN** 方向的生产接线件
// =============================================================================
// 上位文档：docs/agent/moe-down-bs-design.md
//           docs/agent/moe-bs-crash-investigation.md §47（fp4 smem 语义定谳）、§100（缺口）
// 同族先例：tilelang_gen/moe_bs_shim.cu（gate/up 的 blockscaled 臂，本文件逐项对齐其契约）
//
// =============================================================================
// 它替代的是什么
// =============================================================================
// routed 的 down 方向（`expert_gemv_fp4_down_reduce_kernel`，v3 profile 1.00 ms/步、10.3%，
// 385 GB/s、latency-bound）。本 shim 用 B300 原生 block-scaled MMA 直接吃 fp4 的 w2：
//   **e4m3 激活 x e2m1 权重 + ue8m0 标度进 tensor core，零 dequant、零 bf16 副本。**
// ⚠️ 激活必须是 e4m3（1 B/value）—— 这是硬件 kind::mxf8f6f4 的 a_fmt 要求，不是选择：
//    down 的输入**必然**经过一次 block-32 的 e4m3 量化。官方参考 (`ref_inference/model.py:846-849`)
//    同样把 down 输入量化到 e4m3(block=32) ⇒ 本臂在这点上**比 SIMT 更接近官方**（SIMT 吃 f32），
//    这既是本臂相对 SIMT 的**唯一算法差**（~1e-3 量级），也是**精度对齐的方向**（见设计文档 §5）。
//
// =============================================================================
// 导出符号
// =============================================================================
//   int dsv41_moe_bs_down_dev(...)  -- down 的 blockscaled grouped GEMM（DEVICE 段表），
//                                     一次调用完成：量化+gather → 每段 grouped MMA → 散写到
//                                     `out`（per-assignment partial，与 SIMT 的 [slot][dim]
//                                     布局逐格相同 ⇒ 既有 ascending `moe_down_reduce` 直接复用）
//   int dsv41_moe_bs_down_cap(void) -- **能力符号**：本 .so 带 down BS 臂。旧 .so 没有它
//                                     ⇒ Rust 侧不 arm（能力探针的既有模式）
//
// rc 契约（与 wkv / moe_bs / proj_mma / bf16 shim 完全一致）：
//   0 = 已发射；2 = DECLINED（形状/模式/门控不接受，或 INIT 失败）；其它 = cudaGetLastError()
//
// =============================================================================
// 门控（**全部默认 OFF**；本文件的每个门都是"编译进来但运行期关闭"）
// =============================================================================
//   DSV41_MOE_DOWN_BS[=1]        总门。**未设或非 1 ⇒ 本入口直接 return 2（DECLINED）**
//                                ⇒ 即使调用方接了线，OFF 也是零动作（OFF 等价性见设计文档 §6）。
//   DSV41_MOE_DOWN_BS_CPASYNC    1（默认）= cp.async 双缓冲 staging；0 = 顺序 LDG/STS
//                                （逐字节同一批地址，只差延迟是否重叠）
//   DSV41_MOE_DOWN_BS_WAITDBG    1 = MMA 有界等待超时时打详细状态（打印门，**有界性本身无门**）
//   DSV41_MOE_DOWN_BS_RWOP       1 = 路由权重乘进**操作数**（= 官方 `x = weights*x` 之后再量化，
//                                    顺序与 `DSV41_ROUTED_DOWN_QUANT` 一致；两者**互斥**，
//                                    同时开会重复乘）；0（默认）= epilogue 乘（与 SIMT 同序）
//
// =============================================================================
// 布局契约（host 必须对齐；与 gate/up 臂同一套段表）
// =============================================================================
//   act   : [rows][slots][pitch] f32  —— swiglu 后的 down 输入；pitch = 上游 gate/up 写的
//                                       slot 间隔（`2*inter` = RAW 布局；融合 swiglu 后是 `inter`）。
//                                       只读前 `inter` 个 float（与 SIMT down 核一致）。
//   w2    : u8 [E, dim, inter/2]      —— 原生 packed fp4 池（面内 K 连续，行距 inter/2 = 160 B）
//   w2s   : u8 [E, dim, pitch_sf]     —— e8m0 面（**物理行距 = 16 的 pad 面**，见 dsv41_sf_pitch）
//   eid/order/counts/nseg : DEVICE 表 —— 与 BS gate/up 臂**同一套**（BM=128 段表）
//   out   : f32 [rows*slots][dim]     —— per-assignment partial（OVERWRITE），
//                                        out[(idx)*dim + n]，idx = row*topk+slot = order[seg*BM+r]
//
// ⚠️ 本文件**不**做 reduce、不做 atomic：不同段/不同 N-tile 写的 (assignment, dim 列) 互不相交。
//    调用方仍必须跑既有的 ascending-slot `moe_down_reduce`（数值契约：fp 加法不结合）。
//
// =============================================================================
// 四条硬约束（与 moe_bs_shim.cu 同构）
// =============================================================================
//  1. **无 TMA**：本 kernel 用普通 LDG/STS + cp.async（不是 TMA/cp.async.bulk）⇒ 不需要
//     `cuTensorMapEncodeTiled`、不需要 dlopen libcuda、不需要 `-lcuda`。定义域因此更窄也更强：
//     没有描述符可建错（gate/up 臂的两个静默错值根因都是 TMA 描述符参数）。
//  2. `cudaFuncSetAttribute` / `cudaMalloc` 只在 **INIT** 期（capture 内调用会让
//     `cudaStreamEndCapture` 失败）；INIT 失败一律 decline，绝不半发射。
//  3. 用调用方传入的 stream（必须进 capture / 与主流同序）。
//  4. 形状不合规 `return 2`（不是 1 / 负数）。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "moe_bs_dn_handwritten.cu"

// =============================================================================
// §1 常驻 scratch（INIT 期分配一次，进程生命周期复用）
// =============================================================================
namespace {
// gather 的输出：e4m3 激活，[SEG*BM][K]，行距 = K（= inter = 320）
uint8_t* g_dn_a = nullptr;
// 激活 SF：group-major u32，[SPANS][SEG*BM]（与 gate/up 臂的 SFA 池同构）
uint32_t* g_dn_sfa = nullptr;
// 环境门（读一次）
bool g_dn_cpasync = true;
bool g_dn_rwop = false;
}  // namespace

// =============================================================================
// §2 量化 + gather（每调用一次；把 f32 的 down 输入搬进段缓冲并就地量化成 e4m3）
// =============================================================================
// 逐项复刻官方 `act_quant(block=32, scale_fmt=ue8m0)` 的语义（`ref_inference/kernel.py:75-95`），
// 与 `dsv41_glue.cu::routed_down_prep_kernel` 的量化部分同源：
//     amax = max|v| over the 32-block, floored at 1e-4（**参考的**地板，不是本项目的 1e-30）
//     sc   = fast_round_scale(amax, 1/448) = 2^ceil(log2(amax/448))（2 的幂 ⇒ ue8m0 精确）
//     q    = e4m3(clamp(v / sc, +-448))          （RN + 饱和 == __nv_fp8_e4m3）
// 与 routed_down_prep 的**唯一**差别：本 kernel 默认**不乘路由权重、也不做 bf16 边界**
// （那是 `rw_in_operand=1` 的语义，等价于 DSV41_ROUTED_DOWN_QUANT 的顺序）。
//
// 一个 warp = 一个 32 元素 scale block（amax 需要整 warp 的 shuffle 树）：
//   blockIdx.x = K 上的 32-block 下标（0..K/32-1）
//   blockIdx.y = 段
//   段内行按 warp 轮转（warp = 0..3, r = warp, warp+4, ...）
// pad 行（r >= counts[seg] 或 order == -1）写 e4m3 0x00（+0.0）与 SF 字节 0
// ⇒ 0 x 任意有限标度 = 0，绝不产生 NaN。
__device__ __forceinline__ float dn_fast_round_scale(float amax, float max_inv) {
    // 与 dsv41_glue.cu:glue_fast_round_scale 逐字相同（= kernel.py 的 fast_round_scale）
    const uint32_t bits = __float_as_uint(amax * max_inv);
    const int exp = (int)((bits >> 23) & 0xFFu);
    const uint32_t man = bits & 0x7FFFFFu;
    const int e = exp - 127 + (man != 0 ? 1 : 0);
    return __int_as_float((e + 127) << 23);
}
__device__ __forceinline__ uint8_t dn_pow2_to_ue8m0(float s) {
    if (!(s > 0.f)) return 0;
    int e = (int)((__float_as_uint(s) >> 23) & 0xFFu) - 127;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (uint8_t)(e + 127);
}
__device__ __forceinline__ float dn_bf16_round(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__global__ void __launch_bounds__(128) dn_qgather_kernel(
    const float* __restrict__ act, const float* __restrict__ route_w, uint8_t* __restrict__ a,
    uint32_t* __restrict__ sfa, const int* __restrict__ order, const int* __restrict__ counts,
    const int* __restrict__ nseg, int rows_slots, int slots, int topk, int pitch, int k,
    int rw_in_operand) {
    const int seg = blockIdx.y;
    if (seg >= *nseg) return;
    const int b = blockIdx.x;  // 32-element block index
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int live = counts[seg] < DN_BM ? counts[seg] : DN_BM;
    const int i0 = b << 5;
    const bool live_lane = (i0 + lane) < k;
    for (int r = warp; r < DN_BM; r += 4) {
        const int idx = (r < live) ? order[seg * DN_BM + r] : -1;
        uint8_t q = 0;
        uint8_t sfb = 0;
        if (idx >= 0 && idx < rows_slots) {
            const int row = idx / topk;
            const int slot = idx - row * topk;
            const float* src = act + ((size_t)row * (size_t)slots + (size_t)slot) * (size_t)pitch +
                               (size_t)i0;
            float v = live_lane ? src[lane] : 0.f;
            if (route_w != nullptr && rw_in_operand) v *= route_w[idx];
            if (rw_in_operand) v = dn_bf16_round(v);
            float a = fabsf(v);
            for (int off = 16; off > 0; off >>= 1)
                a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
            const float amax = fmaxf(a, 1e-4f);
            const float sc = dn_fast_round_scale(amax, 1.0f / 448.0f);
            const float qf = fminf(fmaxf(v * (1.0f / sc), -448.0f), 448.0f);
            const __nv_fp8_e4m3 f8(qf);
            q = *reinterpret_cast<const uint8_t*>(&f8);
            sfb = dn_pow2_to_ue8m0(sc);
        }
        if (live_lane) {
            a[((size_t)seg * DN_BM + (size_t)r) * (size_t)k + (size_t)(i0 + lane)] = q;
        }
        if (lane == 0) {
            // SF 字 = [spans][SEG*BM] group-major：字 (b>>2)*SEG*BM + seg*BM + r 的第 (b&3) 字节
            uint32_t* w = sfa + (size_t)(b >> 2) * (size_t)(DN_SEGCAP * DN_BM) +
                          (size_t)seg * DN_BM + (size_t)r;
            reinterpret_cast<uint8_t*>(w)[b & 3] = sfb;
        }
    }
}

// =============================================================================
// §3 INIT（懒初始化；SetAttribute / cudaMalloc / 环境门 只在这里）
// =============================================================================
static bool dn_bs_init() {
    static int state = 0;
    if (state != 0) return state > 0;
    static size_t g_dn_smem = 0;
    {
        const char* e = getenv("DSV41_MOE_DOWN_BS_CPASYNC");
        g_dn_cpasync = (e == nullptr) ? true : (e[0] != '0');  // 默认 ON
        e = getenv("DSV41_MOE_DOWN_BS_RWOP");
        g_dn_rwop = (e != nullptr && e[0] == '1');
        g_dn_smem = g_dn_cpasync ? DN_SMEM_DOUBLE : DN_SMEM_SINGLE;
    }
    bool ok = cudaFuncSetAttribute(moe_bs_dn_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)DN_SMEM_DOUBLE) == cudaSuccess;
    (void)cudaGetLastError();
    if (ok) {
        int cp = g_dn_cpasync ? 1 : 0, wd = 0, lw = 1;
        const char* e = getenv("DSV41_MOE_DOWN_BS_WAITDBG");
        if (e != nullptr && e[0] == '1') wd = 1;
        // The TMEM lane field is an absolute lane coordinate (PTX 9.7.18.1.1): the default is the
        // corrected per-warp spelling. `DSV41_MOE_DOWN_BS_LDW=0` restores lane 0 for every warp so
        // the fix can be attributed in one build instead of only pass/fail.
        e = getenv("DSV41_MOE_DOWN_BS_LDW");
        if (e != nullptr && e[0] == '0') lw = 0;
        (void)cudaMemcpyToSymbol(dn_g_cpasync, &cp, sizeof(int));
        (void)cudaMemcpyToSymbol(dn_g_waitdbg, &wd, sizeof(int));
        (void)cudaMemcpyToSymbol(dn_ldw, &lw, sizeof(int));
        (void)cudaGetLastError();
        fprintf(stderr, "[moe-dn-bs] cpasync=%d rw_in_operand=%d smem=%zu waitdbg=%d ldw=%d\n", cp,
                (int)g_dn_rwop, g_dn_smem, wd, lw);
    }
    if (ok) {
        ok = cudaMalloc(&g_dn_a, (size_t)DN_SEGCAP * DN_BM * DN_K) == cudaSuccess &&
             cudaMalloc(&g_dn_sfa, (size_t)DN_SPANS * DN_SEGCAP * DN_BM * 4) == cudaSuccess;
        (void)cudaGetLastError();
    }
    if (!ok) {
        state = -1;
        return false;
    }
    state = 1;
    return true;
}

static void dn_bs_init_failed_note(int rows, int dim, int inter) {
    static int reported = 0;
    if (reported++ == 0)
        fprintf(stderr,
                "[moe-dn-bs] ARMED but INIT FAILED (SetAttribute/malloc) -> this run measures the "
                "OLD path (rows=%d dim=%d inter=%d)\n",
                rows, dim, inter);
}

static void dn_bs_skipped_note(const char* why) {
    static int reported = 0;
    if (reported++ == 0)
        fprintf(stderr, "[moe-dn-bs] skipped: %s -> the down direction stays on the SIMT path\n",
                why);
}

// =============================================================================
// §4 导出符号 1：down blockscaled grouped GEMM（DEVICE 段表 —— 与 BS gate/up 臂同一套表）
// =============================================================================
// 调用契约与 `dsv41_expert_down_fp4_batched` 对齐（便于 A/B 与回退）：
//   act_base/act_stride : down 输入（只读前 `inter` 个 float/槽）
//   out                 : per-assignment partial（OVERWRITE；调用方随后跑 moe_down_reduce）
//   rows/slots/topk/dim/inter : 形状（rows*topk = assignment 数 ≤ SEG_CAP）
//   route_w (rw_stride=1)     : 每 assignment 的标量；nullptr = 不加权
// 与 SIMT 入口不同、必须显式化的三点：
//   * 需要 **w2 的 e8m0 面**（w2s_base/w2s_stride）与 **BS 段表**（eid/order/counts/nseg）；
//   * `rw_in_operand` 决定路由权重的落点（见文件头门控表）；
//   * 输出是 **partial**（SIMT 的 `expert_down_fp4_batched` 也是 partial；融合的
//     `expert_down_reduce_fp4_batched` 才自带 reduce）。⇒ 调用点复用 non-fused 分支的
//     `moe_down_reduce` 合并，**数值契约与那条臂逐项相同**。
extern "C" int dsv41_moe_bs_down_dev(
    const float* act_base, long act_stride, float* out, const uint8_t* w2_base, long w2_stride,
    const uint8_t* w2s_base, long w2s_stride, const int* eid_dev, const int* order_dev,
    const int* counts_dev, const int* nseg_dev, const float* route_w, long rw_stride, int rows,
    int dim, int inter, int topk, cudaStream_t stream) {
    // ---- 门控：默认 OFF = 零动作（OFF 等价性的第一层）----
    {
        const char* e = getenv("DSV41_MOE_DOWN_BS");
        if (!(e != nullptr && e[0] == '1')) return 2;
    }
    // ---- 形状/指针前置检查（任何一条不满足都 DECLINE，绝不静默）----
    // ⚠️ ARMED 但形状不在定义域时**必须出声**（本项目 #1 测量偏置陷阱：静默回落会让这一轮
    //    的 e2e 数字被当成"新臂"读）。所有拒绝走同一条 note。
    if (act_base == nullptr || out == nullptr || w2_base == nullptr || w2s_base == nullptr ||
        eid_dev == nullptr || order_dev == nullptr || counts_dev == nullptr || nseg_dev == nullptr) {
        dn_bs_skipped_note("a required pointer is null");
        return 2;
    }
    if (rows <= 0 || dim <= 0 || topk < 1 || rows * topk > DN_SEGCAP) {
        dn_bs_skipped_note("rows/dim/topk outside the arm's domain (rows*topk must be <= SEG_CAP)");
        return 2;
    }
    if (inter != DN_K) {
        dn_bs_skipped_note("inter != 320 (the frozen down contraction; the K-span geometry is baked)");
        return 2;
    }
    if (dim % DN_BN != 0 || dim > 8192) {
        dn_bs_skipped_note("dim is not a multiple of 128 (N-tile) or exceeds the design range");
        return 2;
    }
    if (w2_stride <= 0 || w2s_stride <= 0 || ((w2_stride & 7) != 0) || ((w2s_stride & 3) != 0)) {
        dn_bs_skipped_note("w2_stride % 8 != 0 or w2s_stride % 4 != 0 (cp.async alignment)");
        return 2;
    }
    if ((((uintptr_t)w2_base) & 7u) != 0u || ((((uintptr_t)w2s_base) & 3u) != 0u)) {
        dn_bs_skipped_note("w2/w2s base pointer is not 8/4-byte aligned");
        return 2;
    }
    if ((((uintptr_t)out) & 15u) != 0u) {
        dn_bs_skipped_note("out is not 16-byte aligned (the epilogue is a float4 store)");
        return 2;
    }
    if (route_w != nullptr && rw_stride != 1) {
        dn_bs_skipped_note("route_w rw_stride != 1 (the arm indexes the flat [rows*slots] plane)");
        return 2;
    }
    // w2s 的物理行距必须以 4 字节为单位（SFB 直接读 u32）；生产配置是 16。
    {
        // 与 kernels/cuda/dsv41_experts_mxf4.cu::dsv41_sf_pitch 同式：(k/32 + 15) & ~15，
        // 但这里只要求它是 4 的倍数（DSV41_SF_STRIDE_PAD=0 时 = k/32 = 10 也成立）。
        const int pitch = inter >> 5;
        const bool pad_on = true;  // 见下方注释：kernel 直接吃 4 字节，两种布局都合法
        const int w2s_pitch = pad_on ? ((pitch + 15) & ~15) : pitch;
        if ((w2s_pitch & 3) != 0 || w2s_pitch < (inter >> 5)) return 2;
        if (!dn_bs_init()) {
            dn_bs_init_failed_note(rows, dim, inter);
            return 2;
        }
        {
            static int reported = 0;
            if (reported++ == 0)
                fprintf(stderr,
                        "[moe-dn-bs] ARMED down_bs rows=%d dim=%d inter=%d topk=%d "
                        "grid=(%d,%d)x128 smem=%zu BM=%d BN=%d K=%d spans=%d\n",
                        rows, dim, inter, topk, dim / DN_BN, DN_SEGCAP, DN_SMEM_DOUBLE, DN_BM,
                        DN_BN, DN_K, DN_SPANS);
        }
        // (1) 量化 + gather：f32 down 输入 -> e4m3 段缓冲 + group-major SF
        dn_qgather_kernel<<<dim3((unsigned)(inter / 32), (unsigned)DN_SEGCAP), 128, 0, stream>>>(
            act_base, route_w, g_dn_a, g_dn_sfa, order_dev, counts_dev, nseg_dev, rows * topk,
            topk, topk, (int)act_stride, inter, g_dn_rwop ? 1 : 0);
        cudaError_t e1 = cudaGetLastError();
        if (e1 != cudaSuccess) return (int)e1;
        // (2) grouped MMA（每段一个专家；blockIdx.x = dim 的 N-tile，blockIdx.y = 段）
        moe_bs_dn_kernel<<<dim3((unsigned)(dim / DN_BN), (unsigned)DN_SEGCAP), 128,
                           g_dn_cpasync ? DN_SMEM_DOUBLE : DN_SMEM_SINGLE, stream>>>(
            g_dn_a, g_dn_sfa, w2_base, w2s_base, eid_dev, order_dev, route_w, out, dim,
            g_dn_rwop ? 1 : 0, w2_stride, w2s_stride, w2s_pitch);
        cudaError_t e2 = cudaGetLastError();
        if (e2 != cudaSuccess) return (int)e2;
    }
    return 0;
}

// 能力符号（Rust 侧 `supports_moe_bs_down()` 探它）。旧 .so 没有 ⇒ 老路径（能力探针惯例）。
extern "C" int dsv41_moe_bs_down_cap(void) { return 1; }
