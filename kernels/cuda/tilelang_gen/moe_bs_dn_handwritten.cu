// moe_bs_dn_handwritten.cu — 手写版 block-scaled fp4 MoE **DOWN** 方向 MMA kernel
// =============================================================================
// 上位文档：docs/agent/moe-down-bs-design.md（设计全文）
//           docs/agent/moe-bs-crash-investigation.md §47（fp4 smem 语义定谳）、§100（缺口）
//
// WHY THIS EXISTS（缺口）
// -----------------------------------------------------------------------------
// routed 的 gate/up 已走 tcgen05 block-scaled MMA（`moe_bs_shim.cu`），但 **down 仍是
// SIMT `expert_gemv_fp4_down_reduce_kernel`**（`dsv41_experts_mxf4.cu:2258`）：v3 profile
// **1.00 ms/步、10.3%**，且是占用率受限（`__launch_bounds__(256,6)` = 1.01 waves、实测
// 385 GB/s 远低于 HBM 峰值 ⇒ latency-bound）。本文件是 down 方向的 blockscaled 对应物，
// **复用 §47 定谳的 fp4 smem 语义**（`dn_pack_sw128` = `hw_pack_sw128` 逐字），不发明新语义。
//
// 复用的定谳结论（一字不改，逐项见 §47）
//   * fp4 操作数 = **packed（2 元素/字节，低 nibble = 偶 k）**，放在 **16 B 容器里、
//     硬件只读每槽前 8 B**（TMA dtype 16U4_ALIGN16B = 16 个 4-bit 元素 = 8 B 数据/16 B 容器）
//     ⇒ 一行 64 packed B 折成 8 容器 = **128 B footprint**；
//   * 写公式 `dn_pack_sw128(row,p)`（= §47 的 hw_pack_sw128）；
//   * 描述符 `lbo=1, sbo=64, layout_type=2`、K 块递进 `ki*32 B`（= ki*2 个 16 B 单位）、
//     idesc `(sf_id<<4) | (a_fmt<<7) | (b_fmt<<10) | ((n>>3)<<17) | (1<<23) | ((m>>4)<<24) | (sf_id<<29)`；
//   * SF 投递：smem 内 **group-major u32（byte j = K-block j）** → `tcgen05.cp.32x128b.warpx4`
//     → TMEM SF 列 → MMA 的 `sf_id` 选 byte。
//
// =============================================================================
// 方向与 A/B 角色推导（与 gate/up 相反的那一步）
// =============================================================================
// down 的数学：`y[dim] = Σ_t x[t] · W2[dim, t]`，t 是 contraction（= inter = 320）、dim 是输出。
// gate/up 的数学：`y[2*inter] = Σ_dim x[dim] · W1[inter, dim]`。
// 两者**不是同一个矩阵的转置**：gate/up 的权重是 [inter, dim]（N = inter = 320，K = dim = 5120），
// down 的权重是 [dim, inter]（N = dim = 5120，K = inter = 320）⇒ **N 与 K 互换角色**，
// 但 **A/B 的"身份"不变**（A = e4m3 激活、B = packed fp4 权重，K-major 约定 B[N,K]）。
//
// 为什么不是 swapAB（A = 权重）？**由写出决定**：
//   * A = 激活 ⇒ MMA 的 M = 段内 assignment 行（128），N = dim 的 128 列 ⇒ C[row][col] 的
//     一行 = dim 上连续的 128 个输出值 ⇒ 每行一次 512 B 连续写（4 个满 128 B sector）；
//   * A = 权重（swapAB）⇒ M = dim 行、N = assignment 行 ⇒ C 的一行 = 128 个**不同 assignment**
//     （不同 (row,slot)）的输出 ⇒ 单元素散写，写合并完全崩掉。
// ⇒ down 的 A/B 角色由**写合并**唯一确定（gate/up 没有这个约束，因为它的 M = token 行、
//    N = 权重行都能自然写出）。本 kernel 取 **A = 激活 / B = 权重**。
//
// =============================================================================
// 几何（down 生产形状；与 gate/up 臂同族的冻结形状契约）
// =============================================================================
//   M = 128（一段 = 一个专家的 assignment 行，bake 自 SEG_CAP/BM，见 moe_bs_shim.cu）
//   N = 128（dim 上的 N-tile；dim = 5120 ⇒ 40 个 tile ⇒ grid.x = 40）
//   K = 320（= inter = 320）——**不是 128 的整数倍**：320 = 128 + 128 + 64
//       ⇒ **3 个 K-span**：span0/1 各 4 个 K-block(32)，span2 只 2 个 ⇒ 共 10 个 MMA
//         （gate/up 是 40 次迭代 x 4 = 160 个）。**不做 K 补零到 384**：补零会引入
//         "pad 区的 SF 字节必须为 0"这一类静默 NaN 风险（0 x 2^128 = NaN），
//         而 10 个 MMA 的写法与 12 个的代价相同（MMA 数不是瓶颈，B 的带宽才是）。
//   每个 span 的 smem 是**独立的 128x128B SW128 tile**（span 之间靠 tile 基址切换，
//   span 内靠 ki*32B 递进）⇒ `dn_pack_sw128` / `dn_sw128_16b` 逐字复用，无需新公式。
//
// =============================================================================
// 数值语义（与 SIMT 参考的关系 —— 见设计文档 §5）
// =============================================================================
//   * A 操作数**必须是 e4m3**（硬件 kind::mxf8f6f4 的 a_fmt=0）⇒ down 的输入**必然**经历
//     block-32 e8m0 量化。这不是本实现的选择，是 tensor core 的输入格式要求。
//     官方参考 (`ref_inference/model.py:846-849`) 同样把 down 输入量化到 e4m3(block=32)
//     ⇒ 本臂在这一点上**比 SIMT 更接近官方**（SIMT 吃 f32）。
//   * 路由权重两种落点，由 launcher 的 `rw_in_operand` 选择（见 shim）：
//       rw_in_operand=0（**默认**）：epilogue 乘（`__fmul_rn`），残留 per-slot partial 后
//                                    交给既有 ascending `moe_down_reduce` ⇒ **与 SIMT 的
//                                    求和顺序逐项同构**（SIMT 融合核也是 fadd_rn(tot, fmul_rn(acc,rw))）；
//       rw_in_operand=1：gather 期把 rw 乘进操作数（= 官方 `x = weights * x` 之后再量化），
//                        epilogue 不再乘（= `DSV41_ROUTED_DOWN_QUANT` 的顺序；两者互斥）。
//   ⇒ 相对 SIMT 的唯一"算法差"就是 e4m3 量化本身（~1e-3 量级，见设计文档的对拍表）。
//
// ⚠️ 头文件包含责任：本文件被 `moe_bs_dn_shim.cu` `#include`（与 `moe_bs_handwritten.cu`
//    同约定），**不能单独编译**（缺 va_list 之外的 `uint32_t` 来自 <cstdint>，这里自带）。
//    编译口径：build.sh 的 `tilelang_gen/*_shim.cu` 分支（-O3，**不带** --use_fast_math：
//    tcgen05 inline-asm 必须无 fast-math，见 build.sh:175-184）。
// ⚠️ 所有辅助函数带 **`dn_` 唯一前缀**：本项目因多线合并时同名重复定义坏过一次完整构建
//    （AGENTS.md「合并纪律」第 4 条），本 TU 与 moe_bs_handwritten.cu 同处一个 .so。

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// =============================================================================
// §1 tcgen05 原语（逐字来自 moe_bs_handwritten.cu / tests_bs_impulse.cu 的 VERIFIED 集合；
//     只改名字加 `dn_` 前缀）
// =============================================================================

__device__ __forceinline__ void dn_tc_alloc(uint32_t* dst, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(dst)),
                 "r"(ncols));
}

__device__ __forceinline__ void dn_tc_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void dn_tc_dealloc(uint32_t tmem, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(ncols));
}

// 运行期开关（shim 从 env 读一次后 cudaMemcpyToSymbol；asm 必须编译期，两种拼写都编进来）
__device__ int dn_g_cpasync = 0;   // DSV41_MOE_DOWN_BS_CPASYNC（默认 1 = 双缓冲）
__device__ int dn_g_waitdbg = 0;   // DSV41_MOE_DOWN_BS_WAITDBG（默认 0）

// block-scaled mxf8f6f4 MMA。`.scale_vec::1X` **不加**（gate/up 臂的定谳：官方生成码没有它，
// 加了会破坏 eager —— 见 AGENTS.md「`DSV41_MOE_BS_SCALEVEC1X` 也须保持 OFF」）。
__device__ __forceinline__ void dn_tc_mma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                          uint32_t idesc, uint32_t sfa_tmem, uint32_t sfb_tmem,
                                          uint32_t enable_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t}"
        ::"r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem),
        "r"(enable_d)
        : "memory");
}

__device__ __forceinline__ void dn_tc_commit(void* mbar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
            (uint32_t)__cvta_generic_to_shared(mbar))
        : "memory");
}

// SMEM descriptor: start_addr[0:14) | lbo[16:30) | sbo[32:46) | version=1[46] | layout[61:64)
__device__ __forceinline__ uint64_t dn_make_desc(const void* smem_ptr, uint32_t lbo_16B,
                                                 uint32_t sbo_16B, uint32_t layout) {
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)(lbo_16B & 0x3FFF) << 16;
    d |= (uint64_t)(sbo_16B & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)(layout & 0x7) << 61;
    return d;
}

// IDESC: b_sf[4:6) | a_fmt[7:10) | b_fmt[10:13) | n_dim[17:23) | sf_fmt=UE8M0[23)
//        | m_dim[24:29) | a_sf[29:31)
__device__ __forceinline__ uint32_t dn_make_idesc(int m, int n, int a_fmt, int b_fmt, int sf_id) {
    uint32_t d = 0;
    d |= (uint32_t)(sf_id & 3) << 4;       // b_sf_id
    d |= (uint32_t)(a_fmt & 7) << 7;       // 0 = E4M3（激活）
    d |= (uint32_t)(b_fmt & 7) << 10;      // 5 = E2M1（权重）
    d |= (uint32_t)((n >> 3) & 63) << 17;  // n_dim = N/8
    d |= (uint32_t)1 << 23;                // scale_format = UE8M0
    d |= (uint32_t)((m >> 4) & 31) << 24;  // m_dim = M/16
    d |= (uint32_t)(sf_id & 3) << 29;      // a_sf_id
    return d;
}

// SF 转置（TileLang `tcgen05_sf_warp_transpose` 的逐字版：4x32 u32 块内转置）。
// 输入：SFA_s/B_s 形的 [128] u32（每行一个 group-major 字）；输出：`tcgen05.cp` 需要的布局。
__device__ __forceinline__ void dn_sf_transpose(uint32_t* smem_ptr) {
    const uint32_t lane = threadIdx.x % 32;
    uint32_t values[4];
    for (uint32_t i = 0; i < 4; ++i) values[i] = smem_ptr[(i ^ (lane >> 3)) * 32 + lane];
    __syncwarp();
    for (uint32_t i = 0; i < 4; ++i) smem_ptr[lane * 4 + (i ^ (lane >> 3))] = values[i];
}

// SF 拷贝到 TMEM（TileLang `tcgen05_cp` 的逐字版：32x128b.warpx4）
__device__ __forceinline__ void dn_tc_cp(uint64_t smem_desc, uint32_t tmem_col) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tmem_col),
                 "l"(smem_desc));
}

// SF smem descriptor（TileLang `make_sf_smem_desc` 的逐字版：sbo>>4 = 8）
__device__ __forceinline__ uint64_t dn_make_sf_desc(void* smem_ptr) {
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= (uint64_t)(addr >> 4) & 0x3FFF;
    desc |= (uint64_t)8u << 32;
    desc |= (uint64_t)1u << 46;
    return desc;
}

// =============================================================================
// §2 §47 定谳的 fp4 smem 布局（逐字）
// =============================================================================
// p = 行内 packed 字节下标 [0,64)；c = 16 B 容器下标 [0,8)。硬件只读每槽前 8 B。
__device__ __forceinline__ int dn_pack_sw128(int row, int p) {
    const int c = (p >> 3) & 7;
    return (row >> 3) * 1024 + (row & 7) * 128 + (((c ^ (row & 7)) & 7) << 4) + (p & 7);
}
// e4m3 侧（1 B/元素）的 SW128：addr(r,c) = (r/8)*1024 + (r%8)*128 + (((c/16) ^ (r%8))*16) + (c%16)
__device__ __forceinline__ int dn_sw128_16b(int m, int c) {
    return (m >> 3) * 1024 + (m & 7) * 128 + (((c ^ (m & 7)) & 7) << 4);
}

// =============================================================================
// §3 有界等待（安全属性，不设门 —— 与门/上臂 `whp_*` 同一契约）
// =============================================================================
// 无界自旋会让整个 rank 永卡（`[ar5-hang]` 洪水、零 step 行的那次事故）。这里：
// 正常路径（首次 try_wait 即真）只多一次比较 + 一个分支；超时则**必须出声**并放弃 TMEM。
#define DN_MMA_SPIN_CAP (1u << 22)
__device__ unsigned long long dn_wait_abort_n = 0ull;

__device__ __forceinline__ bool dn_mbar_probe(uint64_t* bar, uint32_t phase) {
    uint32_t done;
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P;\n\t}"
        : "=r"(done)
        : "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
    return done != 0u;
}

__device__ __noinline__ void dn_mma_timeout_report(uint64_t* bar, uint32_t phase, int sp, int seg,
                                                   int n_tile) {
    const unsigned long long nth = atomicAdd(&dn_wait_abort_n, 1ull) + 1ull;
    if (nth <= 8ull) {
        printf("[moe-bs-dn] TIMEOUT mma-arrive: block=(n_tile=%d,seg=%d) span=%d phase=%u "
               "spins>%u -> ABORT (bounded wait; the kernel bails out instead of spinning)\n",
               n_tile, seg, sp, phase, (unsigned)DN_MMA_SPIN_CAP);
    }
    if (dn_g_waitdbg) {
        printf("[moe-bs-dn-wait-dbg] nth=%llu span=%d expected_parity=%u other_parity_ready=%d "
               "clock=%lld abort_n=%llu\n",
               nth, sp, phase, dn_mbar_probe(bar, phase ^ 1u) ? 1 : 0, (long long)clock64(),
               dn_wait_abort_n);
    }
}

__device__ __forceinline__ bool dn_mma_wait_bounded(uint64_t* bar, uint32_t phase, int sp, int seg,
                                                    int n_tile) {
    for (uint32_t spins = 0;; ++spins) {
        if (dn_mbar_probe(bar, phase)) return false;
        if (spins >= DN_MMA_SPIN_CAP) {
            dn_mma_timeout_report(bar, phase, sp, seg, n_tile);
            return true;
        }
    }
}

// =============================================================================
// §4 cp.async（gate `DSV41_MOE_DOWN_BS_CPASYNC`，默认 ON —— down 是**流式**负载，
//      B 面每 CTA 只读 20 KB 却要等 HBM；不重叠则每个 span 都付一次全暴露延迟）
// =============================================================================
// 只用 sm_80 的 cp.async（**不是** TMA/cp.async.bulk）：无描述符、无 mbarrier、无多级流水
// 屏障对象，完成由 per-thread 的 commit_group/wait_group + 一次 __syncthreads 观察。
// 对齐：A 16 B（src = act + m*320 + sp*128 + c*16，320 % 16 == 0）、B 8 B
// （src = w2 + e*stride + row*160 + sp*64 + c*8，需池基址与 expert stride 是 8 的倍数）、
// SF 4 B。launcher 前置检查，不满足则**该操作数退回普通 LDG/STS**（同一批字节、同一批地址，
// 只是延迟暴露；数值零差别）。
__device__ __forceinline__ void dn_cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void dn_cp8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void dn_cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void dn_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void dn_cp_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// TMEM 读：32x32b.x32 —— 每 lane 从自己的 TMEM lane 读 32 个连续列。
// 分 4 次调用（共 128 列）而不是 tl 的 x128：**寄存器压力**是这里的硬约束
// （`__launch_bounds__(128,3)` ⇒ 170 regs/thread；x128 一次吃 128 个寄存器会 spill，
// 而 spill 的量 = 整个输出）。见文件尾的 epilogue 注释。
__device__ __forceinline__ void dn_tc_ld_x32(uint32_t taddr, uint32_t* v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
          "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]),
          "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]),
          "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]),
          "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
        : "r"(taddr));
}
__device__ __forceinline__ void dn_tc_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// =============================================================================
// §5 几何常量（down 冻结形状）
// =============================================================================
constexpr int DN_BM = 128;      // M tile = 段内 assignment 行（与 BS 臂的 BM 同值）
constexpr int DN_BN = 128;      // N tile = dim 的 128 列
constexpr int DN_KT = 128;      // 一个 K-span 的 K（= SW128 一行 footprint 的元素数）
constexpr int DN_SPANS = 3;     // K = 320 = 128 + 128 + 64
constexpr int DN_K = 320;       // down 的 contraction = inter_local
constexpr int DN_SEGCAP = 36;   // 与 BS 臂同值（SEG_CAP = VERIFY_ROWS(6) x TOPK_MAX(6)）
constexpr int DN_SPAN_BYTES = 16384;      // 128 行 x 128 B（SW128 footprint，e4m3 与 fp4 同）
constexpr int DN_SF_BYTES = DN_BM * 4;    // 128 个 group-major u32
// 单 stage = A(16384) + B(16384) + SFA(512) + SFB(512) = 33792
constexpr int DN_STAGE_BYTES = 2 * DN_SPAN_BYTES + 2 * DN_SF_BYTES;
// NS=1 -> 33800 B（含 mbar 8）；NS=2 -> 67592 B。3 CTA/SM：202776 B < 227 KiB ✓
constexpr size_t DN_SMEM_SINGLE = DN_STAGE_BYTES + 8;
constexpr size_t DN_SMEM_DOUBLE = 2 * DN_STAGE_BYTES + 8;

// =============================================================================
// §6 一个 K-span 的 staging（顺序版；cp.async 版见 dn_issue_span）
// =============================================================================
// 源地址/目的地址与 dn_issue_span **逐字节相同**，只是传输方式不同（与 gate/up 的
// hw_load_stage_seq / hw_issue_stage 对偶）。
//
//   A tile  : e4m3，128 行 x 128 B/span。src = A + (seg*BM+m)*k + sp*128 + c*16
//             dst = dn_sw128_16b(m,c)（16 B chunk 内字节连续）
//             ⚠️ 最后一个 span 只搬 64 B/行（K 256..319）⇒ 只做 c < 4。
//   B tile  : packed fp4，128 行 x (64/32) packed B。src = W2 + e*stride + row*k/2 + sp*64 + c*8
//             dst = dn_pack_sw128(row, c*8)（= 容器前 8 B）⇒ **8 B 粒度**，不能用 16 B：
//             16 B 拷贝会连着写槽内 8..15（硬件不读的）并破坏容器 swizzle。
//   SFA     : [spans][SEG*BM] u32，group-major（字 sp*SEG*BM + seg*BM + m，byte j = K-block 4sp+j）
//   SFB     : **直接读 w2 的原始 e8m0 面**（[dim 行][pitch] byte，pitch = 16 的 pad 面），
//             字 = *(u32*)(srow + 4*sp) —— **不需要装载期 repack 池**（w1/w3 需要，因为
//             它们的面内行距是 k/32 = 160 而这里 pitch 是 16 的 pad 面，四字节天然成字）。
//             ⚠️ 最后一个 span 的字节 2/3（K-block 10/11）落在 pad 区、**从不被 sf_id 选中**
//             ⇒ 不需要它们的值为 0（本 kernel 每个 span 只发 ki < nki 个 MMA）。
__device__ __forceinline__ void dn_load_span(int sp, const uint8_t* __restrict__ A,
                                             const uint32_t* __restrict__ SFA,
                                             const uint8_t* __restrict__ W2,
                                             const uint8_t* __restrict__ W2S, int dim_k,
                                             int seg, int n_tile, int e, int64_t w2_stride,
                                             int64_t w2s_stride, int w2s_pitch, int tid,
                                             uint8_t* A_s, uint8_t* B_s, uint32_t* SFA_s,
                                             uint32_t* SFB_s) {
    const int k = DN_K;
    const int kbytes = k >> 1;                       // 160
    const int abytes = (sp + 1) * DN_KT <= k ? DN_KT : (k - sp * DN_KT);  // 128 / 128 / 64
    const int bbytes = abytes >> 1;                  // 64 / 64 / 32
    const int ksc = k >> 5;                          // 10
    const int nsp = (ksc + 3) >> 2;                  // 3（= DN_SPANS）
    (void)dim_k;
    // (1) A tile —— 16 B chunk
    for (int i = tid; i < DN_BM * 8; i += 128) {
        const int m = i >> 3;
        const int c = i & 7;
        if (c * 16 >= abytes) continue;  // 末 span：只 c < 4
        const uint8_t* src =
            A + ((int64_t)(seg * DN_BM + m)) * (int64_t)k + (int64_t)sp * DN_KT + c * 16;
        const int dst = dn_sw128_16b(m, c);
        uint4 v;
        __builtin_memcpy(&v, src, 16);
        __builtin_memcpy(A_s + dst, &v, 16);
    }
    // (2) B tile —— 8 B chunk（packed fp4 的容器前 8 B）
    for (int i = tid; i < DN_BM * 8; i += 128) {
        const int n = i >> 3;
        const int c = i & 7;
        if (c * 8 >= bbytes) continue;  // 末 span：只 c < 4
        const uint8_t* src = W2 + (int64_t)e * w2_stride +
                             ((int64_t)(n_tile * DN_BN + n)) * (int64_t)kbytes +
                             (int64_t)sp * 64 + c * 8;
        const int dst = dn_pack_sw128(n, c * 8);
        uint2 v;
        __builtin_memcpy(&v, src, 8);
        __builtin_memcpy(B_s + dst, &v, 8);
    }
    // (3) SFA —— 每行一个字（gather 期 group-major 写好的 [spans][SEG*BM]）
    if (tid < DN_BM) {
        if (sp < nsp) {
            SFA_s[tid] = SFA[(int64_t)sp * (DN_SEGCAP * DN_BM) + seg * DN_BM + tid];
        } else {
            SFA_s[tid] = 0u;
        }
    }
    // (4) SFB —— 直接读原始 e8m0 面的 4 字节
    if (tid < DN_BM) {
        const uint8_t* srow = W2S + (int64_t)e * w2s_stride +
                              ((int64_t)(n_tile * DN_BN + tid)) * (int64_t)w2s_pitch +
                              (int64_t)4 * sp;
        SFB_s[tid] = (sp < nsp) ? *(const uint32_t*)srow : 0u;
    }
}

// cp.async 版：与 dn_load_span 的源/目的逐字节相同，只是把传输换成 cp.async
// （A 16 B .cg、B 8 B .ca、SF 4 B .ca），由调用方 commit/wait。
__device__ __forceinline__ void dn_issue_span(int sp, const uint8_t* __restrict__ A,
                                              const uint32_t* __restrict__ SFA,
                                              const uint8_t* __restrict__ W2,
                                              const uint8_t* __restrict__ W2S, int seg, int n_tile,
                                              int e, int64_t w2_stride, int64_t w2s_stride,
                                              int w2s_pitch, int tid, uint8_t* A_s, uint8_t* B_s,
                                              uint32_t* SFA_s, uint32_t* SFB_s, bool a_ok,
                                              bool b_ok, bool sf_ok) {
    const int k = DN_K;
    const int kbytes = k >> 1;
    const int abytes = (sp + 1) * DN_KT <= k ? DN_KT : (k - sp * DN_KT);
    const int bbytes = abytes >> 1;
    const int ksc = k >> 5;
    const int nsp = (ksc + 3) >> 2;
    if (a_ok) {
        for (int i = tid; i < DN_BM * 8; i += 128) {
            const int m = i >> 3;
            const int c = i & 7;
            if (c * 16 >= abytes) continue;
            const uint8_t* src =
                A + ((int64_t)(seg * DN_BM + m)) * (int64_t)k + (int64_t)sp * DN_KT + c * 16;
            dn_cp16(A_s + dn_sw128_16b(m, c), src);
        }
    } else {
        for (int i = tid; i < DN_BM * 8; i += 128) {
            const int m = i >> 3;
            const int c = i & 7;
            if (c * 16 >= abytes) continue;
            const uint8_t* src =
                A + ((int64_t)(seg * DN_BM + m)) * (int64_t)k + (int64_t)sp * DN_KT + c * 16;
            uint4 v;
            __builtin_memcpy(&v, src, 16);
            __builtin_memcpy(A_s + dn_sw128_16b(m, c), &v, 16);
        }
    }
    if (b_ok) {
        for (int i = tid; i < DN_BM * 8; i += 128) {
            const int n = i >> 3;
            const int c = i & 7;
            if (c * 8 >= bbytes) continue;
            const uint8_t* src = W2 + (int64_t)e * w2_stride +
                                 ((int64_t)(n_tile * DN_BN + n)) * (int64_t)kbytes +
                                 (int64_t)sp * 64 + c * 8;
            dn_cp8(B_s + dn_pack_sw128(n, c * 8), src);
        }
    } else {
        for (int i = tid; i < DN_BM * 8; i += 128) {
            const int n = i >> 3;
            const int c = i & 7;
            if (c * 8 >= bbytes) continue;
            const uint8_t* src = W2 + (int64_t)e * w2_stride +
                                 ((int64_t)(n_tile * DN_BN + n)) * (int64_t)kbytes +
                                 (int64_t)sp * 64 + c * 8;
            uint2 v;
            __builtin_memcpy(&v, src, 8);
            __builtin_memcpy(B_s + dn_pack_sw128(n, c * 8), &v, 8);
        }
    }
    if (tid < DN_BM) {
        if (sf_ok) {
            dn_cp4(SFA_s + tid,
                   SFA + (int64_t)(sp < nsp ? sp : 0) * (DN_SEGCAP * DN_BM) + seg * DN_BM + tid);
            const uint8_t* srow = W2S + (int64_t)e * w2s_stride +
                                  ((int64_t)(n_tile * DN_BN + tid)) * (int64_t)w2s_pitch +
                                  (int64_t)4 * sp;
            dn_cp4(SFB_s + tid, srow);
        } else {
            SFA_s[tid] = (sp < nsp) ? SFA[(int64_t)sp * (DN_SEGCAP * DN_BM) + seg * DN_BM + tid]
                                    : 0u;
            const uint8_t* srow = W2S + (int64_t)e * w2s_stride +
                                  ((int64_t)(n_tile * DN_BN + tid)) * (int64_t)w2s_pitch +
                                  (int64_t)4 * sp;
            SFB_s[tid] = (sp < nsp) ? *(const uint32_t*)srow : 0u;
        }
    }
}

// =============================================================================
// §7 down kernel 主体
// =============================================================================
// Grid: (dim/DN_BN, DN_SEGCAP) —— blockIdx.x = dim 上的 N-tile，blockIdx.y = 段（专家）
// Block: 128 线程（4 warps）：warp1 发 SF-copy + MMA，warp2 做 SF 转置，warp0/3 只参与
//        staging 与 epilogue（与 gate/up 手写核同分工；epilogue 需要全部 4 个 warp
//        的 128 lane 才能读完 128 行的 TMEM）。
//
// ⚠️ 输出：本 kernel **直接写 per-assignment partial**（不做 reduce、不做 atomic）：
//      out[Order[seg*BM+r] * dim + n_tile*BN + c] = (rw ? rw_r : 1) * C[r][c]
//    Order 的 pad 行 = -1 ⇒ 该行不写（A 侧 pad 行是 0、SF=0 ⇒ C 恒 0）。
//    不同段/不同 N-tile 的 (assignment, dim 列) 互不相交 ⇒ 无需 atomic、无需零填充，
//    与 SIMT 的 `expert_down_fp4_batched` 写出的 [slot][dim] partial 布局**逐格相同**
//    ⇒ 既有的 ascending `moe_down_reduce` 合并直接复用（数值契约不变）。
extern "C" __global__ void __launch_bounds__(128, 3) moe_bs_dn_kernel(
    const uint8_t* __restrict__ A,       // [SEG*BM][K] e4m3（gather+量化后的 down 输入）
    const uint32_t* __restrict__ SFA,    // [SPANS][SEG*BM] group-major u32
    const uint8_t* __restrict__ W2,      // u8 [E, dim, K/2]（expert 0 的面；e 在 base + e*w2_stride）
    const uint8_t* __restrict__ W2S,     // u8 [E, dim, w2s_pitch] e8m0 面
    const int* __restrict__ Eid,         // [SEG_CAP] i32
    const int* __restrict__ Order,       // [SEG_CAP*BM] i32（pad = -1）
    const float* __restrict__ RouteW,    // [rows*slots] f32 或 nullptr（= 不加权）
    float* __restrict__ Out,             // [rows*slots][dim] f32 partial（OVERWRITE）
    int dim, int rw_in_operand, int64_t w2_stride, int64_t w2s_stride, int w2s_pitch) {
    const int n_tile = blockIdx.x;
    const int seg = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int e = Eid[seg];

    const int NS = dn_g_cpasync ? 2 : 1;
    extern __shared__ __align__(1024) uint8_t dn_smem[];
    uint8_t* A_sh = dn_smem;
    uint8_t* B_sh = dn_smem + NS * DN_SPAN_BYTES;
    uint32_t* SFA_sh = (uint32_t*)(dn_smem + 2 * NS * DN_SPAN_BYTES);
    uint32_t* SFB_sh = SFA_sh + NS * DN_BM;
    uint64_t* mma_bar = (uint64_t*)(SFB_sh + NS * DN_BM);

    __shared__ __align__(16) uint dn_C_tmem;
    __shared__ __align__(16) uint dn_SF_tmem;
    __shared__ volatile int dn_wait_fail;
    if (warp == 0) {
        dn_tc_alloc(&dn_C_tmem, 128);
        dn_tc_alloc(&dn_SF_tmem, 32);
        dn_tc_relinquish();
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t C_tmem = dn_C_tmem;
    const uint32_t SF_tmem = dn_SF_tmem;

    if (tid == 0) {
        dn_wait_fail = 0;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
                         (uint32_t)__cvta_generic_to_shared(mma_bar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    const bool a_cp_ok = dn_g_cpasync && (((uintptr_t)A & 15u) == 0u);
    const bool b_cp_ok = dn_g_cpasync && ((((uintptr_t)W2) & 7u) == 0u) &&
                         (((uint64_t)w2_stride & 7u) == 0u);
    const bool sf_cp_ok = dn_g_cpasync && ((((uintptr_t)SFA) & 3u) == 0u) &&
                          ((((uintptr_t)W2S) & 3u) == 0u) &&
                          (((uint64_t)w2s_stride & 3u) == 0u) && ((w2s_pitch & 3) == 0);

    if (dn_g_cpasync) {
        dn_issue_span(0, A, SFA, W2, W2S, seg, n_tile, e, w2_stride, w2s_stride, w2s_pitch, tid,
                      A_sh, B_sh, SFA_sh, SFB_sh, a_cp_ok, b_cp_ok, sf_cp_ok);
        dn_cp_commit();
    }

    for (int sp = 0; sp < DN_SPANS; ++sp) {
        const int s = dn_g_cpasync ? (sp & 1) : 0;
        uint8_t* A_s = A_sh + (size_t)s * DN_SPAN_BYTES;
        uint8_t* B_s = B_sh + (size_t)s * DN_SPAN_BYTES;
        uint32_t* SFA_s = SFA_sh + s * DN_BM;
        uint32_t* SFB_s = SFB_sh + s * DN_BM;

        if (dn_g_cpasync) {
            if (sp + 1 < DN_SPANS) {
                uint8_t* A_n = A_sh + (size_t)(1 - s) * DN_SPAN_BYTES;
                uint8_t* B_n = B_sh + (size_t)(1 - s) * DN_SPAN_BYTES;
                dn_issue_span(sp + 1, A, SFA, W2, W2S, seg, n_tile, e, w2_stride, w2s_stride,
                              w2s_pitch, tid, A_n, B_n, SFA_sh + (1 - s) * DN_BM,
                              SFB_sh + (1 - s) * DN_BM, a_cp_ok, b_cp_ok, sf_cp_ok);
                dn_cp_commit();
                dn_cp_wait<1>();
            } else {
                dn_cp_wait<0>();
            }
        } else {
            dn_load_span(sp, A, SFA, W2, W2S, dim, seg, n_tile, e, w2_stride, w2s_stride,
                         w2s_pitch, tid, A_s, B_s, SFA_s, SFB_s);
        }
        __syncthreads();

        // SF 转置：smem 内 [128] u32 → tcgen05.cp 需要的布局（warp2）
        if (warp == 2) {
            dn_sf_transpose(SFA_s);
            dn_sf_transpose(SFB_s);
        }
        __syncthreads();

        // 通用写 → async proxy 可见（缺它会读到上一 span 的 stale smem；均匀测试数据看不出来）
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

        if (warp == 1) {
            if (lane == 0) {
                dn_tc_cp(dn_make_sf_desc(SFA_s), SF_tmem + 0);
                dn_tc_cp(dn_make_sf_desc(SFB_s), SF_tmem + 4);
            }
            // 末 span 只有 2 个 K-block（K 256..319）
            const int nki = (sp == DN_SPANS - 1) ? 2 : 4;
            for (int ki = 0; ki < nki; ++ki) {
                const uint32_t idesc = dn_make_idesc(DN_BM, DN_BN, 0 /*E4M3*/, 5 /*E2M1*/, ki);
                const uint32_t enable_d = (sp == 0 && ki == 0) ? 0u : 1u;
                const uint64_t a_desc = dn_make_desc(A_s, 1, 64, 2) + (uint64_t)(ki * 2);
                const uint64_t b_desc = dn_make_desc(B_s, 1, 64, 2) + (uint64_t)(ki * 2);
                if (lane == 0)
                    dn_tc_mma(C_tmem, a_desc, b_desc, idesc, SF_tmem + 0, SF_tmem + 4, enable_d);
            }
            if (lane == 0) dn_tc_commit(mma_bar);
        }
        if (tid == 0) {
            if (dn_mma_wait_bounded(mma_bar, (uint32_t)(sp & 1), sp, seg, n_tile)) dn_wait_fail = 1;
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (dn_wait_fail) break;
    }

    if (dn_wait_fail) {
        __syncthreads();
        if (warp == 0) {
            dn_tc_dealloc(C_tmem, 128);
            dn_tc_dealloc(SF_tmem, 32);
        }
        return;
    }

    // ---- epilogue: TMEM → registers → 全局散写 ----
    // 分 4 批 32 列（寄存器压力；见 dn_tc_ld_x32 的注释）。每批：读 32 列 → 写 32 个 f32
    // （8 个 float4，每行 512 B 连续，4 个满 sector）。
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    {
        const int r = warp * 32 + lane;  // 段内行 = 该 gate/up 臂的 BM 行号
        const int idx = Order[seg * DN_BM + r];
        if (idx >= 0) {
            const float rwv = (RouteW != nullptr && !rw_in_operand)
                                  ? __fmul_rn(1.0f, RouteW[idx])
                                  : 1.0f;
            float* dst = Out + (size_t)idx * (size_t)dim + (size_t)n_tile * DN_BN;
            for (int c0 = 0; c0 < DN_BN; c0 += 32) {
                uint32_t v[32];
                dn_tc_ld_x32(C_tmem + (uint32_t)c0, v);
                dn_tc_wait_ld();
                float f[32];
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const float cv = __uint_as_float(v[i]);
                    f[i] = (rwv == 1.0f) ? cv : __fmul_rn(cv, rwv);
                }
#pragma unroll
                for (int i = 0; i < 32; i += 4) {
                    float4 q;
                    q.x = f[i + 0];
                    q.y = f[i + 1];
                    q.z = f[i + 2];
                    q.w = f[i + 3];
                    *reinterpret_cast<float4*>(dst + c0 + i) = q;
                }
            }
        } else {
            // pad 行：仍必须把 TMEM 读掉（tcgen05.ld 是 warp 级同步操作，不能分叉跳过）
            for (int c0 = 0; c0 < DN_BN; c0 += 32) {
                uint32_t v[32];
                dn_tc_ld_x32(C_tmem + (uint32_t)c0, v);
                dn_tc_wait_ld();
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        dn_tc_dealloc(C_tmem, 128);
        dn_tc_dealloc(SF_tmem, 32);
    }
}
