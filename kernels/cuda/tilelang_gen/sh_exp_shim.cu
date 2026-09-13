// sh_exp_shim.cu — shared expert 的 TileLang fp8 MMA launcher shim
//                 （ferrite 生产链接线件）。
//
// 上位设计：docs/agent/c5-sh-exp-tilelang-design.md（C5 shared expert TileLang）。
// 生成物：`sh_exp_gu_tl.cu`（device：gate+up → swiglu+limit → fp8 quant）
//         `sh_exp_dn_tl.cu`（device：w2 down GEMM + epi_add）
//         `sh_exp_tl_config.txt`（冻结几何 + 签名），由 kernels/tilelang/gen_sh_exp_aot.py 产出。
// 同族先例：tilelang_gen/wkv_shim.cu（第一阶段裸指针臂）、moe_bs_shim.cu（TMA 臂）。
//
// =============================================================================
// 它替代的是什么
// =============================================================================
// `shared_expert_mrows` 的 per-row 兜底链 / `dsv41_gemm_fp8_sh_exp_fused`（1024 线程 +
// grid barrier + phase-1 的 **M× 权重重读**）。本臂把 shared expert 的
//     (w1·w3 → swiglu → fp8 → w2)
// 换成 **TileLang fp8 MMA 双发**：M 行激活**共享同一份权重读**（每层 w1/w3/w2 各读一次），
// 且**没有 grid barrier**（两发顺序发射，L2 直接从全局读 L1 输出的 aq/aqsc——同一次
// launch 内的 phase 之分被两个独立 launch 取代，前者需要 barrier 的地方在这里不存在）。
//
// =============================================================================
// ⚠️ ABI 是**裸指针**，不是 moe_bs 的 TMA 描述符 —— 不要照抄 moe_bs_shim.cu
// =============================================================================
// 两条臂的 ABI 不同，且**必须是**不同的：
//   * moe_bs 的 B operand 是 **packed fp4**（e2m1），它的 smem 必须是
//     `float4_e2m1_unpacked`，而 packed-global → unpacked-smem 这条「展开」**只有 TMA
//     的 tensor 形式能做**（gen_moe_bs_aot.py 文件头 §0.4）⇒ 那个臂**必须**走描述符，
//     host 必须 `cuTensorMapEncodeTiled`、必须 dlopen libcuda、必须有常驻 scratch。
//   * 本臂 A 与 B **都是 fp8 e4m3**（1 B/元素，无 sub-byte 展开）⇒ TMA 在这里**没有**
//     它存在的唯一理由。生成物用的是本树**已认证**的 fp8 MMA 配方
//     （`TL_DISABLE_TMA_LOWER` + `TL_DISABLE_WARP_SPECIALIZED`，见
//     gen_proj_shapes_aot.py 五形状 / 设计 R6 的防 illegal-instruction 措施）。
// ⇒ 本文件**没有** `cuTensorMapEncodeTiled`、**没有** dlopen、**没有**常驻 scratch、
//   **没有** `cudaFuncSetAttribute`（两份动态 smem 分别只有 7 680 B / 4 608 B，都在
//   48 KiB 默认上限之下）、**没有** INIT 期的任何动作。launch 直接吃调用方的指针。
//   ⚠️ 生成物的 fp8 形参类型是 tl_templates 的 `fp8_e4_t` / `fp8_e8_t`（1 字节位模式），
//      ferrite 的 ABI 是裸 `uint8_t*` ⇒ launch 处逐一 `reinterpret_cast`（照
//      wkv_shim.cu:226-228 的先例）。
//
// =============================================================================
// 导出符号
// =============================================================================
//   int dsv41_sh_exp_tilelang_gu(...)   -- L1：gate+up → swiglu+limit → fp8 quant
//   int dsv41_sh_exp_tilelang_dn(...)   -- L2：w2 down GEMM（可选 epi_add 折叠）
//   int dsv41_sh_exp_tilelang(...)      -- 双发组合（设计 §④b 的单入口：先验后发，
//                                           任一门不过则**一发都不发**）
// 三个入口的 rc 契约完全一致（照 wkv / moe_bs / bf16 shim 的先例）：
//   * `0` = 已发射；
//   * `2` = DECLINED（形状/pitch/对齐不接受）——**永不返回 1**（1 是
//           `cudaErrorInvalidValue`，与真实发射失败不可区分）；
//   * 其它非 0 = `cudaGetLastError()` 的真实错误码，由 Rust 的 `kerr` 报出。
// ⇒ Rust 只在 `rc == 2` 回退老路径，其它非 0 一律当错误。
//
// =============================================================================
// 形状域（冻结，来自 sh_exp_tl_config.txt）
// =============================================================================
//   m ∈ [1,8] ∧ n1(inter/sh_il) == 288 ∧ k1(dim) == 5120
//            ∧ n2 == 5120 ∧ k2 == 288 ∧ out_stride == 5120
//            ∧ aq_stride == 288 ∧ aqsc_stride == 9
//            ∧ w2sc_pitch == 9            (★ 硬约束，见下)
// 其余一律 decline，由调用方回退 `gemm_fp8_sh_exp_fused` / per-row 兜底。
//
// =============================================================================
// ★ w2.scale 的 9 B 行 pitch 是硬约束（设计 R3）
// =============================================================================
// ferrite 的 `w2.scale` 面是 `[160, 9]` ue8m0，**行 pitch = 9 B（未 pad）**
// （`weights.rs:355-356` 的 `Shard::Cols`；`sf_pitch_plane` 只匹配 `ExpertCols`，所以
// `plan_pitch` 对它是恒等）。生成物把 pitch=9 bake 进索引（`W2S: (160, 9)`），所以
// **一旦池侧未来把 `sf_pitch_plane` 扩到 `Shard::Cols`（行距变成 16 B），生成物就会
// 静默读错行** —— 那正是 `tcgen05-rank7-verdict.md §10` 那一类静默错值。
// ⇒ `w2sc_pitch` 是本 shim 的**运行期门**：调用方必须把**它实际看到的** pitch 传进来，
//   `!= 9` 立刻 decline（绝不建一个会错读的 launch）。调用点传 `sh_il/32`（chain_dev.rs）。
//
// =============================================================================
// 数值契约（设计 R4）
// =============================================================================
// 不追求与 SIMT 逐位（张量核的块内求和顺序是硬件定义的），但要求：
//   1. **(b′) 双侧同换**：m=1 与 m≤8 跑**同一程序**（生成物的 M 恒为 16，m 是运行期
//      谓词）⇒ row r of an M-row launch == row r of the M=1 launch OF THIS PROGRAM；
//   2. KS / 任何 split 因子不是 m 的函数（本 shim 的 KS 由生成期烘死）；
//   3. epilogue 逐字照抄 `swiglu_limit_q_kernel` 的 clamp/silu/amax/round（在生成物里，
//      见 gen_sh_exp_aot.py 的 `_swiglu` / `_fast_round_scale`）。
// ⚠️ 取这个臂必须**双侧同时**（eager + verify），否则就是 proj-mma 的 WOB 死法。

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

// 生成物：改名后 include（TileLang 把每个 kernel 都叫 main_kernel）。
// KS==1（v1 认证档）：L1 是 fused；KS>1 的备用档另有 partial+reduce 两份 dump
// （设计 §③），本 shim 的 v1 只发射 KS==1 的双发序列 —— 见文件末的说明。
#if __has_include("sh_exp_gu_tl.cu") && __has_include("sh_exp_dn_tl.cu")
#define main_kernel sh_exp_tl_gu_kernel
#include "sh_exp_gu_tl.cu"
#undef main_kernel
#define main_kernel sh_exp_tl_dn_kernel
#include "sh_exp_dn_tl.cu"
#undef main_kernel
#else
// [capture-guard-landing fix] The AOT artifacts are not generated yet (the peer's
// gen_sh_exp_aot.py output lands separately) — without this guard the build.sh
// wildcard compile of tilelang_gen/*_shim.cu fails on the missing include. Same
// pattern as moe_bs_shim.cu's FERRITE_MOE_BS_TL_MISSING.
#define FERRITE_SH_EXP_TL_MISSING 1
#endif

#ifndef FERRITE_SH_EXP_TL_MISSING

// ===========================================================================
// §0 ABI 形态自检（**编译期**，第一道闸）
// ===========================================================================
// 本臂的全部 host 逻辑都建立在「dump 是**裸指针** ABI」这个事实上。若 lowering 变成了
// TMA 描述符形态（有人把生成物的 `TL_DISABLE_TMA_LOWER` 拿掉），本文件的 launch 会以
// `const uint8_t*` 去撞 `__grid_constant__ const CUtensorMap` —— 那是编译错误，但报错点
// 在 launch 而不是这里，读起来像「参数写错」。所以先在这里钉死：**param 0 必须是指针**。
template <class F>
struct FnFirst;
template <class R, class A0, class... An>
struct FnFirst<R (*)(A0, An...)> {
    using type = A0;
};
static_assert(
    std::is_pointer_v<std::remove_cv_t<std::remove_reference_t<
        typename FnFirst<decltype(&sh_exp_tl_gu_kernel)>::type>>>,
    "the AOT dump is NOT the raw-pointer ABI (param 0 is not a pointer). Either the "
    "generator's pass_configs changed (TL_DISABLE_TMA_LOWER must stay ON for this arm: "
    "A and B are both fp8 e4m3, no sub-byte unpacking -> TMA is unnecessary and the shim "
    "below does NOT build descriptors) or the wrong dump was included. Re-derive from "
    "kernels/tilelang/gen_sh_exp_aot.py + docs/agent/c5-sh-exp-tilelang-design.md §4a/§4b.");

namespace {

// ===========================================================================
// §1 冻结几何 —— 逐项来自 tilelang_gen/sh_exp_tl_config.txt（生成器写出，勿手改）
// ===========================================================================
constexpr int kN1 = 288;      // sh_il（TP8 inter，未 pad）= w1/w3 的行数 = L1 的 N
constexpr int kK1 = 5120;     // dim = w1/w3 的 K = L1 的 K
constexpr int kN2 = 5120;     // dim = w2 的行数 = L2 的 N
constexpr int kK2 = 288;      // sh_il = w2 的 K = L2 的 K
constexpr int kOS2 = 5120;    // L2 输出行 stride == dim（生成物 bake）
constexpr int kBm = 16;       // mma M 原子（m ≤ 8 走运行期谓词）
constexpr int kBn = 32;       // N-tile == 一个 swiglu scale 块
constexpr int kThreads = 128;
constexpr int kMRows = 8;     // m 上界（VERIFY_ROWS）
constexpr int kGridGu = kN1 / kBn;   // 9
constexpr int kGridDn = kN2 / kBn;   // 160
// 动态 smem（生成物实际用量；< 48 KiB ⇒ 不需要 SetAttribute，见文件头）
constexpr int kSmemGu = 3 * (kBm * 32 + 2 * kBn * 32);  // NS * (A_sh + WG_sh + WU_sh)
constexpr int kSmemDn = 3 * (kBm * 32 + kBn * 32);      // NS * (A_sh + W_sh)
constexpr int kW2ScPitch = kK2 / 32;                    // 9 —— w2.scale 的行 pitch

// ===========================================================================
// §2 形状门 —— 全部通过才允许发射（部分发射是被禁止的）
// ===========================================================================
bool gu_ok(const uint8_t* a, const float* a_scale, const uint8_t* wg, const uint8_t* wg_scale,
           const uint8_t* wu, const uint8_t* wu_scale, int n1, int k1, int m, const uint8_t* aq,
           const float* aqsc, int aq_stride, int aqsc_stride) {
    if (m < 1 || m > kMRows) return false;
    if (n1 != kN1 || k1 != kK1) return false;
    // 生成物把 aq/aqsc 的行距 bake 成 N1 / N1/32 —— 不等就是写错位置（静默），必须 decline。
    if (aq_stride != kN1 || aqsc_stride != kN1 / 32) return false;
    if (a == nullptr || a_scale == nullptr || wg == nullptr || wg_scale == nullptr ||
        wu == nullptr || wu_scale == nullptr || aq == nullptr || aqsc == nullptr)
        return false;
    // 对齐：生成物的 A/W staging 是 16B 组（cp.async 16B），aq 的行距 288 = 16*18
    // ⇒ 基址 16B 对齐即可；aqsc 只做 4B 标量读写 ⇒ 4B 对齐。
    if ((((uintptr_t)a & 0xF) != 0) || (((uintptr_t)a_scale & 0xF) != 0) ||
        (((uintptr_t)wg & 0xF) != 0) || (((uintptr_t)wg_scale & 0xF) != 0) ||
        (((uintptr_t)wu & 0xF) != 0) || (((uintptr_t)wu_scale & 0xF) != 0) ||
        (((uintptr_t)aq & 0xF) != 0) || (((uintptr_t)aqsc & 0x3) != 0))
        return false;
    return true;
}

bool dn_ok(const uint8_t* aq, const float* aqsc, const uint8_t* w2, const uint8_t* w2_scale,
           int n2, int k2, int w2sc_pitch, int aq_stride, int aqsc_stride, int out_stride, int m,
           const float* out) {
    if (m < 1 || m > kMRows) return false;
    if (n2 != kN2 || k2 != kK2) return false;
    // ★ 硬约束（文件头）：w2.scale 是 [160,9]，行 pitch = 9 B。!= 9 立刻 decline。
    if (w2sc_pitch != kW2ScPitch) return false;
    // 生成物把 aq/aqsc 的行距（= L1 的输出）与 out 的行 stride 都 bake 死了。
    if (aq_stride != kN1 || aqsc_stride != kN1 / 32) return false;
    if (out_stride != kOS2) return false;
    if (aq == nullptr || aqsc == nullptr || w2 == nullptr || w2_scale == nullptr || out == nullptr)
        return false;
    if ((((uintptr_t)aq & 0xF) != 0) || (((uintptr_t)aqsc & 0x3) != 0) ||
        (((uintptr_t)w2 & 0xF) != 0) || (((uintptr_t)w2_scale & 0x3) != 0) ||
        (((uintptr_t)out & 0xF) != 0))
        return false;
    return true;
}

// ===========================================================================
// §3 发射（两条：L1 的 grid=(9,)，L2 的 grid=(160,)；顺序发射，无 barrier）
// ===========================================================================
cudaError_t launch_gu(const uint8_t* a, const float* a_scale, const uint8_t* wg,
                      const uint8_t* wg_scale, const uint8_t* wu, const uint8_t* wu_scale,
                      float limit, uint8_t* aq, float* aqsc, int m, cudaStream_t s) {
    // 形参序/类型 = sh_exp_gu_tl.cu 的 main_kernel 签名（config 文件里有权威行）。
    sh_exp_tl_gu_kernel<<<dim3((unsigned)kGridGu), kThreads, kSmemGu, s>>>(
        reinterpret_cast<const fp8_e4_t*>(a), a_scale, reinterpret_cast<const fp8_e4_t*>(wg),
        reinterpret_cast<const fp8_e8_t*>(wg_scale), reinterpret_cast<const fp8_e4_t*>(wu),
        reinterpret_cast<const fp8_e8_t*>(wu_scale), reinterpret_cast<fp8_e4_t*>(aq), aqsc, limit,
        m);
    return cudaGetLastError();
}

cudaError_t launch_dn(const uint8_t* aq, const float* aqsc, const uint8_t* w2,
                      const uint8_t* w2_scale, float* out, int epi_add, int m, cudaStream_t s) {
    // 形参序/类型 = sh_exp_dn_tl.cu 的 main_kernel 签名。
    sh_exp_tl_dn_kernel<<<dim3((unsigned)kGridDn), kThreads, kSmemDn, s>>>(
        reinterpret_cast<const fp8_e4_t*>(aq), aqsc, reinterpret_cast<const fp8_e4_t*>(w2),
        reinterpret_cast<const fp8_e8_t*>(w2_scale), out, epi_add, m);
    return cudaGetLastError();
}

void armed_note(int m, int split) {
    static int reported = 0;
    if (reported++ == 0)
        fprintf(stderr,
                "[sh-exp-tilelang] ARMED (split=%d) m=%d n1=%d k1=%d n2=%d k2=%d -> "
                "gu grid=(%d,)x%d smem=%d | dn grid=(%d,)x%d smem=%d\n",
                split, m, kN1, kK1, kN2, kK2, kGridGu, kThreads, kSmemGu, kGridDn, kThreads,
                kSmemDn);
}

}  // namespace

// ===========================================================================
// §4 导出符号 1：L1 —— gate+up → swiglu+limit → fp8 quant
// ===========================================================================
// a/a_scale = 块级 fp8 激活（xq [m,dim] e4m3 + xsc [m,dim/32] f32）；
// wg/wg_scale = gate 面（w1）[288,5120] fp8 + [9,160] ue8m0（行 pitch 160 B）；
// wu/wu_scale = up 面（w3）同形；
// aq/aqsc = L1 输出（[m,288] fp8 行距 288 + [m,9] f32 行距 9），即 L2 的输入。
extern "C" int dsv41_sh_exp_tilelang_gu(const uint8_t* a, const float* a_scale, const uint8_t* wg,
                                        const uint8_t* wg_scale, const uint8_t* wu,
                                        const uint8_t* wu_scale, float limit, int n1, int k1, int m,
                                        uint8_t* aq, float* aqsc, int aq_stride, int aqsc_stride,
                                        cudaStream_t s) {
    if (!gu_ok(a, a_scale, wg, wg_scale, wu, wu_scale, n1, k1, m, aq, aqsc, aq_stride,
               aqsc_stride))
        return 2;
    armed_note(m, 1);
    const cudaError_t e =
        launch_gu(a, a_scale, wg, wg_scale, wu, wu_scale, limit, aq, aqsc, m, s);
    return (int)e;
}

// ===========================================================================
// §5 导出符号 2：L2 —— w2 down GEMM（+ 可选 epi_add）
// ===========================================================================
// aq/aqsc = L1 的输出（行距 288 / 9）；w2/w2_scale = down 面 [5120,288] fp8 +
// [160,9] ue8m0（**行 pitch 9 B**，w2sc_pitch 门）；out = [m,dim] f32，行 stride = dim；
// epi_add != 0 ⇒ out[i,j] = out[i,j] + C（与原 `add_inplace_raw(out, sh_out)` 的操作数
// 对相同，逐元素同一次加法）。
extern "C" int dsv41_sh_exp_tilelang_dn(const uint8_t* aq, const float* aqsc, const uint8_t* w2,
                                        const uint8_t* w2_scale, int n2, int k2, int w2sc_pitch,
                                        int aq_stride, int aqsc_stride, int out_stride, int epi_add,
                                        float* out, int m, cudaStream_t s) {
    if (!dn_ok(aq, aqsc, w2, w2_scale, n2, k2, w2sc_pitch, aq_stride, aqsc_stride, out_stride, m,
               out))
        return 2;
    armed_note(m, 2);
    const cudaError_t e = launch_dn(aq, aqsc, w2, w2_scale, out, epi_add, m, s);
    return (int)e;
}

// ===========================================================================
// §6 导出符号 3：双发组合（设计 §④b 的单入口）
// ===========================================================================
// 与 `dsv41_gemm_fp8_sh_exp_fused` **同形 + 一个 `w2sc_pitch`**，所以 Rust 侧的臂体
// 就是把那次调用换成这一次（设计 §⑤d）。**先验后发**：L1 与 L2 的**两个**门都过才
// 发射第一发；任一门不过 ⇒ 返回 2 并且**一发都不发**（调用方回退老路，`moe_out_r`
// 从未被触碰 —— L2 的 epilogue 是唯一写它的地方，L1 只写 aq/aqsc 这两个 scratch）。
// `fold_r` 是 ABI 占位（本臂的「fold」由 M-tile 内建，与 m 无关）；`act` 必须是
// nullptr（生成物没有 act 通路，绝不静默丢输出）。
extern "C" int dsv41_sh_exp_tilelang(const uint8_t* a, const float* a_scale, const uint8_t* wg,
                                     const uint8_t* wg_scale, const uint8_t* wu,
                                     const uint8_t* wu_scale, float limit, int n1, int k1, int m,
                                     int fold_r, float* act, int act_stride, uint8_t* aq,
                                     float* aqsc, int aq_stride, int aqsc_stride, const uint8_t* w2,
                                     const uint8_t* w2_scale, int n2, int w2sc_pitch,
                                     int out_stride, int epi_add, float* out, cudaStream_t s) {
    (void)fold_r;
    (void)act_stride;
    if (act != nullptr) return 2;  // 生成物没有 act 通路
    const int k2 = kK2;
    if (!gu_ok(a, a_scale, wg, wg_scale, wu, wu_scale, n1, k1, m, aq, aqsc, aq_stride,
               aqsc_stride) ||
        !dn_ok(aq, aqsc, w2, w2_scale, n2, k2, w2sc_pitch, aq_stride, aqsc_stride, out_stride, m,
               out))
        return 2;
    armed_note(m, 3);
    cudaError_t e = launch_gu(a, a_scale, wg, wg_scale, wu, wu_scale, limit, aq, aqsc, m, s);
    if (e != cudaSuccess) return (int)e;
    e = launch_dn(aq, aqsc, w2, w2_scale, out, epi_add, m, s);
    return (int)e;
}

#endif  // !FERRITE_SH_EXP_TL_MISSING
