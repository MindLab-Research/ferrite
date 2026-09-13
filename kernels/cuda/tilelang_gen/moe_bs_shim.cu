// moe_bs_shim.cu — tcgen05 BLOCK-SCALED mxfp4 (e2m1 + ue8m0) MoE-up 的 launcher shim
//                 （ferrite 生产链接线件）。
//
// 上位：docs/agent/tcgen05-blockscaled-proto.md（原型全部 GPU 结论）
//       docs/agent/tilelang-moe-bs-wiring.md （映射 / SF pack / Rust patch / 验证手册）
// 生成物：`moe_bs_up_tl.cu`（device）+ `moe_bs_up_tl_host.cu`（**CUtensorMap 权威配方**）
//         + `moe_bs_tl_config.txt`（冻结几何），由 kernels/tilelang/gen_moe_bs_aot.py 产出。
// 同族先例：tilelang_gen/moe_bf16_shim.cu（bf16 臂）、tilelang_gen/wkv_shim.cu（第一阶段）。
//
// =============================================================================
// 它替代的是什么
// =============================================================================
// `moe_rows` / `moe` 的 routed gate/up 那一步（`dsv41_expert_gate_up_fp4_batched`，SIMT
// FMA-issue bound，36 sweep）。本 shim 用 **B300 原生 block-scaled MMA 直接吃 fp4**：
// e2m1 数据 + ue8m0 标度进 tensor core，**零 dequant、零 bf16 副本**。
//   * 与 bf16 臂（`dsv41_moe_tilelang_gate_up_bf16`）的关系：**互斥**。bf16 臂的前提是
//     装载期把专家权重展开成 bf16 常驻（`DSV41_MOE_BF16_DEQUANT`，
//     PROVENANCE §7.5 实测 **+105~113 GiB/rank**）；本臂读原生 fp4 池，那笔显存**不花**。
//     Rust 侧的 `DSV41_MOE_TILELANG_BS` 与 `DSV41_MOE_TILELANG` 互斥（见 wiring §5）。
//   * 原型同刻实测：blockscaled **72.6µs** vs bf16 grouped 116.8µs（contended 箱，
//     折算静默 ≈ 28µs）⇒ **0.62×**，逐位精确（max|err| = 0.00000）。
//
// =============================================================================
// 导出符号
// =============================================================================
//   int dsv41_moe_tilelang_gate_up_bs(...)   -- up（gate‖up）block-scaled grouped GEMM
//                                               + gather + scatter（RAW gate‖up 布局）
//   int dsv41_moe_bs_pack_wsf(...)           -- **装载期**：w1/w3 的 ue8m0 面
//                                               row-major [NP, K/32] -> group-major
//                                               packed uint32 [sf_words*NP]（每 expert）
//
// rc 契约（与 wkv / bf16 shim / proj_mma 完全一致）：
//   * `0` = 已发射；
//   * `2` = DECLINED（形状/模式不接受，或 INIT 失败）——**永不返回 1**（1 是
//           `cudaErrorInvalidValue`，与真实发射失败不可区分）；
//   * 其它非 0 = `cudaGetLastError()` 的真实错误码，由 Rust 的 `kerr` 报出。
// ⇒ Rust 只在 `rc == 2` 回退老路径，其它非 0 一律当错误。
//
// =============================================================================
// 四条硬约束
// =============================================================================
//  1. **TMA 描述符 ABI**：TileLang 默认 lowering 把参与 TMA 搬运的 6 个操作数变成
//     `__grid_constant__ const CUtensorMap` 形参（A/W1/W3/SFA/SFW1/SFW3），C 是 TMA
//     store，也是描述符。**host 必须 `cuTensorMapEncodeTiled`**。裸指针 ABI 在这条路上
//     **走不通**：blockscaled 的 A/B smem 必须是 `float4_e2m1_unpacked`（packed smem
//     静默错值 err=3.0，原型 §4.1），而 packed-global → unpacked-smem 只有 TMA 的
//     tensor 形式能做（`copy_analysis.cc:539`）。
//     ⚠️ **build.sh 不链 `-lcuda`** ⇒ 本文件**不直接引用** driver 符号，而是 `dlopen`
//     `libcuda.so.1` + `dlsym("cuTensorMapEncodeTiled")`。这保持了「编译期无条件编译
//     进来、运行期门控」的既有契约（build.sh 一行不改、BUILD_ID 不变）。
//  2. `cudaFuncSetAttribute` / `cudaMalloc` / `dlopen` 只在 **INIT 期**（capture 内调用
//     会让 `cudaStreamEndCapture` 失败）。先例 `dsv41_experts_mxf4.cu:4137`。本 shim 是
//     懒初始化：第一次调用做一次。SetAttribute 是**必要条件**（动态 smem ≫ 48 KiB 默认
//     上限），INIT 失败一律 decline（返回 2），绝不半发射。
//  3. `<<<..., smem, s>>>` 用调用方传入的 CuStream —— 必须进 capture / 与主流同序。
//  4. 形状不合规 `return 2`（不是 1 / 负数）。
//
// =============================================================================
// 布局契约（生成物 bake 死的东西，host 必须对齐）
// =============================================================================
//   A   : [SEG_CAP*BM, K/2] u8   -- **已 gather + 每段 pad 到 BM 行**的 fp4 激活。
//                                  e2m1 打包规则与 ferrite 一致：低 4 位 = 偶 k。
//                                  **pad 行必须真的全 0**（内核不做 mask）。
//   W1  : [E, NP, K/2]      u8   -- gate 面（ferrite 池里的 w1.weight，K 连续）
//   W3  : [E, NP, K/2]      u8   -- up 面（w3.weight）
//   SFW1: [E, sf_words*NP]  u32  -- 装载期 pack（group-major）
//   SFW3: [E, sf_words*NP]  u32
//   SFA : [sf_words*SEG_CAP*BM] u32 -- 每次调用 pack（group-major）
//   Eid : [SEG_CAP]         i32  -- 每段的 expert id（host 数组，上行到常驻 scratch）
//   C   : [SEG_CAP*BM, 2*NP] f32 -- **列序交错**：tile bx 的第 j 列，j<NH 是 gate 列
//                                  bx*NH+j，j>=NH 是 up 列 bx*NH+(j-NH)。见 wiring §2.3。
//
// **内核不做 gather / mask / atomic**（原型 §2.2）：gather/scatter 在本文件里完成。
// moe_align（host，chain_dev.rs::moe_align_host）产出被本 shim 消费的三张表：
//   order[SEG_CAP*BM] : i32  -- 该 (段, 段内行) 的 assignment 下标（flat = row*topk+slot），
//                               pad 行 = -1
//   counts[SEG_CAP]   : i32  -- 该段的 live 行数（1..2；pad 段 0）
//   eid[SEG_CAP]      : i32  -- 该段的 expert id（pad 段任意，取 0）
// 纯函数、无 atomic、不依赖 block 调度 ⇒ 同一路由表重复计算逐位相同。
//
// =============================================================================
// ⚠️ 这是 EAGER 臂（不是 capture 臂）—— 与 bf16 臂同一条限制
// =============================================================================
// moe_align 要在 HOST 上读 `route_idx_r`（device 侧由 route_topk 写出），所以本臂天然
// 需要一次 D2H 回读 + 小 H2D 上行，在 CUDA-graph capture 内非法。
//   * 调用方在 `dev.capturing()` 时**不派遣**（decline 回老路径）；
//   * shim 自身也在 capture 内 decline（防御性）。
//
// =============================================================================
// 形状域（冻结，来自 moe_bs_tl_config.txt）
// =============================================================================
// dim==5120 && inter==320 && topk∈[1,6] && rows∈[1,6] && nseg∈[1,36]，其余一律
// decline，由调用方回退 `expert_gate_up_fp4_batched`。
//
// =============================================================================
// ⚠️⚠️ 唯一需要人工转写的地方：`moe_bs_encode_tmaps()`（本文件 §3）
// =============================================================================
// 描述符的 dims / strides / box / swizzle 是 **TileLang 内部决定**的（它按 smem layout
// 推断选 swizzle，且 fp4 的 sub-byte 展开方式只有它的 lowering 知道）。**不许猜**：
// 权威配方是 `moe_bs_up_tl_host.cu`（生成器同时 dump 的 TileLang 自己的 host launcher，
// 里面有它对每个张量调 `cuTensorMapEncodeTiled` 的完整实参）。
// 本文件的 §3 已把「所有能钉死的部分」钉死（rank/dtype/interleave/oob/l2、以及
// 由几何推出的 gdim/gstride/box），只有 **swizzle 枚举**与 **fp4 的 box 首维是否按
// 字节数**这两处需要拿 dump 一次比对。运行期 `DSV41_MOE_BS_DEBUG=1` 会把本文件实际
// 用的 spec 打出来，和 host source 一对一 diff 即可。
// **参数序错误不会静默**：形参里描述符 / `float*` / `int*` 是不同类型，任何错位都是
// 编译错误（见 §5 的 `static_assert`）——这正是本 ABI 唯一的救赎。

#include <cuda_runtime.h>
#include <cuda.h>  // 只取 CUtensorMap 的类型与枚举（**不引用任何 driver 函数**）
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <tuple>
#include <type_traits>

// 生成物：改名后 include（TileLang 把每个 kernel 都叫 main_kernel）。
#define main_kernel moe_bs_up_tl_kernel
#if __has_include("moe_bs_up_tl.cu")
#include "moe_bs_up_tl.cu"
#else
// [capture-guard-landing fix] The AOT artifact is not generated yet (bs-moe
// peer's gen_moe_bs_aot.py output lands separately) — without this guard the
// build.sh wildcard compile of tilelang_gen/*_shim.cu fails on the missing
// include. Same pattern as head_bf16_shim.cu's FERRITE_HEAD_TL_GENERATED.
#define FERRITE_MOE_BS_TL_MISSING 1
#endif
#undef main_kernel

// ===========================================================================
// §0 形参形态自检（**编译期**，第一道闸）
// ===========================================================================
// 本臂的全部 host 逻辑都建立在「dump 是 TMA 描述符 ABI」这一个事实上（§4.1）。
// 若 lowering 变成了裸指针形态，描述符层整个不成立 —— **立刻停下**，不要带着一个
// 「能编译但走错路」的 shim 上 GPU。以 `moe_bs_up_tl_host.cu` 的配方转写为准（§4.3）。
template <class F>
struct FnFirst;
template <class R, class A0, class... An>
struct FnFirst<R (*)(A0, An...)> {
    using type = A0;
};
static_assert(
    std::is_same_v<
        std::remove_cv_t<std::remove_reference_t<
            typename FnFirst<decltype(&moe_bs_up_tl_kernel)>::type>>,
        CUtensorMap>,
    "the AOT dump is NOT the TMA-descriptor ABI (param 0 is not a CUtensorMap). "
    "Either lowering changed (re-derive from moe_bs_up_tl_host.cu: docs/agent/"
    "tilelang-moe-bs-wiring.md §4.3) or TL_DISABLE_TMA_LOWER got set - which this arm "
    "cannot use at all (blockscaled smem must be float4_e2m1_unpacked; see §4.1).");

namespace {

// ===========================================================================
// §1 冻结几何 —— 逐项来自 tilelang_gen/moe_bs_tl_config.txt（生成器写出，勿手改）
// ===========================================================================
constexpr int kSegCap = 36;      // SEG_CAP = VERIFY_ROWS(6) * TOPK_MAX(6)
constexpr int kBm = 64;          // MMA M-tile（BM%64==0；64 目标 / 128 原型认证回退）
constexpr int kDim = 5120;       // 模型维度 = up 的 K
constexpr int kNp = 320;         // 一个权重面（w1 / w3）的行数 = inter_local
constexpr int kNup = 2 * kNp;    // 640 = gate ‖ up
constexpr int kE = 384;          // n_routed
constexpr int kTopkMax = 6;
constexpr int kRowsMax = 6;
constexpr int kGran = 32;        // 一个 ue8m0 字节覆盖的 K
constexpr int kSfWords = kDim / (kGran * 4);  // 40 = K/128，每行一个权重面的 SF 字数
constexpr int kBn = 128;         // N-tile
constexpr int kBk = 128;         // K-tile
constexpr int kNh = 64;          // 每个 N-tile 取自一个权重面的行数（= NP/grid.x = 320/5）
constexpr int kGridX = kNup / kBn;            // 5
constexpr int kThreads = 128;    // 3 个工作 warp + 1 空转（原型 §2 的分工）

// 动态 smem（生成器算出；> 48 KiB ⇒ SetAttribute 是必要条件）。
// 公式 = stages*(BM*BK + BN*BK)*1B（unpacked fp4）+ stages*(BM+BN)*4B（SF 字）
//        + BM*BN*4B（C_sh 暂存）。BM=64/BN=BK=128/stages=6 ⇒ 184832 B。
// ⚠️ 重生成后**必须**把 moe_bs_tl_config.txt 的 smem_bytes 抄到这里（不符 ⇒ launch err 1）。
constexpr size_t kSmem = 184832;
// 每 expert 的 packed SF 池字节（w1 与 w3 各一份）
constexpr size_t kSfPlaneBytes = (size_t)kSfWords * kNp * 4;  // 51200 B/面/expert

// ---- 常驻 scratch（INIT 期分配一次，进程生命周期内复用）--------------------
uint8_t* g_a = nullptr;         // [SEG_CAP*BM, dim/2] u8 packed fp4
uint32_t* g_sfa = nullptr;      // [kSfWords * SEG_CAP*BM] u32 group-major
float* g_c = nullptr;           // [SEG_CAP*BM, 2*NP] f32
int* g_eid = nullptr;           // [SEG_CAP]
int* g_order = nullptr;         // [SEG_CAP*BM]
int* g_counts = nullptr;        // [SEG_CAP]

// ===========================================================================
// §2 driver API 的 dlopen 绑定（build.sh 不链 -lcuda，见文件头约束 1）
//     ⚠️ 这里**故意**不使用 cuda.h 里声明的那个符号：直接引用会引入对 libcuda 的
//     链接期依赖。dlsym 得到的函数指针签名与 cuda.h 的声明逐字一致（由 typedef 保证）。
// ===========================================================================
typedef CUresult (*PfnEncodeTiled)(
    CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
    void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);

PfnEncodeTiled g_encode = nullptr;

bool load_driver() {
    // libcuda.so.1 在任何 CUDA 进程里都已加载（cudart 依赖它），所以 dlopen 必然命中；
    // 但仍要判空并 decline，而不是崩。RTLD_GLOBAL：编码失败时的 CUresult 语义与
    // 其它 driver 调用一致（我们只取一个符号，不改变任何既有解析）。
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (h == nullptr) return false;
    g_encode = reinterpret_cast<PfnEncodeTiled>(dlsym(h, "cuTensorMapEncodeTiled"));
    return g_encode != nullptr;
}

// ===========================================================================
// §3 tensormap 构造 —— ⚠️ 逐项 VERIFY 对 moe_bs_up_tl_host.cu
// ===========================================================================
// 一个 spec = 一次 cuTensorMapEncodeTiled 的全部实参，外加人类可读的名字。
struct TmapSpec {
    CUtensorMapDataType dtype;
    cuuint32_t rank;
    const void* addr;
    cuuint64_t gdim[5];      // 全局维度，**最内层在前**（PTX/CUDA 的约定）
    cuuint64_t gstride[4];   // 全局 stride，**字节**，dims 1..rank-1（dim0 隐含 1 元素）
    cuuint32_t box[5];       // 每次搬运的 box（元素数）
    cuuint32_t estride[5];   // box 内元素步长（稠密 tile 全 1）
    CUtensorMapInterleave ilv;
    CUtensorMapSwizzle swz;
    CUtensorMapL2promotion l2;
    CUtensorMapFloatOOBfill oob;
    const char* what;
};

// fp4 的「逻辑元素」是 4 bit。TMA **没有** sub-byte 的 data type，所以全局张量按
// **打包后的字节**描述：dim0 = K/2 个 UINT8。smem 侧必须是 unpacked（1 B/元素）。
// ⚠️ VERIFY #1：box 首维是按**字节（K/2）**还是按**元素（K）**给。按字节时
//    box[0] = kBk/2 = 64；若 dump 给的是 128，说明 lowering 用了别的形态（见 wiring §4）。
constexpr cuuint32_t kABox = (cuuint32_t)(kBk / 2);

// ⚠️ VERIFY #2：swizzle。A_sh / B_sh 的 smem 行 = BK 字节 = 128 B（unpacked fp4）
//    ⇒ CU_TENSOR_MAP_SWIZZLE_128B 是预期值。C_sh 的行 = BN*4 = 512 B（f32）⇒ 也可能
//    是 128B + 多次搬运。以 host source 为准。
constexpr CUtensorMapSwizzle kSwzAB = CU_TENSOR_MAP_SWIZZLE_128B;
constexpr CUtensorMapSwizzle kSwzC = CU_TENSOR_MAP_SWIZZLE_128B;

// ---- 各操作数 --------------------------------------------------------------
// A: [M, K] fp4 packed ⇒ UINT8 [M, K/2]，M = SEG_CAP*BM
TmapSpec spec_a(const void* a) {
    TmapSpec s{};
    s.dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    s.rank = 2;
    s.addr = a;
    s.gdim[0] = (cuuint64_t)(kDim / 2);
    s.gdim[1] = (cuuint64_t)(kSegCap * kBm);
    s.gstride[0] = (cuuint64_t)(kDim / 2);  // 行距 = K/2 字节（连续，无 pad）
    s.box[0] = kABox;
    s.box[1] = (cuuint32_t)kBm;
    s.ilv = CU_TENSOR_MAP_INTERLEAVE_NONE;
    s.swz = kSwzAB;
    s.l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    s.oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    s.what = "A[SEG*BM, K/2] u8";
    return s;
}

// W1/W3: [E, NP, K] fp4 packed ⇒ UINT8 [E, NP, K/2]
TmapSpec spec_w(const void* w, const char* what) {
    TmapSpec s{};
    s.dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    s.rank = 3;
    s.addr = w;
    s.gdim[0] = (cuuint64_t)(kDim / 2);
    s.gdim[1] = (cuuint64_t)kNp;
    s.gdim[2] = (cuuint64_t)kE;
    s.gstride[0] = (cuuint64_t)(kDim / 2);              // 行距（K 连续）
    s.gstride[1] = (cuuint64_t)kNp * (kDim / 2);        // expert 面 stride
    s.box[0] = kABox;
    s.box[1] = (cuuint32_t)kNh;                         // 半块：64 行
    s.box[2] = 1;
    s.ilv = CU_TENSOR_MAP_INTERLEAVE_NONE;
    s.swz = kSwzAB;
    s.l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    s.oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    s.what = what;
    return s;
}

// SFA: 1-D uint32 [sf_words * M]
TmapSpec spec_sfa(const void* p) {
    TmapSpec s{};
    s.dtype = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    s.rank = 1;
    s.addr = p;
    s.gdim[0] = (cuuint64_t)kSfWords * (cuuint64_t)(kSegCap * kBm);
    s.box[0] = (cuuint32_t)kBm;   // 每 k-iter 取 BM 个字（sf_period == 1）
    s.ilv = CU_TENSOR_MAP_INTERLEAVE_NONE;
    s.swz = CU_TENSOR_MAP_SWIZZLE_NONE;   // SF 区必须线性（tcgen05.cp 直读）
    s.l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    s.oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    s.what = "SFA[sf_words*SEG*BM] u32";
    return s;
}

// SFW1/SFW3: 2-D uint32 [E, sf_words*NP]
TmapSpec spec_sfw(const void* p, const char* what) {
    TmapSpec s{};
    s.dtype = CU_TENSOR_MAP_DATA_TYPE_UINT32;
    s.rank = 2;
    s.addr = p;
    s.gdim[0] = (cuuint64_t)kSfWords * (cuuint64_t)kNp;
    s.gdim[1] = (cuuint64_t)kE;
    s.gstride[0] = (cuuint64_t)kSfPlaneBytes;  // = sf_words*NP*4（每 expert 的 SF 面）
    s.box[0] = (cuuint32_t)kNh;   // 半块 64 个字
    s.box[1] = 1;
    s.ilv = CU_TENSOR_MAP_INTERLEAVE_NONE;
    s.swz = CU_TENSOR_MAP_SWIZZLE_NONE;
    s.l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    s.oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    s.what = what;
    return s;
}

// C: [M, 2*NP] f32（TMA store）
TmapSpec spec_c(void* c) {
    TmapSpec s{};
    s.dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    s.rank = 2;
    s.addr = c;
    s.gdim[0] = (cuuint64_t)kNup;
    s.gdim[1] = (cuuint64_t)(kSegCap * kBm);
    s.gstride[0] = (cuuint64_t)kNup * 4;
    s.box[0] = (cuuint32_t)kBn;
    s.box[1] = (cuuint32_t)kBm;
    s.ilv = CU_TENSOR_MAP_INTERLEAVE_NONE;
    s.swz = kSwzC;
    s.l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    s.oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    s.what = "C[SEG*BM, 2*NP] f32";
    return s;
}

CUtensorMap g_tmap_a, g_tmap_w1, g_tmap_w3, g_tmap_sfa, g_tmap_sfw1, g_tmap_sfw3, g_tmap_c;

bool encode_one(CUtensorMap* out, const TmapSpec& s) {
    const CUresult r = g_encode(out, s.dtype, s.rank, const_cast<void*>(s.addr), s.gdim,
                                s.rank > 1 ? s.gstride : nullptr, s.box, s.estride, s.ilv, s.swz,
                                s.l2, s.oob);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[moe-bs] cuTensorMapEncodeTiled(%s) failed: CUresult=%d\n", s.what, (int)r);
        return false;
    }
    return true;
}

// ---- 描述符的生命周期：两层 ------------------------------------------------
//  * **A / SFA / C** 指常驻 scratch（INIT 期 cudaMalloc，进程内地址不变）⇒ INIT 期建一次。
//  * **W1 / W3 / SFW1 / SFW3** 指**专家池**，而 ferrite 的池是**每层一个 allocation**
//    （`LayerDev.expert_pool`，40 层各自一份）⇒ 基址每层都变，必须**按基址缓存重建**。
//    重建只是 host 侧写 128 B 的结构体（不产生 stream 操作）⇒ 在 eager 路径上零成本、
//    且不违反「capture 内不做 INIT 期动作」的规则（这里既不是 INIT 也不是 stream op）。
//  ⚠️ 不缓存的话，第 2 层开始就会用第 1 层的地址去 TMA —— 这是**静默错值**（读到别人的
//    权重），所以 `w_base` 的比较是必须的，不是优化。
bool encode_fixed_tmaps() {
    return encode_one(&g_tmap_a, spec_a(g_a)) && encode_one(&g_tmap_sfa, spec_sfa(g_sfa)) &&
           encode_one(&g_tmap_c, spec_c(g_c));
}

const void* g_w_base[4] = {nullptr, nullptr, nullptr, nullptr};  // w1, w3, sfw1, sfw3

bool ensure_w_tmaps(const void* w1, const void* w3, const void* sfw1, const void* sfw3) {
    if (g_w_base[0] == w1 && g_w_base[1] == w3 && g_w_base[2] == sfw1 && g_w_base[3] == sfw3)
        return true;  // 同一层的重复调用（eager 每步都会来一次）
    const bool ok = encode_one(&g_tmap_w1, spec_w(w1, "W1[E, NP, K/2] u8")) &&
                    encode_one(&g_tmap_w3, spec_w(w3, "W3[E, NP, K/2] u8")) &&
                    encode_one(&g_tmap_sfw1, spec_sfw(sfw1, "SFW1[E, sf_words*NP] u32")) &&
                    encode_one(&g_tmap_sfw3, spec_sfw(sfw3, "SFW3[E, sf_words*NP] u32"));
    if (!ok) return false;
    g_w_base[0] = w1;
    g_w_base[1] = w3;
    g_w_base[2] = sfw1;
    g_w_base[3] = sfw3;
    if (getenv("DSV41_MOE_BS_DEBUG") != nullptr) {
        const TmapSpec sa = spec_a(g_a);
        const TmapSpec sw = spec_w(w1, "W1");
        const TmapSpec sc = spec_c(g_c);
        fprintf(stderr,
                "[moe-bs] tmaps (re)built for pool %p: A box=(%u,%u) swz=%d | "
                "W box=(%u,%u,%u) swz=%d | SFA box=BM | C box=(%u,%u) swz=%d\n",
                w1, sa.box[0], sa.box[1], (int)sa.swz, sw.box[0], sw.box[1], sw.box[2],
                (int)sw.swz, sc.box[0], sc.box[1], (int)sc.swz);
    }
    return true;
}

// ===========================================================================
// §4 device 侧辅助内核（gather / scatter / 装载期 SF pack）
// ===========================================================================

constexpr int kMovThreads = 256;

// f32 幂次标度 -> ue8m0 字节。与 kernels/cuda/dsv41_experts_mxf4.cu:287 逐字同源：
// `fast_round_scale6` 的输出是 2 的幂，ue8m0(b) = 2^(b-127) ⇒ 字节 = 偏置指数。
__device__ __forceinline__ uint8_t tl_bs_f_pow2_to_ue8m0(float s) {
    if (!(s > 0.f)) return 0;
    int e = (int)((__float_as_uint(s) >> 23) & 0xFFu) - 127;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return (uint8_t)(e + 127);
}

// ---------------------------------------------------------------------------
// gather（每调用）：把 assignment 的 fp4 nibble 与 f32 标度搬进段缓冲，并**顺带 pack
// 成 group-major uint32**。
//   A[seg*BM + r][0 .. dim/2)   = xq4[assign][0 .. dim/2)      （pad 行写 0）
//   SFA[g*M + seg*BM + r]       = u32(四字节 ue8m0 for k ∈ [g*128, g*128+128))
//                                 其中第 b 字节覆盖 k ∈ [g*128 + b*32, ..+32)
//                                 = tl_bs_f_pow2_to_ue8m0(xsc4[assign][g*4 + b])
// 一个 block = (段, 段内行)；kThreads 个线程覆盖 dim/2 = 2560 字节 + 40 个字。
// pad 行的 nibble 与标度都写 0（内核无 mask，脏字节会进 MMA）。
// ---------------------------------------------------------------------------
__global__ void tl_moe_bs_gather_kernel(const uint8_t* __restrict__ xq4,
                                        const float* __restrict__ xsc4,
                                        uint8_t* __restrict__ a, uint32_t* __restrict__ sfa,
                                        const int* __restrict__ order,
                                        const int* __restrict__ counts, int k2, int nsc,
                                        int sf_words, int nseg) {
    const int seg = blockIdx.y;
    if (seg >= nseg) return;
    const int r = blockIdx.x;
    const int row = seg * kBm + r;
    const int64_t m = (int64_t)kSegCap * kBm;
    const int live = counts[seg];
    const int idx = (r < live) ? order[seg * kBm + r] : -1;

    // (a) fp4 nibble（2 值/字节，与 ferrite 的打包逐字节同构 ⇒ 纯 memcpy 语义）
    uint8_t* adst = a + (int64_t)row * k2;
    if (idx < 0) {
        for (int i = threadIdx.x; i < k2; i += kMovThreads) adst[i] = 0;
    } else {
        const uint8_t* asrc = xq4 + (int64_t)idx * k2;
        for (int i = threadIdx.x; i < k2; i += kMovThreads) adst[i] = asrc[i];
    }
    // (b) 标度：f32 -> ue8m0，4 字节装一个字；group-major（字 g 覆盖 128 个 K）
    for (int g = threadIdx.x; g < sf_words; g += kMovThreads) {
        uint32_t w = 0u;
        if (idx >= 0) {
            const float* s = xsc4 + (int64_t)idx * nsc + g * 4;
            w = (uint32_t)tl_bs_f_pow2_to_ue8m0(s[0]) |
                ((uint32_t)tl_bs_f_pow2_to_ue8m0(s[1]) << 8) |
                ((uint32_t)tl_bs_f_pow2_to_ue8m0(s[2]) << 16) |
                ((uint32_t)tl_bs_f_pow2_to_ue8m0(s[3]) << 24);
        }
        sfa[(int64_t)g * m + row] = w;
    }
}

// ---------------------------------------------------------------------------
// scatter（每调用）：把 C[seg*BM + r][0..2*NP) 按 order **解交错**回写调用方的 out。
//   C 的列序（生成物 bake）：tile bx 的第 j 列 = (j < NH) ? gate[bx*NH + j]
//                                                       : up  [bx*NH + (j - NH)]
//   而 out 的行内布局是 RAW gate‖up：out[(row*topk+slot)*2*inter + n]，
//   其中 gate 在 [0, NP)、up 在 [NP, 2*NP)。
//   对每个 C 列 c（全局，0..2*NP)：
//     half = c / kBn; j = c % kBn; bx = ...  — 直接由生成物的索引反推：
//     col = bx*kBn + j； n = (j < NH) ? bx*NH + j : kNp + bx*NH + (j - NH)
// ---------------------------------------------------------------------------
__global__ void tl_moe_bs_scatter_kernel(const float* __restrict__ c, float* __restrict__ out,
                                         const int* __restrict__ order,
                                         const int* __restrict__ counts, int nup, int out_pitch,
                                         int split, int nseg) {
    const int seg = blockIdx.y;
    if (seg >= nseg) return;
    const int r = blockIdx.x;
    const int live = counts[seg];
    const int idx = (r < live) ? order[seg * kBm + r] : -1;
    if (idx < 0) return;
    // up 的 out_pitch = topk*2*inter，分页 = (row*topk + slot)*2*inter；dn 直接 idx*dim。
    const int64_t dst = (split > 0)
                            ? ((int64_t)(idx / split) * out_pitch + (int64_t)(idx % split) * nup)
                            : ((int64_t)idx * out_pitch);
    const float* s = c + (int64_t)(seg * kBm + r) * nup;
    for (int col = threadIdx.x; col < nup; col += kMovThreads) {
        const int bx = col / kBn;
        const int j = col - bx * kBn;
        const int n = (j < kNh) ? (bx * kNh + j) : (kNp + bx * kNh + (j - kNh));
        out[dst + n] = s[col];
    }
}

// ---------------------------------------------------------------------------
// 装载期 pack（**一次**，`dsv41_moe_bs_pack_wsf`）：
//   src: 一个 expert 的一个权重面（w1 或 w3）的 ue8m0 面，row-major [rows, nsc] u8
//   dst: group-major packed uint32 [sf_words * rows]
//   语义：dst[g*rows + row] = u32(src[row*nsc + g*4 .. +4])   —— **纯字节搬运**
//         （ue8m0 字节不解释：值仍是 2^(b-127)），所以逐位无损、与原型 pack 同构。
//   为什么装载期：SF 是权重的一部分，永不变化；一次性换来热路径零额外 launch。
//   代价：kSfPlaneBytes(51200 B) × 2 面 × 384 expert × 40 层 ≈ 1.54 GiB/rank
//         （相对 bf16 臂的 +105 GiB/rank 是 1.5%；相对 fp4 池 35 GiB 是 4%）。
//   ⚠️ 装载期跑在**主流**上（load.rs 的一次显式动作），不是 hot path。
// ---------------------------------------------------------------------------
__global__ void tl_moe_bs_pack_wsf_kernel(const uint8_t* __restrict__ src,
                                          uint32_t* __restrict__ dst, int rows, int nsc,
                                          int sf_words) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint8_t* s = src + (int64_t)row * nsc;
    for (int g = 0; g < sf_words; ++g) {
        const uint8_t* p = s + g * 4;
        dst[(int64_t)g * rows + row] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
}

// ===========================================================================
// §5 INIT（懒初始化；SetAttribute / cudaMalloc / dlopen 全部只在这里）
// ===========================================================================

bool tl_bs_init() {
    static int state = 0;  // 0 = 未初始化, 1 = 已就绪, -1 = 失败
    if (state != 0) return state > 0;

    // (a) 动态 smem：> 48 KiB 默认上限，不设就 launch 失败（err 1）。**必要条件**。
    bool ok = cudaFuncSetAttribute(moe_bs_up_tl_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)kSmem) == cudaSuccess;
    (void)cudaGetLastError();

    // (b) 常驻 scratch
    if (ok) {
        ok = cudaMalloc(&g_a, (size_t)kSegCap * kBm * (kDim / 2)) == cudaSuccess &&
             cudaMalloc(&g_sfa, (size_t)kSfWords * kSegCap * kBm * 4) == cudaSuccess &&
             cudaMalloc(&g_c, (size_t)kSegCap * kBm * kNup * 4) == cudaSuccess &&
             cudaMalloc(&g_eid, kSegCap * sizeof(int)) == cudaSuccess &&
             cudaMalloc(&g_order, kSegCap * kBm * sizeof(int)) == cudaSuccess &&
             cudaMalloc(&g_counts, kSegCap * sizeof(int)) == cudaSuccess;
        (void)cudaGetLastError();
    }
    if (!ok) {
        // P1 (no-latch-death): -1 is a PERMANENT latch, kept deliberately. Every
        // failure reachable here is deterministic (SetAttribute rejection / cudaMalloc
        // OOM are repeatable), and the transient "called inside a capture" case is
        // intercepted by the P0-2 guard at the entry point — INIT is never entered
        // while capturing. A retry could not rescue a latched failure, so
        // re-attempting each call would only re-pay a guaranteed-to-fail init.
        state = -1;
        return false;
    }
    // A/SFA/C 的 base 已定 ⇒ 这三张 map 可以建（W 的 map 要等第一次调用拿到池指针）。
    state = 1;
    return true;
}

// 一次性「INIT 失败」提示：避免「armed 但每次静默测老路」（本项目 #1 测量偏置陷阱）。
void bs_init_failed_note(int rows, int dim, int inter) {
    static int reported = 0;
    if (reported++ == 0)
        fprintf(stderr,
                "[moe-bs] ARMED but INIT FAILED (SetAttribute/malloc/dlopen) -> this run measures "
                "the OLD path (rows=%d dim=%d inter=%d)\n",
                rows, dim, inter);
}

}  // namespace

// ===========================================================================
// §6 导出符号 1：up（gate‖up）block-scaled grouped GEMM + gather + scatter
// ===========================================================================
// w1 / w3 = ferrite 专家池里该专家的 gate / up 权重面基址（**零拷贝、零 dequant**）；
// sfw1 / sfw3 = 装载期 pack 出来的 group-major u32 面基址（每 expert 一份）。
// 本臂不接受 `DSV41_EXPERT_ILV`（交错布局把 w1/w3 混在一个区域里，w1 指针不再是一个
// 干净的 [NP, K/2] 面）—— 交错时由调用方 decline（见 wiring §5，与 bf16 臂同一条互斥）。
extern "C" int dsv41_moe_tilelang_gate_up_bs(
    const uint8_t* xq4,      // [rows*topk][dim/2] u8 —— routed 的 fp4 打包激活（行距 dim/2）
    const float* xsc4,       // [rows*topk][dim/32] f32 —— routed 的 per-(row,32) 标度
    float* out,              // [rows][topk][2*inter] f32（RAW gate‖up；swiglu 仍走既有 pass）
    const void* w1,          // u8 [E, NP, K/2]（浅指一个 expert 面；shim 用 base + e*NP*K/2）
    const void* w3,          // u8 [E, NP, K/2]
    const void* sfw1,        // u32 [E, sf_words*NP]
    const void* sfw3,        // u32 [E, sf_words*NP]
    const int* eid,          // [SEG_CAP] i32 —— HOST 数组
    const int* order,        // [SEG_CAP*BM] i32 —— HOST 数组（pad = -1）
    const int* counts,       // [SEG_CAP] i32 —— HOST 数组
    int nseg, int rows, int dim, int inter, int topk, cudaStream_t s) {
    if (xq4 == nullptr || xsc4 == nullptr || out == nullptr || w1 == nullptr || w3 == nullptr ||
        sfw1 == nullptr || sfw3 == nullptr || eid == nullptr || order == nullptr ||
        counts == nullptr)
        return 2;
    if (dim != kDim || inter != kNp) return 2;
    if (topk < 1 || topk > kTopkMax) return 2;
    if (rows < 1 || rows > kRowsMax) return 2;
    if (nseg < 1 || nseg > kSegCap) return 2;
    // 16B 对齐：A/W 走 TMA（必需），out 是 float2 store。
    if ((((uintptr_t)xq4 & 0xF) != 0) || (((uintptr_t)w1 & 0xF) != 0) ||
        (((uintptr_t)w3 & 0xF) != 0) || (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit): NEVER run cudaMalloc inside a capture — it
    // invalidates the caller's capture BEFORE we could decline. Decline here.
    // moe_bs is the EAGER arm (host route-table readback + a CPU-computed
    // tensormap), so it can never be a legal capture node anyway; this guard
    // keeps tl_bs_init()'s cudaMalloc/TMA setup out of the capture.
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone) {
        return 2;  // decline without touching capture
    }

    if (!tl_bs_init()) {
        bs_init_failed_note(rows, dim, inter);
        return 2;
    }
    // driver/tensormap：首次调用做一次 dlopen；A/SFA/C 的 map 在 INIT 后建一次；
    // W/SFW 的 map 按专家池基址缓存重建（池是每层一份 ⇒ 基址每层都变）。
    if (g_encode == nullptr) load_driver();
    if (g_encode == nullptr || !encode_fixed_tmaps() ||
        !ensure_w_tmaps(w1, w3, sfw1, sfw3)) {
        static int once = 0;
        if (once++ == 0)
            fprintf(stderr,
                    "[moe-bs] ARMED but tensormap init FAILED (dlopen libcuda / "
                    "cuTensorMapEncodeTiled) -> this run measures the OLD path\n");
        return 2;
    }

    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[moe-bs] ARMED gate_up_bs rows=%d dim=%d inter=%d topk=%d nseg=%d -> "
                    "grid=(%d,%d)x%d smem=%zu BM=%d BN=%d BK=%d + gather/scatter\n",
                    rows, dim, inter, topk, nseg, kGridX, kSegCap, kThreads, kSmem, kBm, kBn, kBk);
    }

    cudaError_t e;
    // (0) 元数据上行（小数组；EAGER 臂，见文件头）
    e = cudaMemcpyAsync(g_eid, eid, (size_t)nseg * sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpyAsync(g_order, order, (size_t)kSegCap * kBm * sizeof(int),
                        cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;
    e = cudaMemcpyAsync(g_counts, counts, (size_t)nseg * sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;

    // (1) gather + 激活 SF pack：fp4 nibble + f32 标度 -> ue8m0 group-major u32
    tl_moe_bs_gather_kernel<<<dim3((unsigned)kBm, (unsigned)kSegCap), kMovThreads, 0, s>>>(
        xq4, xsc4, g_a, g_sfa, g_order, g_counts, kDim / 2, kDim / 32, kSfWords, nseg);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (2) block-scaled grouped GEMM（生成物）。
    //     ⚠️ ARG ORDER：形参序由 TileLang lowering 决定，**以 moe_bs_tl_config.txt 的
    //     signature 行为准**；描述符 / float* / int* 类型不同 ⇒ 错位是编译错误，不会静默。
    moe_bs_up_tl_kernel<<<dim3((unsigned)kGridX, (unsigned)kSegCap), kThreads, kSmem, s>>>(
        g_tmap_a, g_tmap_w1, g_tmap_w3, g_tmap_sfa, g_tmap_sfw1, g_tmap_sfw3, g_eid, g_c);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (3) scatter：RAW gate‖up 写回 out（swiglu 由既有 kernel 做，与本臂无关）
    tl_moe_bs_scatter_kernel<<<dim3((unsigned)kBm, (unsigned)kSegCap), kMovThreads, 0, s>>>(
        g_c, out, g_order, g_counts, kNup, topk * kNup, topk, nseg);
    e = cudaGetLastError();
    return (int)e;
}

// ===========================================================================
// §7 导出符号 2：装载期 SF pack（每个 expert 的 w1/w3 标度面各调一次）
// ===========================================================================
// src = 该 expert 该面的 ue8m0 面（[rows, nsc] u8，行距 nsc = k/32）；
// dst = 该 expert 该面的 packed 池（[sf_words*rows] u32）。调用 2 × n_routed × layers 次。
// 幂等、无 atomic、纯搬运 ⇒ 重复调用逐位相同。
extern "C" int dsv41_moe_bs_pack_wsf(const void* src, void* dst, int rows, int k,
                                     cudaStream_t s) {
    if (src == nullptr || dst == nullptr) return 2;
    if (rows <= 0 || k <= 0 || (k % 32) != 0) return 2;
    const int sf_words = k / 128;  // gran=32 ⇒ 一个 uint32 覆盖 128 个 K
    if (sf_words <= 0 || (k % 128) != 0) return 2;
    const int nsc = k / 32;        // 面行的逻辑字节数（w1/w3 的 pitch 无 pad —— 见 wiring §3.2）
    tl_moe_bs_pack_wsf_kernel<<<dim3((unsigned)((rows + 255) / 256)), 256, 0, s>>>(
        reinterpret_cast<const uint8_t*>(src), reinterpret_cast<uint32_t*>(dst), rows, nsc,
        sf_words);
    return (int)cudaGetLastError();
}
