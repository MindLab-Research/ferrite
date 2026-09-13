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
// FMA-issue bound，36 sweep）。本 shim 用 **B300 原生 block-scaled MMA 直接吃 fp4 权重**：
// **e4m3 激活 × e2m1 权重** + ue8m0 标度进 tensor core，**零 dequant、零 bf16 副本**。
// ⚠️ 激活是 **fp8 e4m3**（1 B/value），不是 packed fp4 —— D2 精度修复（2026-09-13）：
// 官方 DeepSeek-V4.1 的 routed 激活是 `act_quant(fp8_block_size=32, ue8m0)`。设计记录
// `docs/agent/moe-bs-e4m3-activation-design.md`；`dsv41_moe_bs_act_e4m3_cap` 是它的
// 能力符号（旧 .so 会把这个语义变化静默当 fp4 读，见 §8）。
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
//   int dsv41_moe_tilelang_gate_up_bs(...)     -- up（gate‖up）block-scaled grouped GEMM
//                                                 + gather + scatter（RAW gate‖up 布局）。
//                                                 **HOST 段表**（A/B 基线）
//   int dsv41_moe_tilelang_gate_up_bs_dev(...) -- 同一计算，**DEVICE 段表 + DEVICE
//                                                 `nseg` 指针**：无 D2H 回读、无 H2D 上行
//                                                 ⇒ 可在 CUDA-graph capture 内运行
//   int dsv41_moe_bs_pack_wsf(...)             -- **装载期**：w1/w3 的 ue8m0 面
//                                                 row-major [NP, K/32] -> group-major
//                                                 packed uint32 [sf_words*NP]（每 expert）
//   int dsv41_moe_bs_act_e4m3_cap(void)        -- **能力符号**：本 `.so` 的 A operand 是
//                                                 e4m3 激活（1 B/value）。旧 `.so` 没有它
//                                                 ⇒ Rust 的 bs 臂不 arm（见 §8）
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
//     **走不通**：blockscaled 的 **B** smem 必须是 `float4_e2m1_unpacked`（packed smem
//     静默错值 err=3.0，原型 §4.1），而 packed-global → unpacked-smem 只有 TMA 的
//     tensor 形式能做（`copy_analysis.cc:539`）。A 侧是 e4m3，天然 1 B/元素 ⇒ 那条
//     「展开」对 A 不存在，但 B 仍需要它 ⇒ 本约束与 TMA 形态一概不变。
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
//   A   : [SEG_CAP*BM, K] u8   -- **已 gather + 每段 pad 到 BM 行**的 **e4m3** 激活
//                                  （1 B/value）。pad 行必须真的全 0（内核不做 mask；
//                                  0x00 = +0.0，SFA=0 ⇒ 贡献恰为 0）。
//   W1  : [E, NP, K/2]      u8   -- gate 面（ferrite 池里的 w1.weight，面内 K 连续）
//   W3  : [E, NP, K/2]      u8   -- up 面（w3.weight）
//   ⚠️ 激活的字节布局变了（packed fp4 半字节 → e4m3 直排），但 C ABI 形状**没变**
//      ⇒ 旧 .so 会把 5120 B 的行当 2560 B 的 fp4 读，是**静默错值**。这就是
//      `dsv41_moe_bs_act_e4m3_cap`（§8）存在的唯一理由。
//   ⚠️ expert 之间的步长是**调用方测量的 block stride**（`w_stride` 形参），**不是**
//      `NP*K/2`：ferrite 的池是「每 expert 一个 128 B 对齐的 6 面 block」布局。
//      面内行距 = K/2 不变（一个面内部是干净的 [NP, K/2] 连续块）。
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
// 两条臂：host-table（A/B 基线）与 device-table（**capture 臂**）
// =============================================================================
// `dsv41_moe_tilelang_gate_up_bs` 取三张段表为 **HOST 数组**，自己做一次 D2H 回读
// （`moe_align` 读 `route_idx_r`）+ 小 H2D 上行 ⇒ 在 CUDA-graph capture 内非法，
// 所以调用方在 `dev.capturing()` 时不派遣它。它是 A/B 基线，保留逐位不变。
//
// `dsv41_moe_tilelang_gate_up_bs_dev` 是同一计算的 capture 孪生体：三张段表 +
// `nseg` 都**已经在 device 上**（由 `dsv41_moe_align_from_group` 从
// `dsv41_route_group` 的输出投影出来，与 host 侧 `moe_align_host` 逐位相同），
// 因此**没有 D2H 回读、没有 H2D 上行** —— 整条链在 capture 内合法。
//   * 两个 mover kernel 的 `nseg` 形参统一成**指针**（`seg >= *nseg` 守卫），
//     两条臂共用同一 launch 序列（host 臂传常驻 scratch `g_nseg`，device 臂传
//     `tl_nseg`）；
//   * `nseg` 是 device 值时**没有 host 形状检查**（host 读不到），边界由
//     `dsv41_moe_align_from_group` 的 `min(n_active, SEG_CAP)` 与 mover 守卫保证；
//   * 两条臂都保留 capture guard v2（`g_a == nullptr` → decline），确保 INIT 期的
//     `cudaFuncSetAttribute` / `cudaMalloc` / `dlopen` 不可能落进 capture。
//
// =============================================================================
// 形状域（冻结，来自 moe_bs_tl_config.txt）
// =============================================================================
// dim==5120 && inter==320 && topk∈[1,6] && rows∈[1,6]（host 臂另有 nseg∈[1,36]），
// 其余一律 decline，由调用方回退 `expert_gate_up_fp4_batched`。
//
// =============================================================================
// ⚠️⚠️ 唯一需要人工转写的地方：`moe_bs_encode_tmaps()`（本文件 §3）
// =============================================================================
// 描述符的 dims / strides / box / swizzle 是 **TileLang 内部决定**的（它按 smem layout
// 推断选 swizzle，且 **W** 侧 fp4 的 sub-byte 展开方式只有它的 lowering 知道）。**不许猜**：
// 权威配方是 `moe_bs_up_tl_host.cu`（生成器同时 dump 的 TileLang 自己的 host launcher，
// 里面有它对每个张量调 `cuTensorMapEncodeTiled` 的完整实参）。
// 本文件的 §3 已把「所有能钉死的部分」钉死（rank/dtype/interleave/oob/l2、以及
// 由几何推出的 gdim/gstride/box），只有 **swizzle 枚举**与 **W 的 box 首维是否按
// 字节数（BK/2）** 这两处需要拿 dump 一次比对（A 是 e4m3 ⇒ 字节数 == 元素数，
// 那条二义性在 A 上已经消解）。运行期 `DSV41_MOE_BS_DEBUG=1` 会把本文件实际
// 用的 spec 打出来，和 host source 一对一 diff 即可。
// **参数序错误不会静默**：形参里描述符 / `float*` / `int*` 是不同类型，任何错位都是
// 编译错误（见 §5 的 `static_assert`）——这正是本 ABI 唯一的救赎。

#include <cuda_runtime.h>
#include <cuda.h>  // 只取 CUtensorMap 的类型与枚举（**不引用任何 driver 函数**）
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>  // std::max_element (Eid DIAG)
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
#ifndef FERRITE_MOE_BS_TL_MISSING

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
constexpr int kBm = 128;         // MMA M-tile。⚠️ 必须 = moe_bs_tl_config.txt 的 BM=128：
                                 // TileLang 0.1.14 的 `tcgen05.cp.32x128b.warpx4` 要求 SF
                                 // smem 行数是 128 的倍数 ⇒ BM=64 在 **trace 期**就被库拒掉
                                 // （gen_moe_bs_aot.py:266 `assert BM % 128 == 0`），所以 AOT
                                 // 是以 `--bm 128` 生成的认证几何（64 只是"目标几何"的历史遗留）。
                                 // 旁证：device dump 的 A 行块 = blockIdx.y*128、SFA 段步长
                                 // = 4608 = 36*128、A 的 box = (64 B, 128 行)。
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

// 动态 smem（**生成物实际用量**；> 48 KiB ⇒ SetAttribute 是必要条件）。
// ⚠️ 权威值 = 生成物自己的 launch（moe_bs_up_tl_host.cu 的 launch stack 末项 = 202752），
//    也等于 device 内核 buf_dyn_shmem 的最高偏移 + 尾区：
//      B_sh=98304 | SFA_sh=196608(+3072) | SFW_sh=199680(+3072)  ⇒  202752
//    注意 C_sh 与 A_sh **共享 offset 0**（两者生命周期不重叠：C 只在所有 MMA 之后写），
//    所以 C 的 65536 B **不叠加**。
// ⚠️ 不要抄 moe_bs_tl_config.txt 的 smem_bytes=268288：那是「未去别名」的保守账
//    （196608 fp4 + 6144 sf + 65536 c），268288 B = 262 KiB > sm_100 的 227 KiB/block
//    上限 ⇒ cudaFuncSetAttribute 直接失败。
// ⚠️ 不符 ⇒ launch err 1（cudaErrorInvalidValue），这是静默回退老路径的入口。
constexpr size_t kSmem = 166912;  // stages=3: ab 98304 + sf 3072 + c 65536
// 每 expert 的 packed SF 池字节（w1 与 w3 各一份）
constexpr size_t kSfPlaneBytes = (size_t)kSfWords * kNp * 4;  // 51200 B/面/expert

// ---- 常驻 scratch（INIT 期分配一次，进程生命周期内复用）--------------------
uint8_t* g_a = nullptr;         // [SEG_CAP*BM, dim] u8 e4m3（1 B/value）
uint32_t* g_sfa = nullptr;      // [kSfWords * SEG_CAP*BM] u32 group-major
float* g_c = nullptr;           // [SEG_CAP*BM, 2*NP] f32
int* g_eid = nullptr;           // [SEG_CAP]
int* g_order = nullptr;         // [SEG_CAP*BM]
int* g_counts = nullptr;        // [SEG_CAP]
// `nseg` 的 device 副本：host-table 入口每调用上行一次（4 字节），device-table
// 入口不用它（直接传调用方的 `tl_nseg`）。两个 mover kernel 的 `nseg` 形参因此
// 统一成指针 —— 两条臂共用同一 launch 序列（与 bf16 shim 同构）。
int* g_nseg = nullptr;          // [1]

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

// A 的「元素」是 **e4m3，1 元素 = 1 字节** ⇒ **没有 sub-byte 二义性**（旧的 VERIFY #1
// 「box 首维按字节还是按元素」在 A 上消解：两者相等）。
// W 的「逻辑元素」是 **4 bit**：TMA 用 `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`（值 14，
// 见权威 dump `moe_bs_up_tl_host.cu` 的 dtype=14）按**元素**描述权重，所以 dim0 / box[0]
// 都是 **元素单位（K, K-tile）**，只有 **行距 gstride[0] 仍是字节（K/2）**。
// ⚠️ VERIFY #1（W 侧，2026-09-13 dump 已定案）：dump 给的是 dtype=14、gdim[0]=5120、
//    gstride[0]=2560、box[0]=128 ⇒ 按**元素**给（不是旧注释说的 packed 字节 64）。
//    旧实现按 UINT8 + dim0=K/2 + box[0]=64 描述，与 dump 不符 ⇒ 静默错读。
constexpr cuuint32_t kABoxA = (cuuint32_t)kBk;        // A: BK 字节（= BK 个 e4m3）
// W: `kBk` 个 4-bit 元素 = 64 B（box[0] 与 gdim[0] 同为元素单位；见 spec_w）
constexpr cuuint32_t kABoxW = (cuuint32_t)kBk;

// ⚠️ VERIFY #2：swizzle。A_sh / B_sh 的 smem 行 = BK 字节 = 128 B（unpacked fp4）
//    ⇒ CU_TENSOR_MAP_SWIZZLE_128B 是预期值。C_sh 的行 = BN*4 = 512 B（f32）⇒ 也可能
//    是 128B + 多次搬运。以 host source 为准。
constexpr CUtensorMapSwizzle kSwzAB = CU_TENSOR_MAP_SWIZZLE_128B;
constexpr CUtensorMapSwizzle kSwzC = CU_TENSOR_MAP_SWIZZLE_128B;

// ---- 各操作数 --------------------------------------------------------------
// A: [M, K] e4m3 ⇒ UINT8 [M, K]（1 B/value），M = SEG_CAP*BM
TmapSpec spec_a(const void* a) {
    TmapSpec s{};
    s.dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    s.rank = 2;
    s.addr = a;
    s.gdim[0] = (cuuint64_t)kDim;
    s.gdim[1] = (cuuint64_t)(kSegCap * kBm);
    s.gstride[0] = (cuuint64_t)kDim;  // 行距 = K 字节（1 B/value，连续，无 pad）
    s.box[0] = kABoxA;
    s.box[1] = (cuuint32_t)kBm;
    s.ilv = CU_TENSOR_MAP_INTERLEAVE_NONE;
    s.swz = kSwzAB;
    s.l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    s.oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    s.what = "A[SEG*BM, K] u8 e4m3";
    return s;
}

// W1/W3: [E, NP, K] fp4 packed ⇒ UINT8 [E, NP, K/2]
//
// `w_stride` = **相邻 expert 的同一权重面之间的距离（字节）** —— 由调用方测量后传入。
// 它不是 `NP*K/2`：装载期把每个 expert 放成**一个 128 B 对齐的 block**，里面装 6 个
// 面（w1 | w1.scale | w3 | w3.scale | w2 | w2.scale），所以相邻 expert 的 w1 隔着一整个
// block（生产几何实测 2,641,920 B，是 `NP*K/2` = 819,200 B 的 3.225 倍）。
// **写死 `NP*K/2` 会让 TMA 从错地址取第 2 个 expert 起的权重 —— 静默错值**，
// 这是 2026-09-13 修复的根因。所以 stride 只能由知道池布局的调用方给。
TmapSpec spec_w(const void* w, int64_t w_stride, const char* what) {
    TmapSpec s{};
    // 权威 dump（`moe_bs_up_tl_host.cu` W1_desc）: dtype=14 (16U4_ALIGN16B)、
    // gdim[0]=5120、gstride[0]=2560、box[0]=128。⇒ dim0 / box[0] 按 **4-bit 元素**
    // 计，gstride[0] 按**字节**计（行距 K/2 = 2560 B）。旧的 UINT8 + dim0=K/2 +
    // box[0]=64 与 dump 不符，是 2026-09-13 修复的第二个根因。
    s.dtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;   // == 14（权威 dump）
    s.rank = 3;
    s.addr = w;
    s.gdim[0] = (cuuint64_t)kDim;                      // 5120（4-bit 元素单位）
    s.gdim[1] = (cuuint64_t)kNp;
    s.gdim[2] = (cuuint64_t)kE;
    s.gstride[0] = (cuuint64_t)(kDim / 2);              // 行距仍是字节（K/2 = 2560）
    s.gstride[1] = (cuuint64_t)w_stride;                // expert 面 stride = block stride
    s.box[0] = kABoxW;                                  // 128（4-bit 元素 = 64 B）
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
    s.box[0] = (cuuint32_t)(kBn / 4);  // f32: 128/4=32 elements = 128 B = swizzle span
    s.box[1] = (cuuint32_t)kBm;
    s.ilv = CU_TENSOR_MAP_INTERLEAVE_NONE;
    s.swz = kSwzC;
    s.l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    s.oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    s.what = "C[SEG*BM, 2*NP] f32";
    return s;
}

CUtensorMap g_tmap_a, g_tmap_w1, g_tmap_w3, g_tmap_sfa, g_tmap_sfw1, g_tmap_sfw3, g_tmap_c;

// FIX(2026-09-13): driver 595.91.07 (CUDA 13.2) rejects NULL array pointers —
// elementStrides AND globalStrides must both be non-NULL. Verified on B300:
// es=NULL→1, es=[1,1]→0; rank1 gs=NULL→1, gs=[0]→0.
// elementStrides[0] is ignored when interleave==NONE but must still be >=1 ([0,0]→1).
static constexpr cuuint32_t kElemStrideOnes[5] = {1, 1, 1, 1, 1};

bool encode_one(CUtensorMap* out, const TmapSpec& s) {
    const CUresult r = g_encode(out, s.dtype, s.rank, const_cast<void*>(s.addr), s.gdim,
                                s.gstride,        // never nullptr (rank=1 with gstride=[0] is fine)
                                s.box,
                                kElemStrideOnes,  // never nullptr
                                s.ilv, s.swz, s.l2, s.oob);
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
int64_t g_w_stride = -1;  // 与 g_w_base 同批：池基址变了 block stride 也会变

bool ensure_w_tmaps(const void* w1, const void* w3, const void* sfw1, const void* sfw3,
                    int64_t w_stride) {
    if (g_w_base[0] == w1 && g_w_base[1] == w3 && g_w_base[2] == sfw1 && g_w_base[3] == sfw3 &&
        g_w_stride == w_stride)
        return true;  // 同一层的重复调用（eager 每步都会来一次）
    const bool ok = encode_one(&g_tmap_w1, spec_w(w1, w_stride, "W1[E, NP, K/2] u8")) &&
                    encode_one(&g_tmap_w3, spec_w(w3, w_stride, "W3[E, NP, K/2] u8")) &&
                    encode_one(&g_tmap_sfw1, spec_sfw(sfw1, "SFW1[E, sf_words*NP] u32")) &&
                    encode_one(&g_tmap_sfw3, spec_sfw(sfw3, "SFW3[E, sf_words*NP] u32"));
    if (!ok) return false;
    g_w_base[0] = w1;
    g_w_base[1] = w3;
    g_w_base[2] = sfw1;
    g_w_base[3] = sfw3;
    g_w_stride = w_stride;
    if (getenv("DSV41_MOE_BS_DEBUG") != nullptr) {
        const TmapSpec sa = spec_a(g_a);
        const TmapSpec sw = spec_w(w1, w_stride, "W1");
        const TmapSpec sc = spec_c(g_c);
        fprintf(stderr,
                "[moe-bs] tmaps (re)built for pool %p (W expert stride=%lld B): "
                "A box=(%u,%u) swz=%d | W box=(%u,%u,%u) swz=%d | SFA box=BM | C box=(%u,%u) "
                "swz=%d\n",
                (void*)w1, (long long)w_stride, sa.box[0], sa.box[1], (int)sa.swz, sw.box[0],
                sw.box[1], sw.box[2], (int)sw.swz, sc.box[0], sc.box[1], (int)sc.swz);
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
// gather（每调用）：把 assignment 的 **e4m3 激活字节**与 f32 标度搬进段缓冲，并**顺带
// pack 成 group-major uint32**。
//   A[seg*BM + r][0 .. dim)     = xq4[assign][0 .. dim)        （pad 行写 0）
//   SFA[g*M + seg*BM + r]       = u32(四字节 ue8m0 for k ∈ [g*128, g*128+128))
//                                 其中第 b 字节覆盖 k ∈ [g*128 + b*32, ..+32)
//                                 = tl_bs_f_pow2_to_ue8m0(xsc4[assign][g*4 + b])
// 一个 block = (段, 段内行)；kThreads 个线程覆盖 dim = 5120 字节 + 40 个字。
// pad 行的 e4m3 字节（0x00 = +0.0）与标度都写 0（内核无 mask，脏字节会进 MMA）。
//
// `nseg` 是**指针**：host-table 入口传常驻 scratch `g_nseg`（上行一次），
// device-table 入口直接传调用方由 `dsv41_moe_align_from_group` 写出的 `tl_nseg`。
// 两条臂因此共用同一个 kernel，只有表的来源不同（A/B 基线）。
// ---------------------------------------------------------------------------
__global__ void tl_moe_bs_gather_kernel(const uint8_t* __restrict__ xq4,
                                        const float* __restrict__ xsc4,
                                        uint8_t* __restrict__ a, uint32_t* __restrict__ sfa,
                                        const int* __restrict__ order,
                                        const int* __restrict__ counts, int abytes, int nsc,
                                        int sf_words, int row_div, const int* nseg) {
    const int seg = blockIdx.y;
    if (seg >= *nseg) return;
    const int r = blockIdx.x;
    const int row = seg * kBm + r;
    const int64_t m = (int64_t)kSegCap * kBm;
    const int live = counts[seg] < kBm ? counts[seg] : kBm;
    const int idx = (r < live) ? order[seg * kBm + r] : -1;
    // ⚠️ order stores the FLAT assignment index (row*topk + slot); activations
    //    are quantized PER ROW (xq4 is [rows][dim], NOT [rows*topk][dim]).
    //    Source row = idx / row_div (= topk). The bf16 twin (moe_bf16_shim.cu:225)
    //    and the grouped gather (dsv41_route.cu:341) both do this division —
    //    the BS gather was the only one missing it.
    const int src_row = (idx < 0) ? -1 : (idx / row_div);

    // (a) e4m3 直读（1 B/value ⇒ `abytes == dim`，纯 memcpy 语义）
    uint8_t* adst = a + (int64_t)row * abytes;
    if (src_row < 0) {
        for (int i = threadIdx.x; i < abytes; i += kMovThreads) adst[i] = 0;
    } else {
        const uint8_t* asrc = xq4 + (int64_t)src_row * abytes;
        for (int i = threadIdx.x; i < abytes; i += kMovThreads) adst[i] = asrc[i];
    }
    // (b) 标度：f32 -> ue8m0，4 字节装一个字；group-major（字 g 覆盖 128 个 K）
    for (int g = threadIdx.x; g < sf_words; g += kMovThreads) {
        uint32_t w = 0u;
        if (src_row >= 0) {
            const float* s = xsc4 + (int64_t)src_row * nsc + g * 4;
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
//
// `nseg` 是指针，与 gather 同约定（见上）。
// ---------------------------------------------------------------------------
__global__ void tl_moe_bs_scatter_kernel(const float* __restrict__ c, float* __restrict__ out,
                                         const int* __restrict__ order,
                                         const int* __restrict__ counts, int nup, int out_pitch,
                                         int split, const int* nseg) {
    const int seg = blockIdx.y;
    if (seg >= *nseg) return;
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
        ok = cudaMalloc(&g_a, (size_t)kSegCap * kBm * kDim) == cudaSuccess &&
             cudaMalloc(&g_sfa, (size_t)kSfWords * kSegCap * kBm * 4) == cudaSuccess &&
             cudaMalloc(&g_c, (size_t)kSegCap * kBm * kNup * 4) == cudaSuccess &&
             cudaMalloc(&g_eid, kSegCap * sizeof(int)) == cudaSuccess &&
             cudaMalloc(&g_order, kSegCap * kBm * sizeof(int)) == cudaSuccess &&
             cudaMalloc(&g_counts, kSegCap * sizeof(int)) == cudaSuccess &&
             cudaMalloc(&g_nseg, sizeof(int)) == cudaSuccess;
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
// w_stride = **相邻 expert 的同一权重面之间的距离（字节）**，由调用方测量后传入 ——
// 池是「每 expert 一个 6 面 block」布局，所以这个 stride 是 block stride，不是
// `NP*K/2`（见 spec_w 的长注释）。SF 池是另一块分配（分段连续），stride 不需要传。
// 本臂不接受 `DSV41_EXPERT_ILV`（交错布局把 w1/w3 混在一个区域里，w1 指针不再是一个
// 干净的 [NP, K/2] 面）—— 交错时由调用方 decline（见 wiring §5，与 bf16 臂同一条互斥）。
extern "C" int dsv41_moe_tilelang_gate_up_bs(
    const uint8_t* xq4,      // [rows*topk][dim] u8 —— routed 的 **e4m3** 激活（行距 dim）
    const float* xsc4,       // [rows*topk][dim/32] f32 —— routed 的 per-(row,32) 标度
    float* out,              // [rows][topk][2*inter] f32（RAW gate‖up；swiglu 仍走既有 pass）
    const void* w1,          // u8 [E, NP, K/2]（expert 0 的面；expert e 在 base + e*w_stride）
    const void* w3,          // u8 [E, NP, K/2]
    const void* sfw1,        // u32 [E, sf_words*NP]
    const void* sfw3,        // u32 [E, sf_words*NP]
    const int* eid,          // [SEG_CAP] i32 —— HOST 数组
    const int* order,        // [SEG_CAP*BM] i32 —— HOST 数组（pad = -1）
    const int* counts,       // [SEG_CAP] i32 —— HOST 数组
    int nseg, int rows, int dim, int inter, int topk, int64_t w_stride, cudaStream_t s) {
    if (xq4 == nullptr || xsc4 == nullptr || out == nullptr || w1 == nullptr || w3 == nullptr ||
        sfw1 == nullptr || sfw3 == nullptr || eid == nullptr || order == nullptr ||
        counts == nullptr)
        return 2;
    if (dim != kDim || inter != kNp) return 2;
    if (topk < 1 || topk > kTopkMax) return 2;
    if (rows < 1 || rows > kRowsMax) return 2;
    if (nseg < 1 || nseg > kSegCap) return 2;
    // block stride：`cuTensorMapEncodeTiled` 要求 gstride 是 16 B 的倍数且为正。
    // 不合规一律 decline（返回 2）—— 绝不建一个静默错值的描述符。
    if (w_stride <= 0 || (w_stride % 16) != 0) return 2;
    // 16B 对齐：A/W 走 TMA（必需），out 是 float2 store。
    if ((((uintptr_t)xq4 & 0xF) != 0) || (((uintptr_t)w1 & 0xF) != 0) ||
        (((uintptr_t)w3 & 0xF) != 0) || (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit, v2 state-gated — matching the bf16 shim's pattern):
    // decline ONLY when INIT hasn't completed (g_a == nullptr — can't cudaMalloc
    // inside capture); once scratch is allocated, launches are capture-safe and
    // SHOULD enter the verify graph (the device-side moe_align from 0818d78
    // eliminated the host D2H that made this arm EAGER-only).
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone
        && g_a == nullptr) {
        return 2;  // INIT hasn't run yet — decline without touching capture
    }

    if (!tl_bs_init()) {
        bs_init_failed_note(rows, dim, inter);
        return 2;
    }
    // driver/tensormap：首次调用做一次 dlopen；A/SFA/C 的 map 在 INIT 后建一次；
    // W/SFW 的 map 按专家池基址缓存重建（池是每层一份 ⇒ 基址每层都变）。
    if (g_encode == nullptr) load_driver();
    if (g_encode == nullptr || !encode_fixed_tmaps() ||
        !ensure_w_tmaps(w1, w3, sfw1, sfw3, w_stride)) {
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
    // `nseg` 也上行到常驻 scratch —— 两个 mover 的 `nseg` 形参已统一成指针，
    // 这样本入口与 device-table 孪生体共用同一 launch 序列（见文件头「EAGER 臂」）。
    e = cudaMemcpyAsync(g_nseg, &nseg, sizeof(int), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return (int)e;

    // (1) gather + 激活 SF pack：fp4 nibble + f32 标度 -> ue8m0 group-major u32
    tl_moe_bs_gather_kernel<<<dim3((unsigned)kBm, (unsigned)kSegCap), kMovThreads, 0, s>>>(
        xq4, xsc4, g_a, g_sfa, g_order, g_counts, kDim, kDim / 32, kSfWords, (int)topk, g_nseg);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (2) block-scaled grouped GEMM（生成物）。
    //     ABI：形参序/类型 **100% 由 TileLang lowering 决定**，权威配方 =
    //       * moe_bs_up_tl_host.cu 的 `TVMFFIFunctionCall(main_kernel, args, 14)`（args[0..7]）
    //       * moe_bs_tl_config.txt 的 `device signature` 行
    //     本 dump 的顺序（⚠️ 与直觉相反的三点：C 是**描述符**、SFA 是**裸指针**、W 在**最后**）：
    //       (0) A_desc    CUtensorMap   TMA load
    //       (1) C_desc    CUtensorMap   TMA store（**不是** raw float*）
    //       (2) Eid       const int*    裸指针（唯一随形参走的元数据）
    //       (3) SFA       const uint*   裸指针 —— SFA 走 cp.async.bulk（tma_load(dst,src,...)），
    //                                   **不需要描述符**（§3 的 g_tmap_sfa 因此是 dead 的）
    //       (4) SFW1_desc CUtensorMap
    //       (5) SFW3_desc CUtensorMap
    //       (6) W1_desc   CUtensorMap   ← W 排在 SFW 之后！
    //       (7) W3_desc   CUtensorMap
    //     描述符 / float* / int* 类型互不兼容 ⇒ 任何错位都是编译错误，不会静默。
    moe_bs_up_tl_kernel<<<dim3((unsigned)kGridX, (unsigned)kSegCap), kThreads, kSmem, s>>>(
        g_tmap_a, g_tmap_c, g_eid, g_sfa, g_tmap_sfw1, g_tmap_sfw3, g_tmap_w1, g_tmap_w3);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // (3) scatter：RAW gate‖up 写回 out（swiglu 由既有 kernel 做，与本臂无关）
    tl_moe_bs_scatter_kernel<<<dim3((unsigned)kBm, (unsigned)kSegCap), kMovThreads, 0, s>>>(
        g_c, out, g_order, g_counts, kNup, topk * kNup, topk, g_nseg);
    e = cudaGetLastError();
    return (int)e;
}

// ===========================================================================
// §6b DEVICE-TABLE 入口 —— 让本臂进 CUDA-graph capture
// ===========================================================================
// 与上面的 host-table 入口**逐行相同**，只有一个本质区别：三张段表 + `nseg`
// **已经在 device 上**，由 `dsv41_moe_align_from_group`（kernels/cuda/
// dsv41_moe_align.cu）从 `dsv41_route_group` 的输出投影出来 —— 与 host 侧
// `moe_align_host` 的表逐位相同（等价性论证见该文件的头注释）。因此：
//   * **没有 D2H 回读、没有 H2D 上行**：整条链在 capture 内合法，这正是本入口
//     存在的理由（host-table 入口的 `cudaMemcpyAsync(..., HostToDevice, s)` 是
//     capture 里的非法操作，所以它被 `moe_tilelang_bs_ready()` 的 `!capturing()`
//     挡在 graph 外）；
//   * `nseg` 是 DEVICE 指针：没有 host 形状检查（设备上的值 host 读不到），
//     边界由 `dsv41_moe_align_from_group` 的 `min(n_active, SEG_CAP)` 和两个
//     mover 的 `seg >= *nseg` 守卫保证；`grid.y = SEG_CAP` 恒定；
//   * TMA 描述符的构建**完全不变**：W/SFW 的 map 按专家池基址缓存重建
//     （`ensure_w_tmaps` 只写 host 侧 128 B 结构体，不产生 stream 操作 ⇒
//     capture 内合法），A/SFA/C 的 map 在 INIT 后建一次。
// 旧入口保留为 A/B 基线：同一 `ARMED` 回执格式，便于逐行对比两条臂。
// ===========================================================================
extern "C" int dsv41_moe_tilelang_gate_up_bs_dev(
    const uint8_t* xq4,      // [rows*topk][dim] u8 —— routed 的 **e4m3** 激活（行距 dim）
    const float* xsc4,       // [rows*topk][dim/32] f32 —— routed 的 per-(row,32) 标度
    float* out,              // [rows][topk][2*inter] f32（RAW gate‖up；swiglu 仍走既有 pass）
    const void* w1,          // u8 [E, NP, K/2]（expert 0 的面；expert e 在 base + e*w_stride）
    const void* w3,          // u8 [E, NP, K/2]
    const void* sfw1,        // u32 [E, sf_words*NP]
    const void* sfw3,        // u32 [E, sf_words*NP]
    const int* eid_dev,      // [SEG_CAP] i32 —— **DEVICE**
    const int* order_dev,    // [SEG_CAP*BM] i32 —— **DEVICE**（pad = -1）
    const int* counts_dev,   // [SEG_CAP] i32 —— **DEVICE**
    const int* nseg_dev,     // [1] i32 —— **DEVICE**（dsv41_moe_align_from_group 的输出）
    int rows, int dim, int inter, int topk, int64_t w_stride, cudaStream_t s) {
    if (xq4 == nullptr || xsc4 == nullptr || out == nullptr || w1 == nullptr || w3 == nullptr ||
        sfw1 == nullptr || sfw3 == nullptr || eid_dev == nullptr || order_dev == nullptr ||
        counts_dev == nullptr || nseg_dev == nullptr)
        return 2;
    if (dim != kDim || inter != kNp) return 2;
    if (topk < 1 || topk > kTopkMax) return 2;
    if (rows < 1 || rows > kRowsMax) return 2;
    // nseg 的形状检查在这里**不存在**（device 上的值 host 读不到）——见上方文件头。
    // block stride：`cuTensorMapEncodeTiled` 要求 gstride 是 16 B 的倍数且为正
    // （与 host-table 入口同一条闸；见 spec_w 的长注释）。
    if (w_stride <= 0 || (w_stride % 16) != 0) return 2;
    // 16B 对齐：A/W 走 TMA（必需），out 是 float2 store。
    if ((((uintptr_t)xq4 & 0xF) != 0) || (((uintptr_t)w1 & 0xF) != 0) ||
        (((uintptr_t)w3 & 0xF) != 0) || (((uintptr_t)out & 0x1F) != 0))
        return 2;

    // P0-2 (graph-capture audit, v2 state-gated — matching the bf16 shim's pattern):
    // decline ONLY when INIT hasn't completed (g_a == nullptr — can't cudaMalloc
    // inside capture); once scratch is allocated, launches are capture-safe and
    // SHOULD enter the verify graph. This is the guard that makes `cudaMalloc`
    // unreachable from inside a capture.
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (s && cudaStreamIsCapturing(s, &cap_st) == cudaSuccess
        && cap_st != cudaStreamCaptureStatusNone
        && g_a == nullptr) {
        return 2;  // INIT hasn't run yet — decline without touching capture
    }

    if (!tl_bs_init()) {
        bs_init_failed_note(rows, dim, inter);
        return 2;
    }
    // driver/tensormap：首次调用做一次 dlopen；A/SFA/C 的 map 在 INIT 后建一次；
    // W/SFW 的 map 按专家池基址缓存重建（池是每层一份 ⇒ 基址每层都变）。这三步
    // 都是 host 侧动作（无 stream 操作、无 INIT 期动作）⇒ capture 内合法。
    if (g_encode == nullptr) load_driver();
    if (g_encode == nullptr || !encode_fixed_tmaps() ||
        !ensure_w_tmaps(w1, w3, sfw1, sfw3, w_stride)) {
        static int once = 0;
        if (once++ == 0)
            fprintf(stderr,
                    "[moe-bs] ARMED (device tables) but tensormap init FAILED (dlopen libcuda / "
                    "cuTensorMapEncodeTiled) -> this run measures the OLD path\n");
        return 2;
    }

    {
        static int reported = 0;
        if (reported++ == 0)
            fprintf(stderr,
                    "[moe-bs] ARMED gate_up_bs (device tables) rows=%d dim=%d inter=%d topk=%d "
                    "nseg_dev=<device> -> grid=(%d,%d)x%d smem=%zu BM=%d BN=%d BK=%d + "
                    "gather/scatter\n",
                    rows, dim, inter, topk, kGridX, kSegCap, kThreads, kSmem, kBm, kBn, kBk);
    }

    // 表已在 device 上 ⇒ 没有 (0) 元数据上行这一步。

    // (1) gather + 激活 SF pack：fp4 nibble + f32 标度 -> ue8m0 group-major u32
    tl_moe_bs_gather_kernel<<<dim3((unsigned)kBm, (unsigned)kSegCap), kMovThreads, 0, s>>>(
        xq4, xsc4, g_a, g_sfa, order_dev, counts_dev, kDim, kDim / 32, kSfWords, (int)topk, nseg_dev);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    // SYNC-DIAG (DSV41_MOE_BS_SYNC_DIAG=1): per-kernel sync to isolate which
    // kernel faults. Default OFF (sync kills perf). Enable for debugging only.
    static const bool g_sync_diag = []() {
        const char* v = getenv("DSV41_MOE_BS_SYNC_DIAG");
        return v != nullptr && v[0] != '0';
    }();
    if (g_sync_diag) {
        e = cudaStreamSynchronize(s);
        fprintf(stderr, "[moe-bs][SYNC-DIAG] gather done: %s\n",
                e == cudaSuccess ? "OK" : cudaGetErrorString(e));
        if (e != cudaSuccess) return (int)e;
    }

    // (2) block-scaled grouped GEMM（生成物）。ABI 与 host-table 入口一字不差 ——
    //     权威配方见那里的长注释（C 是描述符、SFA 是裸指针、W 排在最后）。
    //     唯一的变化：Eid 直接用调用方的 device 表（`eid_dev`），不再是常驻 scratch。
    // ⚠️ DIAG (2026-09-14): one-shot Eid bounds check — copy Eid[0..SEG_CAP) to
    // host and verify all values are valid LOCAL expert IDs (< kE_local). An
    // out-of-range Eid (e.g., a GLOBAL id like 336-383 for rank 7 in TP8) would
    // make the TMA read at global_id * w_stride from a pool that only holds
    // kE_local experts — massive OOB → illegal memory access. This check runs
    // ONCE (first call) and prints all values for visual inspection.
    static bool g_eid_diag_done = false;
    if (!g_eid_diag_done) {
        g_eid_diag_done = true;
        // FIX(eid-init-audit): skip during CUDA graph capture — D2H + stream
        // sync inside a capture is illegal and would fail the capture (which
        // is the whole point of this device-tables entry point).
        cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &cap_st);
        if (cap_st == cudaStreamCaptureStatusNone) {
            int eid_host[kSegCap];
            cudaError_t ec = cudaMemcpyAsync(eid_host, eid_dev, kSegCap * sizeof(int),
                                             cudaMemcpyDeviceToHost, s);
            if (ec == cudaSuccess) ec = cudaStreamSynchronize(s);
            if (ec == cudaSuccess) {
                int mx = eid_host[0];
                for (int i = 1; i < (int)kSegCap; ++i) if (eid_host[i] > mx) mx = eid_host[i];
                fprintf(stderr, "[moe-bs][DIAG] Eid[0..%d):", (int)kSegCap);
                for (int i = 0; i < (int)kSegCap && i < 12; ++i) fprintf(stderr, " %d", eid_host[i]);
                fprintf(stderr, "%s | max=%d\n",
                        kSegCap > 12 ? " ..." : "", mx);
            }
        }
    }
    moe_bs_up_tl_kernel<<<dim3((unsigned)kGridX, (unsigned)kSegCap), kThreads, kSmem, s>>>(
        g_tmap_a, g_tmap_c, eid_dev, g_sfa, g_tmap_sfw1, g_tmap_sfw3, g_tmap_w1, g_tmap_w3);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    if (g_sync_diag) {
        e = cudaStreamSynchronize(s);
        fprintf(stderr, "[moe-bs][SYNC-DIAG] MMA done: %s\n",
                e == cudaSuccess ? "OK" : cudaGetErrorString(e));
        if (e != cudaSuccess) return (int)e;
    }

    // (3) scatter：RAW gate‖up 写回 out（swiglu 由既有 kernel 做，与本臂无关）
    tl_moe_bs_scatter_kernel<<<dim3((unsigned)kBm, (unsigned)kSegCap), kMovThreads, 0, s>>>(
        g_c, out, order_dev, counts_dev, kNup, topk * kNup, topk, nseg_dev);
    e = cudaGetLastError();
    if (g_sync_diag) {
        cudaError_t es = cudaStreamSynchronize(s);
        fprintf(stderr, "[moe-bs][SYNC-DIAG] scatter done: %s\n",
                es == cudaSuccess ? "OK" : cudaGetErrorString(es));
    }
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

// ===========================================================================
// §8 导出符号 3：能力符号 —— A operand = **e4m3 激活**
// ===========================================================================
// D2 修复把 `xq4` 的**语义**从「packed fp4 半字节（dim/2 B/行）」改成「e4m3
// （dim B/行）」，但 **C ABI 的形状没变**（还是 `const uint8_t*` + 同样的形参序）
// ⇒ 旧 `.so` 会**静默**把 5120 B 的行当 2560 B 的 fp4 读（错值，不是报错）。
// Rust 侧用一个能力探针把这个语义版本钉死（先例：`dsv41_expert_act_e4m3_cap`）：
//   * 带本符号的 `.so` ⇒ 激活按 e4m3 喂（Rust 的 `supports_moe_bs_act_e4m3`）；
//   * 不带 ⇒ 探针为 None，`DSV41_MOE_TILELANG_BS` 臂**不 arm**（报一声），
//     而不会把 e4m3 字节喂给一个 fp4 内核。
// 没有这一条，D2 就是一个静默错值面。
extern "C" int dsv41_moe_bs_act_e4m3_cap(void) {
    return 1;  // A operand = e4m3 激活（1 B/value）
}
#endif
