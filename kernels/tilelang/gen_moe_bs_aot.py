#!/usr/bin/env python3
# gen_moe_bs_aot.py — 从 tcgen05 BLOCK-SCALED（A = fp8 e4m3 激活 × B = fp4 e2m1 权重，
# ue8m0 per-32 标度）MoE-up 原型
# 冻结出 ferrite 生产可链接的 AOT CUDA 源码，供
# `kernels/cuda/tilelang_gen/moe_bs_shim.cu` 合入 ferrite 的 nvcc 构建。
#
# 原型与全部结论：docs/agent/tcgen05-blockscaled-proto.md（远端 B300 实测）。
# 接线设计与验证手册：docs/agent/tilelang-moe-bs-wiring.md。
#
# 用法（远端 B300，tilelang 0.1.14）：
#   # STEP 0（前置，幂等）：把 vendored 的源码补丁打到**远端**已安装的 tilelang 上
#   scp -r kernels/tilelang/vendor ubuntu@43.202.208.136:~/tl_bs/vendor
#   ssh ubuntu@43.202.208.136 \
#     'cd ~/tl_bs && /opt/dlami/nvme/dsv41_venv/bin/python vendor/apply_tilelang_patch.py'
#   # STEP 1（本脚本）：lowering + codegen + dump 源码。库里没修 ⇒ 直接 RuntimeError。
#   scp kernels/tilelang/gen_moe_bs_aot.py ubuntu@43.202.208.136:~/tl_bs/
#   ssh ubuntu@43.202.208.136 \
#     'cd ~/tl_bs && /opt/dlami/nvme/dsv41_venv/bin/python gen_moe_bs_aot.py aot_gen'
# 产出（<outdir>）：
#   moe_bs_up_tl.cu       # device：up（gate‖up）block-scaled grouped GEMM
#   moe_bs_up_tl_host.cu  # host：TileLang 自己的 launcher —— CUtensorMap 的**权威配方**
#   moe_bs_tl_config.txt  # 冻结几何 + grid/block/smem + 参数签名 + tensormap 表
#
# down 臂（`--down`，见 §5）产出：
#   moe_bs_dn_tl.cu / moe_bs_dn_tl_host.cu / moe_bs_dn_tl_config.txt
#   #   dn : A [NSEG*BM, K_pad=384] e4m3（尾 64 列 0）× W2 [E, dim, 384/2] packed fp4
#   #        -> C [NSEG*BM, dim] f32（**列序不交错**，与 up 不同）
#
# ⚠️ GENERATED — do not edit（生成物由本脚本产出，shim 是手写件）。
#
# =============================================================================
# 0. 本文件相对原型的四处生成差异（都是为了「接进 ferrite 生产链」）
# =============================================================================
# 1. **权重不再按 segment 复制**（原型 `W: [NSEG, N, K]`），而是**按 expert 直取**
#    ferrite 的专家权重池（`W1: [E, NP, K]`、`W3: [E, NP, K]`，`e = Eid[by]`）。
#    原型的 `[NSEG, N, K]` 是「每个 segment 一份去重权重」的 staging，35 段 ≈ 2.2 GB/层
#    的副本；生产的池是**每 expert 一份、只读一次**，零副本、零 dequant。
# 2. **N 轴切成两个 64 宽的半块**（`B_sh[st, 0:NH]` ← w1、`B_sh[st, NH:2*NH]` ← w3）。
#    原因见 §1（ferrite 的池布局把 `w1.scale` 插在 w1 与 w3 之间 ⇒ 不存在连续 `[640, K]`
#    的 gate‖up 面；而 320 = 5×64 恰好让 5 个 BN=128 的 N-tile 无 pad 覆盖 640 列）。
# 3. **SF 走两条路**（见 §2 与 wiring §3）：
#    * 权重 SF：**装载期一次性** pack 成 group-major uint32（`SFW1/SFW3: [E, sf_words*NP]`）；
#    * 激活 SF：**每次调用**在 shim 的 gather kernel 里 pack（`SFA: [sf_words*M]`），
#      因为它每步都变（`xsc4_r` 是 f32 幂次标度，gather 顺带做 f32→ue8m0 的位转换）。
# 4. **ABI 是 TMA 描述符形态**（TileLang 默认 lowering，见 §3）。裸指针 ABI
#    （`TL_DISABLE_TMA_LOWER`）**走不通**：blockscaled 的 **B** smem 必须是
#    `float4_e2m1_unpacked`（packed smem 静默错值，原型 §4.1），而
#    packed-global → unpacked-smem 这个「展开」只有 TMA 的 tensor 形式能做
#    （`copy_analysis.cc:539`）。⇒ host 必须 `cuTensorMapEncodeTiled`。
#    （A 侧自 D2 修复起是 e4m3，天然 1 B/元素 ⇒ 这条「展开」对 A 整体消失；
#      B 仍需要它，所以 TMA 形态与 `cuTensorMapEncodeTiled` 一概不变。）
# 5. **A operand 是 fp8 e4m3**（D2 精度修复，2026-09-13）：官方 DeepSeek-V4.1 的
#    routed 激活是 `act_quant(fp8_block_size=32, ue8m0)` 的 **e4m3**，权重才是
#    MXFP4 e2m1；本臂此前把两侧都当 e2m1，激活整整少 4 bit 位宽。设计记录：
#    `docs/agent/moe-bs-e4m3-activation-design.md`。**W1/W3/SFW 一字未改**
#    （权重仍是 packed fp4 + ue8m0）。副产物：A 的 `VERIFY #1`（box 首维字节 vs
#    元素）在 A 上**自动消解**（1 元素 = 1 字节），只剩 W 那一侧仍是 packed 字节视图。
#
# =============================================================================
# 1. 为什么 B_sh 要拆两半（ferrite 池布局的硬约束）
# =============================================================================
# ferrite 的专家池（load.rs::load_expert_pool）把一个 expert 的六个面放在一个 128B
# 对齐的 block 里，**plain 顺序**是：
#     [w1][w1.scale][w3][w3.scale][w2][w2.scale]
# 所以 w1（[NP=320, K/2] u8，行距 K/2 连续）与 w3 之间隔着 w1.scale ⇒ **不存在**
# 一个连续的 `[2*NP=640, K]` gate‖up 面。而 blockscaled 的 N 必须 BN%128==0，320 整除
# 不了 128 ⇒ 也不能按面各起一个 kernel（320/128 = 2.5）。
#
# 解：把 BN=128 的 tile 理解成「两个 64 宽的半块」，第 bx 个 N-tile 取
#     B_sh[st, 0:64]  <- w1[e, bx*64 : bx*64+64, k*BK : k*BK+BK]
#     B_sh[st, 64:128] <- w3[e, bx*64 : bx*64+64, k*BK : k*BK+BK]
# 于是 grid.x = 640/128 = **5** 恰好覆盖全部 640 列（每次取 64 列/面 ⇒ 5×64 = 320 ✓），
# 无 pad、无越界读、无 id 掩码。这是 640 = 2×320 = 2×5×64 的直接结果。
#
# ⚠️ 代价：**C 的列序是交错的**（tile bx 的第 j 列，j<64 是 gate 列 bx*64+j，
# j>=64 是 up 列 bx*64+(j-64)）。scatter 必须按这个映射回写，见 shim 的
# `tl_moe_bs_scatter_kernel` 与 wiring §2.3。
# 备选（**不是**本生成器的默认）：让 loader 把池重排成 [w1][w3][w1.scale]...，则
# w1‖w3 连续（零副本、零额外显存），W 每迭代一次 TMA 即可。见 wiring §2.4。
#
# =============================================================================
# 2. SF（scale factor）语义与 pack 布局（原型 §1.4 的通读结论）
# =============================================================================
# `sf_*_granularity_k = gran` 是「一个 ue8m0 字节覆盖多少 K」，不是 MMA 的 K 原子。
# gran=32 = ferrite 原生 per-(row,32) MXFP4；一个 **uint32 = 4 个连续 e8m0 字节**
# = 128 个 K。host 侧布局是 **group-major**：`SF[g * rows + row]`，`g = k // 128`。
#
#   sf_words = K / (4*gran)                     # 40（K=5120）
#   sf_period = 4*gran / BK                     # 1  ⇒ 每 k-iter 一组 cp+transpose
#
# 权重 pack（**装载期**，`dsv41_moe_bs_pack_wsf`）：
#   源：w1.scale / w3.scale 面，`[NP, K/32]` u8，行距 = K/32 = 160 B（已是 16B 倍数，
#       `sf_pitch_plane` 只重排 w2.scale，所以这里**没有** pitch 例外）；
#   目标：`[E, sf_words * NP]` uint32，`word[g*NP + row] = u32(src[row*160 + g*4 .. +4])`。
#   一次转换、逐字节搬运、不解释数值 ⇒ **无损且与原型逐位一致**。
# 激活 pack（**每调用**，shim 的 gather kernel）：
#   源：`xq4_r`（[m, dim] u8 **e4m3**，1 B/value）与 `xsc4_r`（[m, dim/32] **f32** 标度）；
#   目标：`SFA[g*M + seg*BM + r]`，`u32 = b0 | b1<<8 | b2<<16 | b3<<24`，其中
#       `bb = f_pow2_to_ue8m0(xsc4_r[assign, g*4 + j])`
#       = `(bits(x) >> 23) - 127 + 127`（`fast_round_scale6` 的输出是 2 的幂，
#         ue8m0 字节 = 偏置指数 ⇒ **一个移位**，与 `dsv41_experts_mxf4.cu:287` 一致）。
#
# =============================================================================
# 3. ABI / 参数序（shim 必须逐项对齐；`moe_bs_tl_config.txt` 里有权威签名行）
# =============================================================================
# TileLang 默认 lowering 会把**参与 TMA 搬运**的张量变成 `__grid_constant__ const
# CUtensorMap` 形参（实测形态见 docs/agent/tilelang-integration-design.md §1.3 实验 A）。
# 本 kernel 参与 TMA 的有 6 个：A / W1 / W3 / SFA / SFW1 / SFW3；`Eid` 走普通 ld、
# `C` 走 TMA store ⇒ 也是描述符。**权威配方 = `moe_bs_up_tl_host.cu`**（TileLang 自己
# 生成的 host launcher，里面有它构造每个 CUtensorMap 的全套参数）——shim 的
# `moe_bs_encode_tmaps()` 就是那份配方的转写。**不要凭猜测写 descriptor。**
#
# ⚠️ 0.1.14 上游 bug（§4）修掉之前，本 kernel 连 lowering 都过不去 —— 修法是
# kernels/tilelang/vendor/ 里的**源码补丁**（不是运行时注入），见该目录 README。

import argparse
import ast
import hashlib
import inspect
import sys
import textwrap

import tilelang
import tilelang.language as T
import tilelang.language.gemm_op as _gemm_op

# =============================================================================
# 4. TileLang 0.1.14 上游缺陷的**前置校验**（本脚本不再注入任何东西）
# =============================================================================
# `T.tcgen05_gemm_blockscaled()` 只写了 `sf_a_granularity_k` / `sf_b_granularity_k`
# 两个注解，**漏写 `ann["is_tcgen05"] = 1`**（`T.tcgen05_gemm()` 有）。于是
# `cuda::Gemm::SelectInst`（src/cuda/op/gemm.cc）跳过 `isTcgen05_` 分支，落到
# 「SFA/SFB region 已定义 ⇒ 这一定是 SM120 的 NVF4 mma.sync 路径」并 FATAL：
#
#     InternalError: T.mma_gemm_blockscaled() requires an SM120 CUDA target,
#         but got target={..., "arch":"sm_103a"}
#
# 正确修法是**改库的源码**（vendored 补丁 + 幂等施补器，见
# `kernels/tilelang/vendor/README.md`）——不是在本进程里重新 exec 函数体把那一行
# 注入进去。用户裁决：**不允许任何 hack**。所以本脚本**只做校验**：
#
#   * 库里没修 ⇒ 立刻 `RuntimeError`，并把施补命令原文打出来（不猜、不降级）；
#   * 库里修好了 ⇒ 返回一句可审计的 reason，由 main() 打进 config 文件。
#
# 用 AST 判定而不是字符串搜索：`is_tcgen05` 在隔壁 `tcgen05_gemm()` 里也有，全文
# grep 会把「隔壁函数有」误判成「这个函数有」——那正是这个 bug 的形状。
def blockscaled_fix_state():
    """`("fixed"|"missing"|"unknown", detail)` —— 只读库的源码，不注入。"""
    fn = getattr(_gemm_op, "tcgen05_gemm_blockscaled", None)
    if fn is None:
        return "unknown", "tilelang.language.gemm_op has no tcgen05_gemm_blockscaled"
    try:
        src = textwrap.dedent(inspect.getsource(fn))  # 只取目标函数本身
    except (OSError, TypeError) as e:  # 源码不可得（纯 .pyc 安装）
        return "unknown", f"inspect.getsource failed: {type(e).__name__}: {e}"
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if (
                isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant)
                and t.slice.value == "is_tcgen05"
            ):
                return "fixed", "ann['is_tcgen05'] = 1 present in gemm_op.py"
    return "missing", "tcgen05_gemm_blockscaled() does not set ann['is_tcgen05']"


def verify_blockscaled_fix():
    """库没修就炸 —— 这是 AOT 生成的前置条件，不是可选检查。"""
    state, detail = blockscaled_fix_state()
    if state == "fixed":
        return detail
    cmd = "python3 kernels/tilelang/vendor/apply_tilelang_patch.py"
    raise RuntimeError(
        "TileLang's T.tcgen05_gemm_blockscaled() is UNFIXED "
        f"({detail}).\n"
        "  The 0.1.14 (and upstream main) omission of ann['is_tcgen05'] = 1 makes\n"
        "  cuda::Gemm::SelectInst fall through to the SM120 NVF4 branch and hard-fail\n"
        "  on sm_103a. Fix the INSTALLED tilelang SOURCE first (no monkey-patching):\n\n"
        "      scp -r kernels/tilelang/vendor ubuntu@<b300>:~/tl_bs/vendor\n"
        "      ssh ubuntu@<b300> 'cd ~/tl_bs && "
        "/opt/dlami/nvme/dsv41_venv/bin/python vendor/apply_tilelang_patch.py'\n\n"
        "  or, locally, with the interpreter that owns that tilelang:\n\n"
        f"      {cmd}\n\n"
        "  See kernels/tilelang/vendor/README.md §2/§3 for the pristine sha256 and the\n"
        "  expected 'tcgen05_gemm_blockscaled.is_tcgen05 : fixed' verdict."
    )


# ---- 冻结几何（照 tcgen05-blockscaled-proto.md §2 与 ferrite 的池布局）--------
E = 384               # n_routed（本 rank 的专家数）
DIM = 5120            # 模型维度 = up 的 K
INTER = 320           # inter_local = padded_inter(inter/world) = up 的 N/2
NP = INTER            # 一个权重面（w1 或 w3）的行数
N_UP = 2 * NP         # 640 = gate(320) ‖ up(320)
VERIFY_ROWS = 6       # m 的上界（chain_dev.rs 的 VERIFY_ROWS）
TOPK_MAX = 6          # topk 上界
SEG_CAP = VERIFY_ROWS * TOPK_MAX  # 36 = 最坏情况的段数上界（每 assignment 一段）
GRAN = 32             # 一个 ue8m0 字节覆盖的 K（= ferrite 原生 MXFP4 粒度）

# 默认几何。BM=64 是目标（M 浪费 32× 而非 64×、A 流量减半、smem 余量换来更深的流水）；
# BM=128 是原型实测过的那一档，**保留为认证回退**（`--bm 128` 一键切换）。
# ⚠️ BM=64 的 lowering 有效性见 wiring §5 step 1（SFA 的 `tcgen05.cp.32x128b.warpx4`
# 对 64 行的 SF 区是否成立，是唯一没被原型覆盖的点）。
BM_DEFAULT = 64
BN_DEFAULT = 128      # BN%128==0 是硬约束；128 也是 N 轴的切法（5 个 tile）
BK_DEFAULT = 128      # ≤ 4*gran = 128，且 ≥128（内层 128B ≥ 64B swizzle）
THREADS = 128         # 3 个工作 warp + 1 个空转（原型 §2 的分工）
STAGES_DEFAULT = 6    # 原型实测最优（stg=8 超 228KB smem）

# ---- down 臂（`--down`，见 §5）------------------------------------------------
# down 的 reduction 是 inter（不是 dim），N 轴才是 dim：K_pad = 384 = 3×128。
K_DN_PAD_DEFAULT = 384  # 320 → 384：唯一能让 `BK ≡ 128 ∧ BK | K` 成立的最近倍数
N_DN_DEFAULT = DIM      # down 的 N 轴 = 模型维度
# 默认几何与 up 同档（BM=128 是 TileLang 0.1.14 上唯一可行的 M-tile）。
# stages：K_pad/BK = 3 个 k-iter ⇒ 3 个 stage 刚好铺满，多出来的 buffer 只会白占 smem。
# ⚠️ `moe_bs_up` 的 `--stages` 默认 6 在这里**不适用**（k_iters=3）。
STAGES_DOWN_DEFAULT = 3


# ---------------------------------------------------------------------------
# ⚠️ TileLang 0.1.14 API 契约：`tilelang.jit` 必须装饰**返回 PrimFunc 的工厂函数**，
# **不能**装饰已经 `@T.prim_func` 构造好的对象。
#
# 0.1.14 的 eager 前端里 `T.prim_func(fn)` 是**立即构造** IR 的（builder.py:1660-1667），
# 返回一个 `tvm.tirx.PrimFunc`；而 `tilelang.jit` 的 decorator 第一件事是
#     pf = prim_func(func, eager_jit=True)        # jit/__init__.py:626
#     sig = inspect.signature(func)               # eager/builder.py:1634
# 即它把收到的东西**再当 Python 函数包一层 prim_func**。于是：
#   * `@T.prim_func` 对象没有 `get_kernel_source()`（它是 PrimFunc，不是 JITKernel）；
#   * 而且 `PrimFunc` 对象不是 callable ⇒ `inspect.signature()` 抛
#         TypeError: <该对象的 TVMScript 形态> is not a callable object
#     —— 报错里那句 `T.copy(T.region(C_tmem[0, 0], 1, 128, 128), ...)` 不是本文件的
#     kernel 源码（kernel 里根本没有 `T.region` 调用），而是那个 PrimFunc 对象的 repr，
#     被 `inspect.signature` 拼进了消息里。**这是伪线索**：epilogue 的
#     `T.copy(C_tmem, C_l)` 写法本身没有问题（原型同形且已认证）。
#
# 正确形态 = 原型 `k_up_bs` 的 lazy 模式（也是本文件修复前的目标形态）：
#     @tilelang.jit(...)
#     def factory(...):
#         @T.prim_func
#         def main(...): ...
#         return main
# 调用 factory(...) 返回 **JITKernel**（有 `get_kernel_source()` / `get_host_source()`
# / `export_sources()`），M 轴/形状参数走 factory 形参，方便 AOT 扫档。
# ---------------------------------------------------------------------------
# 配置：**不传 `pass_configs`**（与原型 `k_up_bs` 的 `@tilelang.jit(out_idx=[-1])` 完全同形）。
#
# ⚠️ 本文件此前写的是 `pass_configs={"tl::disable_tma_lower": False}` —— 两个问题：
#   1. **键名拼错**：0.1.14 的合法键是 `tl.disable_tma_lower`（点号），`tl::...` 会在
#      `PassConfigManager::Legalize` 立刻抛
#          AttributeError: Invalid config option 'tl::disable_tma_lower'
#      （JITKernel 构造阶段，晚于 lowering 一开始的 front-end 报错，所以这条错误在
#       修好装饰器之后才会浮出来）；
#   2. **该键已废弃**：`tl.disable_tma_lower` 在 0.1.14 的 `pass_config.py:345-349`
#      里是 deprecated（推荐改用 `T.copy(..., disable_tma=True)` 逐调用控制）。
# TMA lowering 在本版本**默认就是开的**（值 `False` 等于什么都不写），而 §3 的 ABI
# （6 个 `__grid_constant__ const CUtensorMap`）正是默认 lowering 的产物 ⇒ 干脆不传，
# 既少一个可弃依赖，也和认证过的原型一模一样。
@tilelang.jit(out_idx=[-1])
def moe_bs_up(NSEG, BM, NP_, K, E_, BN, BK, NH, threads=THREADS, stages=STAGES_DEFAULT,
              gran=GRAN):
    """grouped block-scaled (A e4m3 / B e2m1, ue8m0) up-GEMM，tcgen05 1-CTA，显式 async。

    grid (N_UP/BN, NSEG)；每个 CTA 认领一个 (expert 段, N-tile)。

    A   : [NSEG*BM, K]        float8_e4m3fn  段内行已 gather + BM-padding（行距 K 字节）
    W1  : [E_, NP_, K]        float4_e2m1fn  gate 面（K 连续，行距 K/2）
    W3  : [E_, NP_, K]        float4_e2m1fn  up 面
    SFA : [sf_words * NSEG*BM] uint32        每调用 pack（group-major）
    SFW1: [E_, sf_words*NP_]   uint32        装载期 pack（group-major）
    SFW3: [E_, sf_words*NP_]   uint32
    Eid : [NSEG] int32                       每段的 expert id
    C   : [NSEG*BM, 2*NP_]     float32       列序交错（见文件头 §1）

    唯一与原型不同的结构：B 的 128 行由**两次 TMA**（w1 半块 + w3 半块）拼成。
    """
    assert K % (gran * 4) == 0, "K must be a multiple of one packed SF word (4*gran)"
    assert K % BK == 0 and (K // BK) % (gran * 4 // BK) == 0
    assert BK % gran == 0 and BK % 32 == 0 and BK <= 4 * gran
    assert BN % 128 == 0
    # ⚠️ 本地（无 GPU）复现实测：BM=64 在 **trace 阶段**就会被库拒掉 ——
    #     T.tcgen05_cp_warpx4(SFA_sh[st, :], SFA_tmem)
    #   → builtin.py:_tcgen05_num_smem_chunks
    #     ValueError: Packed scale-factor helpers require total extent to be a multiple of 128, got 64.
    # 原因：`tcgen05.cp.32x128b.warpx4` 的 SF 区天然按 128 行铺；BM=64 时 SFA 的 smem 只有 64 个字。
    # ⇒ **0.1.14 上 BM 必须是 128 的倍数**（这正是 wiring §5 step 1 那个"唯一没被原型覆盖的点"
    #   的答案：不成立）。默认值 BM_DEFAULT=64 是「目标几何」的遗留值，AOT 生成请显式 `--bm 128`。
    assert BM % 128 == 0, (
        "tcgen05_cp_warpx4 requires the packed scale-factor smem extent to be a multiple of 128, "
        "so BM must be a multiple of 128 on TileLang 0.1.14 (BM=64 fails at trace time in "
        "builtin.py:_tcgen05_num_smem_chunks). Use --bm 128 for the certified geometry."
    )
    assert 2 * NP_ % BN == 0, "the N axis must tile by BN with no pad"
    assert NH == NP_ // (2 * NP_ // BN), "half-tile rows must partition the weight plane"
    sf_words = K // (gran * 4)
    sf_period = gran * 4 // BK
    k_iters = K // BK
    M = NSEG * BM
    GRID_X = 2 * NP_ // BN

    # ⚠️ 这里**只**留 `@T.prim_func`：外层 factory 已由 `@tilelang.jit` 装饰（见本文件
    # 上方 §"0.1.14 API 契约"）。在 `@T.prim_func` 之上再加 `@tilelang.jit` 会让 jit
    # 收到一个 **PrimFunc 对象**而不是 Python 函数 ⇒
    # `TypeError: ... is not a callable object`。原型的 `k_up_bs` 就是这个形态。
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float8_e4m3fn),
             W1: T.Tensor((E_, NP_, K), T.float4_e2m1fn),
             W3: T.Tensor((E_, NP_, K), T.float4_e2m1fn),
             SFA: T.Tensor((sf_words * M,), T.uint32),
             SFW1: T.Tensor((E_, sf_words * NP_), T.uint32),
             SFW3: T.Tensor((E_, sf_words * NP_), T.uint32),
             Eid: T.Tensor((NSEG,), "int32"),
             C: T.Tensor((M, 2 * NP_), "float32")):
        with T.Kernel(GRID_X, NSEG, threads=threads) as (bx, by):
            # smem dtype：B 必须是 unpacked（packed 会静默错值，原型 §4.1）；
            # A 是 e4m3 ⇒ 天然 1 B/元素，不需要（也没有）`_unpacked` 形态。
            A_sh = T.alloc_shared((stages, BM, BK), T.float8_e4m3fn)
            B_sh = T.alloc_shared((stages, BN, BK), T.float4_e2m1_unpacked)
            SFA_sh = T.alloc_shared((stages, BM), "uint32")
            SFW_sh = T.alloc_shared((stages, BN), "uint32")

            C_tmem = T.alloc_tmem([BM, BN], "float32")
            SFA_tmem = T.alloc_tmem([BM, 4], "uint32")
            SFW_tmem = T.alloc_tmem([BM, BN // 128 * 4], "uint32")

            C_l = T.alloc_fragment((BM, BN), "float32")
            C_sh = T.alloc_shared((BM, BN), "float32")

            loaded = T.alloc_barrier([32] * stages)    # TMA 完成
            sf_full = T.alloc_barrier([32] * stages)   # SF 转置 + fence 完成
            consumed = T.alloc_barrier([1] * stages)   # UMMA 已消费该 stage
            tmem_full = T.alloc_barrier([1])           # 累加器就绪

            tx = T.get_thread_binding()
            e = Eid[by]                                # 专家 id（W1/W3 的第三坐标）

            if tx < 32:
                # warp0：TMA producer（A + w1/w3 两个半块 + SFA + 两个 SFW 半块）
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(consumed[st], ph ^ 1)
                    T.tma_copy(A[by * BM:(by + 1) * BM, k * BK:(k + 1) * BK],
                               A_sh[st, :, :], barrier=loaded[st])
                    T.tma_copy(W1[e, bx * NH:(bx + 1) * NH, k * BK:(k + 1) * BK],
                               B_sh[st, 0:NH, :], barrier=loaded[st])
                    T.tma_copy(W3[e, bx * NH:(bx + 1) * NH, k * BK:(k + 1) * BK],
                               B_sh[st, NH:2 * NH, :], barrier=loaded[st])
                    if k % sf_period == 0:
                        g = k // sf_period
                        T.tma_copy(SFA[g * M + by * BM:g * M + (by + 1) * BM],
                                   SFA_sh[st, :], barrier=loaded[st])
                        T.tma_copy(SFW1[e, g * NP_ + bx * NH:g * NP_ + (bx + 1) * NH],
                                   SFW_sh[st, 0:NH], barrier=loaded[st])
                        T.tma_copy(SFW3[e, g * NP_ + bx * NH:g * NP_ + (bx + 1) * NH],
                                   SFW_sh[st, NH:2 * NH], barrier=loaded[st])
                    T.mbarrier_arrive(loaded[st])

            elif tx < 64:
                # warp1：SF smem→TMEM，随后发 block-scaled UMMA
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(loaded[st], ph)
                    T.mbarrier_wait_parity(sf_full[st], ph)
                    if k % sf_period == 0:
                        T.tcgen05_cp_warpx4(SFA_sh[st, :], SFA_tmem)
                        T.tcgen05_cp_warpx4(SFW_sh[st, :], SFW_tmem)
                    T.tcgen05_gemm_blockscaled(
                        A_sh[st, :, :], B_sh[st, :, :], C_tmem, SFA_tmem, SFW_tmem,
                        transpose_B=True,
                        mbar=consumed[st],
                        clear_accum=(k == 0),
                        k_start=k * BK,
                        sf_a_granularity_k=gran,
                        sf_b_granularity_k=gran,
                    )
                T.tcgen05_mma_arrive(tmem_full)

            elif tx < 96:
                # warp2：`tcgen05.cp.32x128b.warpx4` 要求 SF 字先转置（原型 §1.3）
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(loaded[st], ph)
                    if k % sf_period == 0:
                        T.tcgen05_sf_warp_transpose(SFA_sh[st, :])
                        T.tcgen05_sf_warp_transpose(SFW_sh[st, :])
                        T.fence_proxy_async()
                    T.mbarrier_arrive(sf_full[st])

            # epilogue：全部 warp
            T.mbarrier_wait_parity(tmem_full, 0)
            T.sync_threads()
            T.copy(C_tmem, C_l)
            T.copy(C_l, C_sh)
            T.copy(C_sh, C[by * BM, bx * BN])

    return main


# =============================================================================
# 5. down 臂：K-pad 320 → 384 让 blockscaled 的形状约束重新成立
# =============================================================================
# **为什么 down 需要一个 pad**（roadmap 判决的缺口）：routed down 的 reduction 是
# `K = inter_local = 320`，而 blockscaled 路径要求 `BK ≥ 128`、`BK ≤ 4·gran = 128`
# （即 `BK ≡ 128`）且 `BK | K`。`320 / 128 = 2.5` 不整除 ⇒ **K=320 上无解**，
# 所以 fp4 BS 臂此前只覆盖 gate/up（`N_UP=640, K=5120`）。
#
# 修法：**只把 K 轴补到 384**（`384 / 128 = 3` 整除 ✓），补出来的 64 列**全零**：
#   * A 侧：`[NSEG*BM, K_pad=384]` e4m3，`k ∈ [320, 384)` 的 64 个字节写 0
#     （0x00 = +0.0）；字节代价 +64 B/行 × 4608 行 = **+0.29 MB/层**（噪声）；
#   * W2 侧：每个权重面的行距从 320/2 = **160 B 变成 384/2 = 192 B**，尾部 32 B 为 0。
#     这是**真代价**：`192/160 = +20%` ⇒ w2 面 +163840 B/expert
#     ⇒ `+163840 × 384 expert × 40 层 ≈ +2.5 GB/rank`（见 PROVENANCE §7.5 的对照：
#     bf16 臂是 +105~113 GiB/rank，本臂这笔仍在同一量级之下）。
#     实现上**不需要新池、不需要新 pack kernel**：这就是 `load.rs::plan_pitch` 里
#     `w2.scale` 那一次 10 → 16 的**同一种 re-pitch**（`load.rs:165`），
#     把 `w2.weight` 也重排一次 160 → 192 即可 —— `upload_from_2d` 按物理 pitch
#     落行，池在分配时已 `zero_at` 过，所以尾部 32 B **天然是零**。
#   * SF 侧：`sf_words = 384/128 = 3`（K=320 时只有 2.5 个字，本来就不可达）。
#     装载期 pack 的词 2 只有前 2 个字节是真的（k ∈ [256,320)），高 2 字节写 0；
#     实现上给 `dsv41_moe_bs_pack_wsf` 加一个「源行逻辑宽度」上界（`nsc_src = 320/32 = 10`）
#     即可，越界字节写 0 —— 纯字节搬运语义不变。
#
# ⚠️ **K-pad 的数值影响 = 恒等**（验收 ⑤ 的答案）：补出来的每一项都含一个**恰好
#    +0.0 的乘数**（e4m3 `0x00` = +0.0、e2m1 nibble 0 = +0.0），所以
#    `sum_k (a_k·sa_k)·(b_k·sb_k)` 里这些项逐项为 0，MMA 的 fp32 累加加上 0 **不改变
#    任何真实项的位**（+0.0 加到 x 上等于 x；只有 x = -0.0 时才变 +0.0，而 -0.0 与
#    +0.0 在任何后续乘加里等价）。⇒ K-pad 不引入误差、不改变累加和。
#    推论：块的**零填充必须是真的零**（内核不做 mask，见 wiring §1.2 第 4 条）——
#    SFA/SFW2 的 pad 字也一并写 0，省得将来有人改 nibble 语义。
#
# 与 up 臂的**唯一结构差异**：down 的 N 轴（= `dim`）来自**一个**权重面
# （`w2.weight`，[E, dim, K_pad]），不像 up 要把 `B_sh` 拆成 w1/w3 两个 64 宽半块
# （ferrite 池把 `w1.scale` 插在两个面之间 ⇒ 不存在连续的 640 宽 gate‖up 面）。
# ⇒ 每个 k-iter 只有**一次** W2 TMA（box `(BK, BN, 1)`）与**一次** SFW2 TMA
# （box `(BN,)` 个字），且 **C 的列序不交错**（scatter 是普通行拷贝）。
@tilelang.jit(out_idx=[-1])
def moe_bs_down(NSEG, BM, N, KDIM, E_, BN, BK, threads=THREADS,
                stages=STAGES_DOWN_DEFAULT, gran=GRAN):
    """grouped block-scaled (A e4m3 / B e2m1, ue8m0) down-GEMM，tcgen05 1-CTA，显式 async。

    grid (N/BN, NSEG)；每个 CTA 认领一个 (expert 段, N-tile)。

    A   : [NSEG*BM, KDIM]      float8_e4m3fn  段内行已 gather + BM-padding + K-pad
                                               （行距 KDIM 字节；尾 KDIM-K_real 列 = 0）
    W2  : [E_, N, KDIM]        float4_e2m1fn  down 面（K 连续，**行距 KDIM/2 字节**）
    SFA : [sf_words * NSEG*BM] uint32         每调用 pack（group-major）
    SFW2: [E_, sf_words*N]     uint32         装载期 pack（group-major；尾字高字节 0）
    Eid : [NSEG] int32                        每段的 expert id
    C   : [NSEG*BM, N]         float32        **列序 = N**（与 up 的交错列序不同）

    唯一与 `moe_bs_up` 不同的结构：B 的 BN 行由**一次** TMA 填满（up 是两次半块）。
    """
    assert KDIM % (gran * 4) == 0, "K must be a multiple of one packed SF word (4*gran)"
    assert KDIM % BK == 0 and (KDIM // BK) % (gran * 4 // BK) == 0
    assert BK % gran == 0 and BK % 32 == 0 and BK <= 4 * gran
    # ⚠️ 下界同样是硬的（原型 §3.2 实测）：`BK < 128` 的内层只有 64 B < 128 B swizzle
    # 原子 ⇒ TMA 描述符非法，**运行期**才报 `Invalid TMA descriptor arguments`。
    # up 臂的同一约束只写在注释里（其认证几何是 BK=128，生成物已冻结）；down 的
    # 全部存在理由就是 BK ≡ 128，所以这里把它 assert 死 —— 一个不合法的 BK 不该
    # 产出一份只会在 GPU 上炸的 AOT 件。
    assert BK >= 128, (
        "BK < 128 leaves the inner row at 64 B < the 128 B swizzle atom: the TMA descriptor is "
        "invalid (measured in tcgen05-blockscaled-proto.md §3.2). BK must be exactly 128 here "
        "(BK <= 4*gran caps it too)."
    )
    assert BN % 128 == 0
    # `tcgen05_cp.32x128b.warpx4` 要求 SF 区行数是 128 的倍数 ⇒ BM 必须是 128 的倍数。
    assert BM % 128 == 0, (
        "tcgen05_cp_warpx4 requires the packed scale-factor smem extent to be a multiple of 128, "
        "so BM must be a multiple of 128 on TileLang 0.1.14 (BM=64 fails at trace time in "
        "builtin.py:_tcgen05_num_smem_chunks)."
    )
    assert N % BN == 0, "the N axis (dim) must tile by BN with no pad"
    sf_words = KDIM // (gran * 4)
    sf_period = gran * 4 // BK
    k_iters = KDIM // BK
    M = NSEG * BM
    GRID_X = N // BN

    @T.prim_func
    def main(A: T.Tensor((M, KDIM), T.float8_e4m3fn),
             W2: T.Tensor((E_, N, KDIM), T.float4_e2m1fn),
             SFA: T.Tensor((sf_words * M,), T.uint32),
             SFW2: T.Tensor((E_, sf_words * N), T.uint32),
             Eid: T.Tensor((NSEG,), "int32"),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(GRID_X, NSEG, threads=threads) as (bx, by):
            # smem dtype：B 必须是 unpacked（packed 会静默错值，原型 §4.1）；
            # A 是 e4m3 ⇒ 天然 1 B/元素，不需要（也没有）`_unpacked` 形态。
            A_sh = T.alloc_shared((stages, BM, BK), T.float8_e4m3fn)
            B_sh = T.alloc_shared((stages, BN, BK), T.float4_e2m1_unpacked)
            SFA_sh = T.alloc_shared((stages, BM), "uint32")
            SFW_sh = T.alloc_shared((stages, BN), "uint32")

            C_tmem = T.alloc_tmem([BM, BN], "float32")
            SFA_tmem = T.alloc_tmem([BM, 4], "uint32")
            SFW_tmem = T.alloc_tmem([BM, BN // 128 * 4], "uint32")

            C_l = T.alloc_fragment((BM, BN), "float32")
            C_sh = T.alloc_shared((BM, BN), "float32")

            loaded = T.alloc_barrier([32] * stages)    # TMA 完成
            sf_full = T.alloc_barrier([32] * stages)   # SF 转置 + fence 完成
            consumed = T.alloc_barrier([1] * stages)   # UMMA 已消费该 stage
            tmem_full = T.alloc_barrier([1])           # 累加器就绪

            tx = T.get_thread_binding()
            e = Eid[by]                                # 专家 id（W2 的第三坐标）

            if tx < 32:
                # warp0：TMA producer（A + W2 + SFA + SFW2 —— 比 up 少两次半块 TMA）
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(consumed[st], ph ^ 1)
                    T.tma_copy(A[by * BM:(by + 1) * BM, k * BK:(k + 1) * BK],
                               A_sh[st, :, :], barrier=loaded[st])
                    T.tma_copy(W2[e, bx * BN:(bx + 1) * BN, k * BK:(k + 1) * BK],
                               B_sh[st, :, :], barrier=loaded[st])
                    if k % sf_period == 0:
                        g = k // sf_period
                        T.tma_copy(SFA[g * M + by * BM:g * M + (by + 1) * BM],
                                   SFA_sh[st, :], barrier=loaded[st])
                        T.tma_copy(SFW2[e, g * N + bx * BN:g * N + (bx + 1) * BN],
                                   SFW_sh[st, :], barrier=loaded[st])
                    T.mbarrier_arrive(loaded[st])

            elif tx < 64:
                # warp1：SF smem→TMEM，随后发 block-scaled UMMA（与 up 逐行同构）
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(loaded[st], ph)
                    T.mbarrier_wait_parity(sf_full[st], ph)
                    if k % sf_period == 0:
                        T.tcgen05_cp_warpx4(SFA_sh[st, :], SFA_tmem)
                        T.tcgen05_cp_warpx4(SFW_sh[st, :], SFW_tmem)
                    T.tcgen05_gemm_blockscaled(
                        A_sh[st, :, :], B_sh[st, :, :], C_tmem, SFA_tmem, SFW_tmem,
                        transpose_B=True,
                        mbar=consumed[st],
                        clear_accum=(k == 0),
                        k_start=k * BK,
                        sf_a_granularity_k=gran,
                        sf_b_granularity_k=gran,
                    )
                T.tcgen05_mma_arrive(tmem_full)

            elif tx < 96:
                # warp2：`tcgen05.cp.32x128b.warpx4` 要求 SF 字先转置（原型 §1.3）
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(loaded[st], ph)
                    if k % sf_period == 0:
                        T.tcgen05_sf_warp_transpose(SFA_sh[st, :])
                        T.tcgen05_sf_warp_transpose(SFW_sh[st, :])
                        T.fence_proxy_async()
                    T.mbarrier_arrive(sf_full[st])

            # epilogue：全部 warp（列序就是 N，scatter 不需要解交错）
            T.mbarrier_wait_parity(tmem_full, 0)
            T.sync_threads()
            T.copy(C_tmem, C_l)
            T.copy(C_l, C_sh)
            T.copy(C_sh, C[by * BM, bx * BN])

    return main


def _dump(kern, path):
    src = kern.get_kernel_source()
    # POST-PROCESS (2026-09-14): inject tcgen05.relinquish_alloc_permit after the
    # tmem_allocate calls. TileLang 0.1.14's codegen omits this, violating the PTX
    # ISA precondition for tcgen05.dealloc (the CTA must relinquish its allocation
    # permit before dealloc). Without this, the kernel traps with "illegal
    # instruction" at the tail dealloc. Verified convention:
    # tests_tcgen05_mxf8f6f4_1x.cu:874-875, dsv41_experts_mxf4.cu:327-332.
    _RELINQUISH = ('    asm volatile('
                   '"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"'
                   ' ::: "memory");\n')
    _ALLOC_TAIL = '  }'
    _NEEDLE = 'tl::tmem_allocate'
    if _NEEDLE in src and 'relinquish_alloc_permit' not in src:
        lines = src.split('\n')
        out = []
        in_alloc_block = False
        alloc_seen = False
        for i, ln in enumerate(lines):
            out.append(ln)
            if _NEEDLE in ln:
                alloc_seen = True
                # The alloc calls are inside `if (warp==0) { ... }` — find the
                # closing brace of that block and inject before it.
                # Simpler: the very next `  }` line after the last alloc.
                in_alloc_block = True
            elif in_alloc_block and ln.rstrip() == _ALLOC_TAIL:
                # This is the closing brace of the alloc block.
                out.pop()  # remove the brace
                out.append(_RELINQUISH.rstrip('\n'))
                out.append(_ALLOC_TAIL)
                in_alloc_block = False
        src = '\n'.join(out)
    with open(path, "w") as f:
        f.write(src)
    return src


def _dump_host(kern, path):
    """TileLang 自己的 host launcher —— **CUtensorMap 的权威配方**。

    0.1.14 三个入口的名字在不同小版本间漂过，所以按可用性探测（导出源码是路线 (a)
    的必要信息：descriptor 的 dims/strides/box/swizzle 全在里面）。拿不到就返回 None，
    由 config 文件记一行 MISSING —— 主 agent 手工从 `JITKernel` 缓存里取。
    """
    for getter in ("get_host_source", "get_host_source_only"):
        f = getattr(kern, getter, None)
        if callable(f):
            try:
                src = f()
            except Exception as e:  # noqa: BLE001 — 探测，不是逻辑错误
                print(f"[warn] {getter}() failed: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            if src:
                with open(path, "w") as fh:
                    fh.write(src)
                return src
    try:
        kern.export_sources(kernel_path=None, host_path=path)
        with open(path) as fh:
            return fh.read()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] export_sources() failed: {type(e).__name__}: {e}", file=sys.stderr)
    return None


def _sha(src):
    return hashlib.sha256(src.encode()).hexdigest()


def _banner(name, raw_sha, raw_bytes):
    return (
        f"// {name} — GENERATED by kernels/tilelang/gen_moe_bs_aot.py; DO NOT EDIT.\n"
        f"// raw sha256 (this dump, before this banner) = {raw_sha}\n"
        f"// raw bytes = {raw_bytes}\n"
        f"// Regenerate: see kernels/cuda/tilelang_gen/PROVENANCE.md §9.\n"
    )


def _gen_down(args, reason):
    """`--down`：emit moe_bs_dn_tl.{cu,host.cu,banner} + moe_bs_dn_tl_config.txt。

    与 `main()` 的 up 块逐项同构（同一套几何推导 + 同一份 CUtensorMap 表），
    差别只在：K 是 `--kpad`（不是 dim）、N 是 dim、B 来自**一个**面（没有半块拆分）、
    C 的列序不交错。所有数字仍由**本函数**算，shim 只许读 config（勿手改）。
    """
    BM, BN, BK, ST = args.dn_bm, args.dn_bn, args.dn_bk, args.dn_stages
    gran, KDIM, N = args.gran, args.kpad, args.n
    assert KDIM % (gran * 4) == 0, f"kpad={KDIM} 必须是一个 SF 字（4*gran={gran * 4}）的倍数"
    assert KDIM % BK == 0, f"kpad={KDIM} 必须被 BK={BK} 整除（BK | K 是硬约束）"
    assert BN % 128 == 0 and BM % 128 == 0
    assert N % BN == 0, f"N={N} must be divisible by BN={BN}"
    grid_x = N // BN
    sf_words = KDIM // (gran * 4)
    k_iters = KDIM // BK

    kern = moe_bs_down(SEG_CAP, BM, N, KDIM, E, BN, BK,
                       threads=THREADS, stages=ST, gran=gran)
    src = _dump(kern, f"{args.outdir}/moe_bs_dn_tl.cu")
    host = _dump_host(kern, f"{args.outdir}/moe_bs_dn_tl_host.cu")
    sha = _sha(src)
    with open(f"{args.outdir}/moe_bs_dn_tl.banner", "w") as f:
        f.write(_banner("moe_bs_dn_tl.cu", sha, len(src)))

    # smem 预算（与 up 同一个公式：A = e4m3、B = unpacked e2m1 ⇒ 都是 1 B/元素；
    # C_sh 与 A_sh 生命周期不重叠，TileLang 会把它们叠在 offset 0 ⇒ 本式是**保守上界**，
    # 真实峰值见 dump 的 buf_dyn_shmem 偏移；保守值才是 SetAttribute 该传的）
    smem_ab = ST * (BM * BK + BN * BK)
    smem_sf = ST * (BM + BN) * 4
    smem_c = BM * BN * 4
    smem_total = smem_ab + smem_sf + smem_c
    smem_aliased = max(smem_c, ST * BM * BK) + ST * BN * BK + smem_sf
    host_src_note = (
        "moe_bs_dn_tl_host.cu"
        if host
        else "MISSING - transcribe the CUtensorMap recipe by hand (see wiring §4)"
    )
    sig = [l for l in src.splitlines() if "main_kernel" in l and "__global__" in l]
    w2_pitch = KDIM // 2  # packed fp4 行距（字节）
    w2_real = INTER // 2  # K-pad 之前每行的真实字节数（160）
    lines = [
        "# MoE tcgen05 block-scaled (A e4m3 / B e2m1, ue8m0) DOWN-GEMM AOT config",
        "# GENERATED by gen_moe_bs_aot.py --down — the shim's constants are read off THIS file.",
        f"is_tcgen05_fix=verified-in-source ({reason})",
        f"E={E} DIM={DIM} INTER={INTER} K_REAL={INTER} K_PAD={KDIM} N={N} SEG_CAP={SEG_CAP} "
        f"VERIFY_ROWS={VERIFY_ROWS} TOPK_MAX={TOPK_MAX}",
        f"BM={BM} BN={BN} BK={BK} threads={THREADS} stages={ST} gran={gran}",
        f"sf_words={sf_words} sf_period={gran * 4 // BK} k_iters={KDIM // BK}",
        f"grid=({grid_x}, {SEG_CAP}) ctas={grid_x * SEG_CAP}",
        f"smem_bytes={smem_total} (ab {smem_ab} + sf {smem_sf} + c {smem_c})",
        f"smem_bytes_aliased={smem_aliased} (C/A 叠放后的预期峰值："
        f"max(c {smem_c}, A {ST * BM * BK}) + B {ST * BN * BK} + sf {smem_sf})",
        f"raw_sha256(dn)={sha}",
        f"dn_cu_bytes={len(src)}",
        f"host_source={host_src_note}",
        "",
        "# ---- device signature (the shim's parameter order / types) ----",
        *sig,
        "",
        "# ---- K-pad 的形状契约（补出来的列/字节必须**真的是 0**）----",
        f"#   A : [SEG_CAP*BM, {KDIM}] e4m3  行距 {KDIM} B；k ∈ [{INTER}, {KDIM}) 的 "
        f"{KDIM - INTER} 列写 0x00（= +0.0）",
        f"#   W2: [E, {N}, {KDIM}] 4-bit 元素  行距 {w2_pitch} B（= {KDIM}/2；K-pad 前是 {w2_real} B，"
        f"+{(w2_pitch - w2_real) * 100 // w2_real}%）；尾部 {w2_pitch - w2_real} B = 0",
        f"#     ⚠️ 池侧实现 = load.rs::plan_pitch 对 w2.weight 做同 w2.scale 的一次 re-pitch"
        f"（{w2_real} -> {w2_pitch}），尾部零由池的 zero_at 保证，无需新池/新 pack",
        f"#   SFW2: [E, {sf_words}*{N}] u32  词 g={KDIM // 128 - 1}（最后一个）覆盖 "
        f"k ∈ [{KDIM - 128}, {KDIM})，其中 k >= {INTER} 的字节写 0",
        f"#   SFA : [sf_words*M] 同规则（pad 行的 e4m3 字节与 SF 都写 0）",
        "",
        "# ---- tensormap operands (authoritative recipe = moe_bs_dn_tl_host.cu) ----",
        "# kind=tma tensors, in the order TileLang will pass them:",
        f"#   A     [M, {KDIM}]        e4m3 (row stride {KDIM} B, 1 B/value)",
        f"#   W2    [E, {N}, {KDIM}]    fp4 packed (dtype=16U4_ALIGN16B；row stride {w2_pitch} B, "
        f"expert stride = measured block)",
        f"#   SFA   [sf_words*M]        uint32 (1-D，裸指针，**不走描述符**：dump 里是 cp.async.bulk)",
        f"#   SFW2  [E, {sf_words}*{N}] uint32 (2-D: expert x word)",
        f"#   Eid   [NSEG]              int32  (plain, no TMA)",
        f"#   C     [M, {N}]            f32    (TMA store; row stride {N * 4} B)",
        "",
        "# ---- 对新 dump 的审计值（机械比对，勿凭记忆）----",
        "#   idesc_blockscaled=144708608 (0x08A01400): a_format=0 E4M3, b_format=5 E2M1, K32, E8M0"
        "  <- 与 up 同一个 idesc（格式没变）",
        f"#   a_tma_bytes_per_k_iter={BM * BK} (BM*BK*1 B)",
        f"#   b_tma_bytes_per_k_iter={(BN * BK) // 2} (BN*BK 个 4-bit = 半数字节；up 是两个半块各 "
        f"{(BN * BK) // 4} B)",
        f"#   mma_template=tcgen05mma_blockscaled_ss<tl::DataType::kFloat8_e4m3,false>",
        f"#   a_row_stride_bytes={KDIM}",
        f"#   w2_row_stride_bytes={w2_pitch}",
        "",
        "# ---- host-side layout contract (frozen in the POOL, not in this dump) ----",
        f"#   w2 plane（K-pad 后）: [{N}, {w2_pitch}] u8   row pitch {w2_pitch} B",
        f"#   w2.scale plane       : [{N}, {INTER // 32}] u8 逻辑宽度 {INTER // 32} B "
        f"（物理 16 B，SF-pitch fix）",
        f"#     -> down pack 输出 word[g*{N} + row] = u32(src + row*{INTER // 32} + g*4)，"
        f"g*4+b >= {INTER // 32} 的字节写 0",
        f"#   packed SF pool: [E, {sf_words}*{N}] uint32 = {sf_words * N * 4} B/expert",
        f"#   C column order: tile bx covers dn[{BN}*bx .. +{BN}) —— **无交错**",
    ]
    with open(f"{args.outdir}/moe_bs_dn_tl_config.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default=".")
    ap.add_argument("--bm", type=int, default=BM_DEFAULT,
                    help="MMA M-tile；必须 %%64==0（64 是目标，128 是原型认证回退）")
    ap.add_argument("--bn", type=int, default=BN_DEFAULT)
    ap.add_argument("--bk", type=int, default=BK_DEFAULT)
    ap.add_argument("--stages", type=int, default=STAGES_DEFAULT)
    ap.add_argument("--gran", type=int, default=GRAN)
    # ---- down 臂（可选；见 §5）------------------------------------------------
    # 默认**只出 up**（保持既有 REGEN 手册逐字不变）。`--down` 追加 dn 的三件产出；
    # `--skip-up` 单独出 dn（重生成 dn 时不必重写 up 的 dump / sha）。
    ap.add_argument("--down", action="store_true",
                    help="also emit the routed-down artifacts (moe_bs_dn_tl.*)")
    ap.add_argument("--skip-up", action="store_true", help="with --down: skip the up artifacts")
    ap.add_argument("--kpad", type=int, default=K_DN_PAD_DEFAULT,
                    help="down 的 K 轴 pad 宽度（默认 384 = 3*BK；320 不整除 128 ⇒ 不可用）")
    ap.add_argument("--dn-bm", type=int, default=BM_DEFAULT)
    ap.add_argument("--dn-bn", type=int, default=BN_DEFAULT)
    ap.add_argument("--dn-bk", type=int, default=BK_DEFAULT)
    ap.add_argument("--dn-stages", type=int, default=STAGES_DOWN_DEFAULT)
    args = ap.parse_args()

    reason = verify_blockscaled_fix()
    print(f"[vendor] tilelang fix verified: {reason}")

    if args.down:
        _gen_down(args, reason)
    if args.down and args.skip_up:
        return

    BM, BN, BK, ST = args.bm, args.bn, args.bk, args.stages
    gran = args.gran
    grid_x = N_UP // BN
    assert N_UP % BN == 0, f"N_UP={N_UP} must be divisible by BN={BN}"
    assert NP % grid_x == 0, f"NP={NP} must be divisible by grid.x={grid_x}"
    NH = NP // grid_x                     # 每个 N-tile 取自一个权重面的行数（64）
    sf_words = DIM // (gran * 4)          # 40

    kern = moe_bs_up(SEG_CAP, BM, NP, DIM, E, BN, BK, NH,
                     threads=THREADS, stages=ST, gran=gran)
    src = _dump(kern, f"{args.outdir}/moe_bs_up_tl.cu")
    host = _dump_host(kern, f"{args.outdir}/moe_bs_up_tl_host.cu")
    sha = _sha(src)
    with open(f"{args.outdir}/moe_bs_up_tl.banner", "w") as f:
        f.write(_banner("moe_bs_up_tl.cu", sha, len(src)))

    # smem 预算（A = e4m3、B = unpacked e2m1，两者都是 1 B/元素 ⇒ A/B 同名同量；
    # C_sh 与 TMA 缓冲共享生命周期，按峰值估）
    smem_ab = ST * (BM * BK + BN * BK)
    smem_sf = ST * (BM + BN) * 4
    smem_c = BM * BN * 4
    smem_total = smem_ab + smem_sf + smem_c
    host_src_note = (
        "moe_bs_up_tl_host.cu"
        if host
        else "MISSING - transcribe the CUtensorMap recipe by hand (see wiring §4)"
    )
    sig = [l for l in src.splitlines() if "main_kernel" in l and "__global__" in l]
    lines = [
        "# MoE tcgen05 block-scaled (A e4m3 / B e2m1, ue8m0) up-GEMM AOT config",
        "# GENERATED by gen_moe_bs_aot.py — the shim's constants are read off THIS file.",
        f"is_tcgen05_fix=verified-in-source ({reason})",
        f"E={E} DIM={DIM} INTER={INTER} NP={NP} N_UP={N_UP} SEG_CAP={SEG_CAP} "
        f"VERIFY_ROWS={VERIFY_ROWS} TOPK_MAX={TOPK_MAX}",
        f"BM={BM} BN={BN} BK={BK} NH={NH} threads={THREADS} stages={ST} gran={gran}",
        f"sf_words={sf_words} sf_period={gran * 4 // BK} k_iters={DIM // BK}",
        f"grid=({grid_x}, {SEG_CAP})",
        f"smem_bytes={smem_total} (ab {smem_ab} + sf {smem_sf} + c {smem_c})",
        f"raw_sha256(up)={sha}",
        f"up_cu_bytes={len(src)}",
        f"host_source={host_src_note}",
        "",
        "# ---- device signature (the shim's parameter order / types) ----",
        *sig,
        "",
        "# ---- tensormap operands (authoritative recipe = moe_bs_up_tl_host.cu) ----",
        "# kind=tma tensors, in the order TileLang will pass them:",
        "#   A     [M, K]            e4m3 (row stride K B = 5120, 1 B/value)  swizzle from the dump",
        "#   W1    [E, NP, K]        fp4 packed (row stride K/2 B, expert stride = measured block)",
        "#   W3    [E, NP, K]        same",
        "#   SFA   [sf_words*M]      uint32 (1-D)",
        "#   SFW1  [E, sf_words*NP]  uint32 (2-D: expert x word)",
        "#   SFW3  [E, sf_words*NP]  uint32 (2-D)",
        "#   Eid   [NSEG]            int32  (plain, no TMA)",
        "#   C     [M, 2*NP]         f32    (TMA store; row stride 2*NP*4 B)",
        "",
        "# ---- D2 (A=e4m3) audit values -- mechanically compare with the NEW dump ----",
        "#   idesc_blockscaled=144708608 (0x08A01400): a_format=0 E4M3, b_format=5 E2M1, K32, E8M0",
        "#     (was 144709248 / 0x08A01680 = a_format=5 E2M1 with the old packed-fp4 A)",
        "#   a_tma_bytes_per_k_iter=16384 (BM*BK*1 B; was 8192 = BM*BK/2 with the packed fp4 A)",
        "#   mma_template=tcgen05mma_blockscaled_ss<tl::DataType::kFloat8_e4m3,false>",
        "#   a_row_stride_bytes=5120 (= DIM, 1 B/value; was 2560 = DIM/2)",
        "",
        "# ---- host-side layout contract (frozen in the POOL, not in this dump) ----",
        f"#   w1 plane: [{NP}, {DIM // 2}] u8   row pitch {DIM // 2} B (continuous, 16B ok)",
        f"#   w3 plane: [{NP}, {DIM // 2}] u8   row pitch {DIM // 2} B",
        f"#   w1/w3 scale planes: [{NP}, {DIM // 32}] u8  row pitch {DIM // 32} B",
        f"#     -> pack_wsf output word[g*{NP} + row] = u32(src + row*{DIM // 32} + g*4)",
        f"#   packed SF pool: [E, {sf_words}*{NP}] uint32 = {sf_words * NP * 4} B/expert/plane",
        f"#   C column order: tile bx covers gate[{NH}*bx .. +{NH}) then up[{NH}*bx .. +{NH})",
    ]
    with open(f"{args.outdir}/moe_bs_tl_config.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
