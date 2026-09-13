#!/usr/bin/env python3
# gen_moe_bs_aot.py — 从 tcgen05 BLOCK-SCALED（mxfp4: e2m1 + ue8m0）MoE-up 原型
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
#    （`TL_DISABLE_TMA_LOWER`）**走不通**：blockscaled 的 A/B smem 必须是
#    `float4_e2m1_unpacked`（packed smem 静默错值，原型 §4.1），而
#    packed-global → unpacked-smem 这个「展开」只有 TMA 的 tensor 形式能做
#    （`copy_analysis.cc:539`）。⇒ host 必须 `cuTensorMapEncodeTiled`。
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
#   源：`xq4_r`（[m, dim/2] u8 打包 e2m1）与 `xsc4_r`（[m, dim/32] **f32** 标度）；
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


def moe_bs_up(NSEG, BM, NP_, K, E_, BN, BK, NH, threads=THREADS, stages=STAGES_DEFAULT,
              gran=GRAN):
    """grouped block-scaled fp4 (e2m1+ue8m0) up-GEMM，tcgen05 1-CTA，显式 async。

    grid (N_UP/BN, NSEG)；每个 CTA 认领一个 (expert 段, N-tile)。

    A   : [NSEG*BM, K]        float4_e2m1fn  packed，段内行已 gather + BM-padding
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
    assert BM % 64 == 0, "blockscaled tcgen05 has no M=16/32 atom (disable_ws path)"
    assert 2 * NP_ % BN == 0, "the N axis must tile by BN with no pad"
    assert NH == NP_ // (2 * NP_ // BN), "half-tile rows must partition the weight plane"
    sf_words = K // (gran * 4)
    sf_period = gran * 4 // BK
    k_iters = K // BK
    M = NSEG * BM
    GRID_X = 2 * NP_ // BN

    @tilelang.jit(pass_configs={"tl::disable_tma_lower": False})
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float4_e2m1fn),
             W1: T.Tensor((E_, NP_, K), T.float4_e2m1fn),
             W3: T.Tensor((E_, NP_, K), T.float4_e2m1fn),
             SFA: T.Tensor((sf_words * M,), T.uint32),
             SFW1: T.Tensor((E_, sf_words * NP_), T.uint32),
             SFW3: T.Tensor((E_, sf_words * NP_), T.uint32),
             Eid: T.Tensor((NSEG,), "int32"),
             C: T.Tensor((M, 2 * NP_), "float32")):
        with T.Kernel(GRID_X, NSEG, threads=threads) as (bx, by):
            # smem dtype 必须是 unpacked（packed 会静默错值，原型 §4.1）
            A_sh = T.alloc_shared((stages, BM, BK), T.float4_e2m1_unpacked)
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


def _dump(kern, path):
    src = kern.get_kernel_source()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default=".")
    ap.add_argument("--bm", type=int, default=BM_DEFAULT,
                    help="MMA M-tile；必须 %%64==0（64 是目标，128 是原型认证回退）")
    ap.add_argument("--bn", type=int, default=BN_DEFAULT)
    ap.add_argument("--bk", type=int, default=BK_DEFAULT)
    ap.add_argument("--stages", type=int, default=STAGES_DEFAULT)
    ap.add_argument("--gran", type=int, default=GRAN)
    args = ap.parse_args()

    reason = verify_blockscaled_fix()
    print(f"[vendor] tilelang fix verified: {reason}")

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

    # smem 预算（unpacked fp4 = 1 B/元素；C_sh 与 TMA 缓冲共享生命周期，按峰值估）
    smem_fp4 = ST * (BM * BK + BN * BK)
    smem_sf = ST * (BM + BN) * 4
    smem_c = BM * BN * 4
    smem_total = smem_fp4 + smem_sf + smem_c
    host_src_note = (
        "moe_bs_up_tl_host.cu"
        if host
        else "MISSING - transcribe the CUtensorMap recipe by hand (see wiring §4)"
    )
    sig = [l for l in src.splitlines() if "main_kernel" in l and "__global__" in l]
    lines = [
        "# MoE tcgen05 block-scaled (mxfp4: e2m1 + ue8m0) up-GEMM AOT config",
        "# GENERATED by gen_moe_bs_aot.py — the shim's constants are read off THIS file.",
        f"is_tcgen05_fix=verified-in-source ({reason})",
        f"E={E} DIM={DIM} INTER={INTER} NP={NP} N_UP={N_UP} SEG_CAP={SEG_CAP} "
        f"VERIFY_ROWS={VERIFY_ROWS} TOPK_MAX={TOPK_MAX}",
        f"BM={BM} BN={BN} BK={BK} NH={NH} threads={THREADS} stages={ST} gran={gran}",
        f"sf_words={sf_words} sf_period={gran * 4 // BK} k_iters={DIM // BK}",
        f"grid=({grid_x}, {SEG_CAP})",
        f"smem_bytes={smem_total} (fp4 {smem_fp4} + sf {smem_sf} + c {smem_c})",
        f"raw_sha256(up)={sha}",
        f"up_cu_bytes={len(src)}",
        f"host_source={host_src_note}",
        "",
        "# ---- device signature (the shim's parameter order / types) ----",
        *sig,
        "",
        "# ---- tensormap operands (authoritative recipe = moe_bs_up_tl_host.cu) ----",
        "# kind=tma tensors, in the order TileLang will pass them:",
        "#   A     [M, K]            fp4 packed (row stride K/2 B)  swizzle from the dump",
        "#   W1    [E, NP, K]        fp4 packed (row stride K/2 B, expert stride NP*K/2)",
        "#   W3    [E, NP, K]        same",
        "#   SFA   [sf_words*M]      uint32 (1-D)",
        "#   SFW1  [E, sf_words*NP]  uint32 (2-D: expert x word)",
        "#   SFW3  [E, sf_words*NP]  uint32 (2-D)",
        "#   Eid   [NSEG]            int32  (plain, no TMA)",
        "#   C     [M, 2*NP]         f32    (TMA store; row stride 2*NP*4 B)",
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
