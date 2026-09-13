#!/usr/bin/env python3
# gen_sh_exp_aot.py — 冻结 shared expert 的 TileLang fp8 MMA AOT CUDA 源码，供
# `kernels/cuda/tilelang_gen/sh_exp_shim.cu` 合入 ferrite 的 nvcc 构建。
#
# 设计：docs/agent/c5-sh-exp-tilelang-design.md（C5 shared expert TileLang 设计）。
# 口径模板：kernels/tilelang/gen_proj_shapes_aot.py（五形状 fp8 MMA 的认证配方 ——
# 裸指针 ABI + 运行期 m 谓词 + per-32 ue8m0 scale）。
#
# 用法（远端 B300，tilelang 0.1.14）：
#   /opt/dlami/nvme/dsv41_venv/bin/python gen_sh_exp_aot.py <outdir> [--bm 16 --bn 32 --stages 3 --ks 1]
# 产出（<outdir>）：
#   sh_exp_gu_tl.cu        # device：gate+up → swiglu+limit → fp8 quant（fused，KS==1）
#   sh_exp_dn_tl.cu        # device：w2 down GEMM + epi_add
#   sh_exp_tl_config.txt   # 冻结几何 + 参数签名 + smem（shim 的出处）
#   （--ks>1 备用档另出 sh_exp_gu_partial_tl.cu + sh_exp_gu_red_tl.cu，见 §KS）
#
# ⚠️ GENERATED — do not edit（生成物由本脚本产出，shim 是手写件）。
#
# =============================================================================
# 0. 本生成器与 gen_proj_shapes_aot.py 的相对差异（只有三处，设计 §④ 的口径）
# =============================================================================
# 1. **M 恒为 MPAD=16**（mma.m16n8k32 的 M 原子）。m ≤ 8 是运行期谓词：行 i >= m 的
#    A_sh 写 0（不读激活那一行），输出行同样带 `i < m` 谓词。⇒ 同一程序服务 m=1
#    （eager/draft）与 m≤8（verify），即设计要求的 (b′)「双侧同换」——row r of an
#    M-row launch == row r of the M=1 launch OF THIS PROGRAM。
# 2. **L1 带 swiglu+limit+fp8 quant epilogue**（gate 上界 / up 双侧 clamp → silu →
#    per-row amax over 本 CTA 的 32 个 inter 列 → fast_round_scale → e4m3 饱和）。
#    bN1=32 由 amax 语义反向决定：`swiglu_limit_q` 的一个 scale 块 = 32 个连续 inter 值，
#    一个 CTA 必须拥有整块才能一次成树。
# 3. **L2 的 out_stride 烘成 OS=dim=5120**（调用点事实：chain_dev.rs 的 out=moe_out_r，
#    行距 = dim）。
#
# =============================================================================
# 1. 为什么是**裸指针 ABI**而不是 TMA 描述符（设计 §④b / R6）
# =============================================================================
# 本条与 moe_bs 臂不同，**不要照抄 moe_bs_shim.cu 的 descriptor 做法**：
#   * moe_bs 的 B operand 是 **packed fp4**（e2m1），它的 smem 必须是
#     `float4_e2m1_unpacked`，而 packed-global → unpacked-smem 这条「展开」**只有 TMA
#     的 tensor 形式能做**（gen_moe_bs_aot.py 文件头 §0.4）⇒ 那个臂**必须**走描述符。
#   * 本臂 A 与 B **都是 fp8 e4m3**（1 B/元素，无 sub-byte 展开）⇒ TMA 在这里**没有**
#     它存在的唯一理由。裸指针 ABI（`TL_DISABLE_TMA_LOWER` + `TL_DISABLE_WARP_SPECIALIZED`）
#     正是本树**已认证**的 fp8 MMA 配方（gen_proj_shapes_aot.py 五形状），且设计 R6 把它
#     列为「不引入新 lowering 特性」的防 illegal-instruction 措施。
#   ⇒ 生成物不含任何 `__grid_constant__ const CUtensorMap`，shim 里**没有**
#     `cuTensorMapEncodeTiled`、没有 dlopen、没有常驻 scratch、没有 SetAttribute
#     （两份动态 smem 分别只有 7 680 B / 4 608 B，都在 48 KiB 默认上限之下）。
#
# =============================================================================
# 2. scale 语义（与 SIMT 臂逐式一致）
# =============================================================================
#   * 权重 scale：`WGS: (N1//32, K1//32)` —— 行块号 × k 块号。bN1 == 32 ⇒ 一个 CTA 的
#     32 个 inter 列恰好是**一条** scale 行 ⇒ 读 `WGS[bx, ko]`（与核内
#     `wg_scale + (i>>5)*nb_k + kb` 同形，只是 i>>5 退化成 bx）。
#   * 激活 scale：`ASC: (MPAD, K1//32)` f32，per-(row, 32)。
#   * 累加形 `C_l[i,j] += C_p[i,j] * ASC[i,ko] * cast(WGS[bx,ko], f32)` —— 把每 32-K 块的
#     scale 乘在**该块的 mma 部分和**上（与 proj 的 `C_l += C_p * ASC * WSC` 同式）。
#   * 输出 scale：per-(row, 32-inter) 的 f32，`AQS: (MPAD, N1//32)`。
#   ⚠️ `w2.scale` 的**行 pitch 是 9 B（未 pad！）**：`W2S: (N2//32, K2//32)` = (160, 9)，
#     生成物把 pitch=9 bake 进索引 ⇒ **shim 必须把实际 pitch 作为运行期参数校验**
#     （见 sh_exp_shim.cu 的 `w2sc_pitch` 门）。这是设计的硬约束 R3。
#
# =============================================================================
# 3. 数值契约（设计 R4）—— 不追 SIMT 逐位，但要求程序内确定
# =============================================================================
# epilogue **逐字照抄** `swiglu_limit_q_kernel` 的 clamp/silu/amax/round 表达式：
#   limit>0: gg = fminf(gg, limit); uu = fminf(fmaxf(uu, -limit), limit)
#   v = (gg / (1 + expf(-gg))) * uu
#   am = max_j |v_j|                       （本 CTA 的 32 个 inter 值）
#   sc = fmaxf(fast_round_scale(am, 1/448), 1e-30)
#   q  = fminf(fmaxf(v * (1/sc), -448), 448); e4m3 饱和
# `fast_round_scale`（dsv41_kernels.cu:113）用**位运算**在 TileLang 里逐位复刻（见
# `_fast_round_scale`），不是「乘个常数」的近似形 —— 它是 2^ceil(log2(am/448))，
# 少了那一步 ceil 就不是同一个 scale。

import argparse
import sys

import tilelang
import tilelang.language as T

# ---- 精度类型（与 SIMT 臂 / proj 臂同名字符串）-----------------------------
FP8 = "float8_e4m3fn"
E8M0 = "float8_e8m0fnu"

# ---- 裸指针 ABI 的两个 pass_configs（gen_proj_shapes_aot.py 的认证配方）-----
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

# ---- 冻结几何（= chain_dev.rs:16870/17893，weights.rs:236-356）-------------
N1, K1 = 288, 5120   # w1/w3: N = sh_il（TP8 未 pad）, K = dim
N2, K2 = 5120, 288   # w2   : N = dim, K = sh_il
OS2 = 5120           # L2 输出行 stride == dim（调用点事实）
MPAD = 16            # mma m16n8k32 的 M 原子（m ≤ 8 走运行期谓词）
BN = 32              # N 向 tile：== 一个 swiglu scale 块（amax 语义钉死）
THREADS = 128
NS = 3               # num_stages


# ---------------------------------------------------------------------------
# 表达式助手（两条都逐字镜像 SIMT 臂的 C 表达式）
# ---------------------------------------------------------------------------
def _fast_round_scale(amax, max_inv):
    """`fast_round_scale(amax, max_inv)`（dsv41_kernels.cu:113）的位级复刻。

        bits = float_as_uint(amax * max_inv);
        exp  = (bits >> 23) & 0xFF;  man = bits & 0x7FFFFF;
        e    = exp - 127 + (man != 0);
        return int_as_float((e + 127) << 23);      # 2^ceil(log2(amax*max_inv))
    """
    bits = T.reinterpret("uint32", amax * max_inv)
    exp = T.Cast("int32", (bits >> T.Cast("uint32", 23)) & T.Cast("uint32", 0xFF))
    man = bits & T.Cast("uint32", 0x7FFFFF)
    e = exp - 127 + T.if_then_else(man != T.Cast("uint32", 0), 1, 0)
    return T.reinterpret("float32", T.Cast("uint32", e + 127) << T.Cast("uint32", 23))


def _swiglu(g, u, limit):
    """`swiglu_limit_kernel` 的 clamp + silu，逐项同式（limit 是运行期 f32）。

        if (limit > 0) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        v = (g / (1 + expf(-g))) * u;
    `if_then_else` 对两个分支都求值但只取一支 —— 无副作用，数值上与 C 的 `if` 等价。
    """
    gc = T.if_then_else(limit > T.float32(0.0), T.min(g, limit), g)
    uc = T.if_then_else(limit > T.float32(0.0),
                        T.min(T.max(u, T.float32(0.0) - limit), limit), u)
    return (gc / (T.float32(1.0) + T.exp(T.float32(0.0) - gc))) * uc


def _gu_epilogue(Cg_l, Cu_l, AQ, AQS, limit, m, bx, bm, bn):
    """L1 的 swiglu + amax + fast_round_scale + e4m3 饱和（设计 §④ 的 epilogue）。

    写成一个小函数，KS==1 的 fused 臂与 KS>1 的 reduce 臂**共用同一段代码** ⇒ 两条路
    的数值契约不可能漂移。
    """
    V = T.alloc_fragment((bm, bn), "float32")
    am = T.alloc_fragment((bm,), "float32")
    for i, j in T.Parallel(bm, bn):
        V[i, j] = T.if_then_else(i < m, T.abs(_swiglu(Cg_l[i, j], Cu_l[i, j], limit)),
                                 T.float32(0.0))
    T.reduce_max(V, am, dim=1, clear=True)
    for i, j in T.Parallel(bm, bn):
        if i < m:
            sc = T.max(_fast_round_scale(am[i], T.float32(1.0) / T.float32(448.0)),
                       T.float32(1e-30))
            AQS[i, bx] = sc
            v = _swiglu(Cg_l[i, j], Cu_l[i, j], limit) * (T.float32(1.0) / sc)
            AQ[i, bx * bn + j] = T.cast(
                T.min(T.max(v, T.float32(-448.0)), T.float32(448.0)), FP8)


# =============================================================================
# L1 `sh_exp_gu` —— gate+up → swiglu+limit → fp8 quant（KS == 1 的 fused 形）
# =============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def sh_exp_gu(bm=MPAD, bn=BN, threads=THREADS, ns=NS):
    """grid (ceil(N1/bn),) = (9,)，tile bm×bn×32，num_stages=ns。

    A   : [bm, K1]            fp8 e4m3（行 i >= m 谓词 —— 不读激活那一行）
    ASC : [bm, K1//32]        f32      per-(row,32) 激活标度
    WG  : [N1, K1]            fp8 e4m3 gate 面（行距 K1）
    WGS : [N1//32, K1//32]    ue8m0    [9,160]，行 pitch 160 B
    WU  : [N1, K1]            fp8 e4m3 up 面
    WUS : [N1//32, K1//32]    ue8m0
    AQ  : [bm, N1]            fp8 e4m3 输出（行距 N1 = 288）
    AQS : [bm, N1//32]        f32      per-(row,32-inter) 输出标度
    limit, m : 运行期标量
    """
    assert N1 % bn == 0 and bn % 32 == 0, "bN1 必须整除 N1 且是 swiglu scale 块（32）的倍数"
    assert K1 % 32 == 0
    kit = K1 // 32   # 160 个 k-block（每块一个 ue8m0 字节）

    @T.prim_func
    def main(A: T.Tensor((bm, K1), FP8),
             ASC: T.Tensor((bm, K1 // 32), "float32"),
             WG: T.Tensor((N1, K1), FP8),
             WGS: T.Tensor((N1 // 32, K1 // 32), E8M0),
             WU: T.Tensor((N1, K1), FP8),
             WUS: T.Tensor((N1 // 32, K1 // 32), E8M0),
             AQ: T.Tensor((bm, N1), FP8),
             AQS: T.Tensor((bm, N1 // 32), "float32"),
             limit: T.float32, m: T.int32):
        with T.Kernel(T.ceildiv(N1, bn), threads=threads) as bx:
            A_sh = T.alloc_shared((bm, 32), FP8)
            WG_sh = T.alloc_shared((bn, 32), FP8)
            WU_sh = T.alloc_shared((bn, 32), FP8)
            Cg_p = T.alloc_fragment((bm, bn), "float32")
            Cu_p = T.alloc_fragment((bm, bn), "float32")
            Cg_l = T.alloc_fragment((bm, bn), "float32")
            Cu_l = T.alloc_fragment((bm, bn), "float32")
            T.clear(Cg_l)
            T.clear(Cu_l)
            for ko in T.Pipelined(kit, num_stages=ns):
                # 运行期 m 谓词：行 >= m 的 A_sh 写 0（不读激活那一行）。
                for i, j in T.Parallel(bm, 32):
                    if i < m:
                        A_sh[i, j] = A[i, ko * 32 + j]
                    else:
                        A_sh[i, j] = T.cast(0, FP8)
                # bN1 == 32 ⇒ 本 CTA 恰好拥有**一条** [9,160] 的 scale 行。
                T.copy(WG[bx * bn, ko * 32], WG_sh)
                T.copy(WU[bx * bn, ko * 32], WU_sh)
                T.gemm(A_sh, WG_sh, Cg_p, transpose_B=True, clear_accum=True)
                T.gemm(A_sh, WU_sh, Cu_p, transpose_B=True, clear_accum=True)
                # 每 32-K 块一个 ue8m0：乘在该块的 mma 部分和上（= proj 的同式）。
                for i, j in T.Parallel(bm, bn):
                    Cg_l[i, j] += Cg_p[i, j] * ASC[i, ko] * T.cast(WGS[bx, ko], "float32")
                    Cu_l[i, j] += Cu_p[i, j] * ASC[i, ko] * T.cast(WUS[bx, ko], "float32")
            _gu_epilogue(Cg_l, Cu_l, AQ, AQS, limit, m, bx, bm, bn)

    return main


# =============================================================================
# L2 `sh_exp_dn` —— w2 down GEMM + epi_add
# =============================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def sh_exp_dn(bm=MPAD, bn=BN, threads=THREADS, ns=NS):
    """grid (ceil(N2/bn),) = (160,)，tile bm×bn×32，num_stages=ns。

    AQ  : [bm, K2]          fp8 e4m3（L1 的输出；行 i >= m 谓词）
    AQS : [bm, K2//32]      f32      per-(row,32) 激活标度（L1 的输出）
    W2  : [N2, K2]          fp8 e4m3（行距 K2 = 288）
    W2S : [N2//32, K2//32]  ue8m0    [160, 9] —— **行 pitch 9 B（未 pad，硬约束）**
    OUT : [bm, OS2]         f32      行 stride = OS2 = dim
    epi_add, m : 运行期标量（epi_add != 0 ⇒ OUT += C，否则 OUT = C）
    """
    assert N2 % bn == 0
    assert K2 % 32 == 0
    kit = K2 // 32   # 9 个 k-block

    @T.prim_func
    def main(AQ: T.Tensor((bm, K2), FP8),
             AQS: T.Tensor((bm, K2 // 32), "float32"),
             W2: T.Tensor((N2, K2), FP8),
             W2S: T.Tensor((N2 // 32, K2 // 32), E8M0),
             OUT: T.Tensor((bm, OS2), "float32"),
             epi_add: T.int32, m: T.int32):
        with T.Kernel(T.ceildiv(N2, bn), threads=threads) as bx:
            A_sh = T.alloc_shared((bm, 32), FP8)
            W_sh = T.alloc_shared((bn, 32), FP8)
            C_l = T.alloc_fragment((bm, bn), "float32")
            C_p = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(kit, num_stages=ns):
                for i, j in T.Parallel(bm, 32):
                    if i < m:
                        A_sh[i, j] = AQ[i, ko * 32 + j]
                    else:
                        A_sh[i, j] = T.cast(0, FP8)
                T.copy(W2[bx * bn, ko * 32], W_sh)
                T.gemm(A_sh, W_sh, C_p, transpose_B=True, clear_accum=True)
                # W2S[bx, ko]：bx == 行块号（bN2 == 32），行 pitch 9（bake）。
                for i, j in T.Parallel(bm, bn):
                    C_l[i, j] += C_p[i, j] * AQS[i, ko] * T.cast(W2S[bx, ko], "float32")
            # epi_add 的 read-modify-write：与原 `add_inplace_raw(out, sh_out)` 的
            # 操作数对相同（out[...] + C），逐元素同一次加法。
            for i, j in T.Parallel(bm, bn):
                if i < m:
                    OUT[i, bx * bn + j] = T.if_then_else(
                        epi_add != 0, OUT[i, bx * bn + j], T.float32(0.0)) + C_l[i, j]

    return main


# =============================================================================
# KS > 1 备用档（设计 §③ 「几何对比与 K-split 备用档」）
# =============================================================================
# v1 只出 KS=1（2 launch / 层）。KS>1 把 L1 的 K 轴切成 ks 段：partial（grid (9,ks)）
# 写 f32 部分和 PG/PU，reduce 求和后**做同一个 swiglu+quant epilogue**（与 KS==1 的
# fused 臂共用 `_gu_epilogue`）。触发器 = 微基准实测 L1 > 2× 模型值。
@tilelang.jit(pass_configs=PASS_CONFIGS)
def sh_exp_gu_partial(ks, bm=MPAD, bn=BN, threads=THREADS, ns=NS):
    """KS>1 的 L1 分片：grid (N1/bn, ks)，每块算一段 K 的 f32 partial 写 PG/PU[kp]。"""
    kc = K1 // ks

    @T.prim_func
    def main(A: T.Tensor((bm, K1), FP8),
             ASC: T.Tensor((bm, K1 // 32), "float32"),
             WG: T.Tensor((N1, K1), FP8),
             WGS: T.Tensor((N1 // 32, K1 // 32), E8M0),
             WU: T.Tensor((N1, K1), FP8),
             WUS: T.Tensor((N1 // 32, K1 // 32), E8M0),
             PG: T.Tensor((ks, bm, N1), "float32"),
             PU: T.Tensor((ks, bm, N1), "float32"),
             m: T.int32):
        with T.Kernel(T.ceildiv(N1, bn), ks, threads=threads) as (bx, kp):
            A_sh = T.alloc_shared((bm, 32), FP8)
            WG_sh = T.alloc_shared((bn, 32), FP8)
            WU_sh = T.alloc_shared((bn, 32), FP8)
            Cg_p = T.alloc_fragment((bm, bn), "float32")
            Cu_p = T.alloc_fragment((bm, bn), "float32")
            Cg_l = T.alloc_fragment((bm, bn), "float32")
            Cu_l = T.alloc_fragment((bm, bn), "float32")
            T.clear(Cg_l)
            T.clear(Cu_l)
            for ko in T.Pipelined(kc // 32, num_stages=ns):
                gko = kp * (kc // 32) + ko
                for i, j in T.Parallel(bm, 32):
                    if i < m:
                        A_sh[i, j] = A[i, gko * 32 + j]
                    else:
                        A_sh[i, j] = T.cast(0, FP8)
                T.copy(WG[bx * bn, gko * 32], WG_sh)
                T.copy(WU[bx * bn, gko * 32], WU_sh)
                T.gemm(A_sh, WG_sh, Cg_p, transpose_B=True, clear_accum=True)
                T.gemm(A_sh, WU_sh, Cu_p, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(bm, bn):
                    Cg_l[i, j] += Cg_p[i, j] * ASC[i, gko] * T.cast(WGS[bx, gko], "float32")
                    Cu_l[i, j] += Cu_p[i, j] * ASC[i, gko] * T.cast(WUS[bx, gko], "float32")
            T.copy(Cg_l, PG[kp, 0, bx * bn])
            T.copy(Cu_l, PU[kp, 0, bx * bn])

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def sh_exp_gu_reduce(ks, bm=MPAD, bn=BN, threads=THREADS):
    """KS>1 的 L1 归约：按 kp 升序求和 ks 个 partial（确定性），做 swiglu+quant。"""
    idx = tuple(range(ks))

    @T.prim_func
    def main(PG: T.Tensor((ks, bm, N1), "float32"),
             PU: T.Tensor((ks, bm, N1), "float32"),
             AQ: T.Tensor((bm, N1), FP8),
             AQS: T.Tensor((bm, N1 // 32), "float32"),
             limit: T.float32, m: T.int32):
        with T.Kernel(T.ceildiv(N1, bn), threads=threads) as bx:
            Cg_l = T.alloc_fragment((bm, bn), "float32")
            Cu_l = T.alloc_fragment((bm, bn), "float32")
            for i, j in T.Parallel(bm, bn):
                Cg_l[i, j] = sum(PG[kq, i, bx * bn + j] for kq in idx)
                Cu_l[i, j] = sum(PU[kq, i, bx * bn + j] for kq in idx)
            _gu_epilogue(Cg_l, Cu_l, AQ, AQS, limit, m, bx, bm, bn)

    return main


# =============================================================================
# dump / config
# =============================================================================
def _dump(kern, path):
    src = kern.get_kernel_source()
    # POST-PROCESS (2026-09-14): inject tcgen05.relinquish_alloc_permit after the
    # tmem_allocate calls — same fix as gen_moe_bs_aot.py. TileLang 0.1.14's
    # codegen omits this, violating the PTX ISA precondition for tcgen05.dealloc.
    _RELINQUISH = ('    asm volatile('
                   '"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"'
                   ' ::: "memory");')
    _NEEDLE = 'tl::tmem_allocate'
    if _NEEDLE in src and 'relinquish_alloc_permit' not in src:
        lines = src.split('\n')
        out = []
        in_alloc_block = False
        for ln in lines:
            out.append(ln)
            if _NEEDLE in ln:
                in_alloc_block = True
            elif in_alloc_block and ln.rstrip() == '  }':
                out.pop()
                out.append(_RELINQUISH)
                out.append('  }')
                in_alloc_block = False
        src = '\n'.join(out)
    with open(path, "w") as f:
        f.write(src)
    return src


def _sig_lines(src):
    """取生成物的 `__global__` 签名行 —— shim 的参数序/类型的权威出处。"""
    return [l for l in src.splitlines() if "main_kernel" in l and "__global__" in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default=".")
    ap.add_argument("--bm", type=int, default=MPAD, help="mma M-tile（默认 16 = m16 原子）")
    ap.add_argument("--bn", type=int, default=BN, help="N-tile（默认 32 = 一个 swiglu scale 块）")
    ap.add_argument("--stages", type=int, default=NS, help="num_stages（默认 3）")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--ks", type=int, default=1, help="K-split 备用档（默认 1 = v1 认证档）")
    args = ap.parse_args()
    bm, bn, ns, th, ks = args.bm, args.bn, args.stages, args.threads, args.ks
    outdir = args.outdir

    if ks < 1 or (K1 // ks) % 32 != 0:
        sys.exit(f"--ks={ks} 非法：K1={K1} 必须被 ks 整除，且每段是 32 的倍数")

    lines = [
        "# shared-expert TileLang fp8 MMA AOT config "
        "(GENERATED by gen_sh_exp_aot.py)",
        f"# dim(model)={K1} sh_il(N1)={N1} w2=[{N2},{K2}] OS2={OS2}",
        f"MPAD={bm} BN={bn} THREADS={th} NS={ns} KS={ks}",
        "# ABI=raw-pointer (TL_DISABLE_TMA_LOWER + TL_DISABLE_WARP_SPECIALIZED) "
        "— NOT the moe_bs TMA descriptor form",
        "# op: name grid block smem_bytes blk_m blk_n blk_k",
    ]

    if ks == 1:
        k_gu = sh_exp_gu(bm, bn, th, ns)
        s_gu = _dump(k_gu, f"{outdir}/sh_exp_gu_tl.cu")
        smem_gu = ns * (bm * 32 + 2 * bn * 32)   # A_sh + WG_sh + WU_sh
        lines.append(f"gu grid=({N1 // bn},) block={th} smem_bytes={smem_gu} "
                     f"blk_m={bm} blk_n={bn} blk_k=32")
        lines += ["--- sh_exp_gu_tl.cu signature (main_kernel) ---", *_sig_lines(s_gu)]
        lines.append(f"gu_cu_bytes={len(s_gu)}")
    else:
        kp = sh_exp_gu_partial(ks, bm, bn, th, ns)
        kr = sh_exp_gu_reduce(ks, bm, bn, th)
        sp = _dump(kp, f"{outdir}/sh_exp_gu_partial_tl.cu")
        sr = _dump(kr, f"{outdir}/sh_exp_gu_red_tl.cu")
        smem_p = ns * (bm * 32 + 2 * bn * 32)
        lines.append(f"gu_partial grid=({N1 // bn}, {ks}) block={th} smem_bytes={smem_p} "
                     f"blk_m={bm} blk_n={bn} blk_k=32")
        lines.append(f"gu_red grid=({N1 // bn},) block={th} smem_bytes=0 "
                     f"(P 常驻 {2 * ks * bm * N1 * 4} B)")
        lines += ["--- sh_exp_gu_partial_tl.cu signature (main_kernel) ---", *_sig_lines(sp)]
        lines += ["--- sh_exp_gu_red_tl.cu signature (main_kernel) ---", *_sig_lines(sr)]
        lines.append(f"gu_partial_cu_bytes={len(sp)} gu_red_cu_bytes={len(sr)}")

    k_dn = sh_exp_dn(bm, bn, th, ns)
    s_dn = _dump(k_dn, f"{outdir}/sh_exp_dn_tl.cu")
    smem_dn = ns * (bm * 32 + bn * 32)           # A_sh + W_sh
    lines.append(f"dn grid=({N2 // bn},) block={th} smem_bytes={smem_dn} "
                 f"blk_m={bm} blk_n={bn} blk_k=32")
    lines += ["--- sh_exp_dn_tl.cu signature (main_kernel) ---", *_sig_lines(s_dn)]
    lines.append(f"dn_cu_bytes={len(s_dn)}")

    lines += [
        "",
        "--- host-side layout contract (frozen in the POOL, not in this dump) ---",
        f"w1/w3 plane   : [{N1}, {K1}] u8  row pitch {K1} B",
        f"w1/w3.scale   : [{N1 // 32}, {K1 // 32}] u8  row pitch {K1 // 32} B (=160, %16==0)",
        f"w2 plane      : [{N2}, {K2}] u8  row pitch {K2} B",
        f"w2.scale      : [{N2 // 32}, {K2 // 32}] u8  row pitch {K2 // 32} B (=9, "
        "UNPADDED — hard constraint, shim gates w2sc_pitch)",
        f"aq (L1 out)   : [{bm}, {N1}] u8  row pitch {N1} B (=288, %16==0)",
        f"aqsc (L1 out) : [{bm}, {N1 // 32}] f32  row pitch {N1 // 32} B (=9)",
        f"out (L2)      : [{bm}, {OS2}] f32 row pitch {OS2 * 4} B",
    ]
    with open(f"{outdir}/sh_exp_tl_config.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
