#!/usr/bin/env python3
# proj_fp8_tilelang.py — TileLang 原型：ferrite 投影族的 fp8 e4m3 投影 GEMM（tensor core 路线）。
#
# STATUS: 原型 / 未接线。**不在 build.sh 里、不被 Rust 引用**。见
#   docs/agent/tilelang-proj-proto.md（实测数字、数值形态、AOT 产物、集成路径）。
#
# 目标：让 M=6（verify 行批）的成本 ≈ M=1（eager/draft），镜像 SGLang 的 1.2-1.3x 机制。
# 老 kernel `gemm_fp8_mrows<M>` 是 M-in-register 串行，M=6/M=1 实测 3.0-5.7x；
# 本原型把 M 缩进 mma 的 tile 维度（pad 到 16），M=6/M=1 实测 0.98-1.03。
#
# 布局（以 ferrite `gemm_fp8_mrows_kernel` 的 ABI 为准，NOT per-128）：
#   a        [m, k]      fp8 e4m3      （本原型 pad 到 MPAD=16 —— mma m16n8k32 的 M 约束）
#   a_scale  [m, k/32]   f32
#   w        [n, k]      fp8 e4m3
#   w_scale  [n/32,k/32] ue8m0（块 32x32）
#
# 数值路线：route A（原生 fp8 mma + per-32-K-block scale 作用在 mma 输出上），
# 镜 `gemm_fp8_swapab_kernel:800-813`。实测比 f32 SIMT 参考更接近真值（见 doc §4）。
# route B（dequant bf16）数值不可用（mean_rel ~1.7e-2），本文件不提供。
#
# 依赖：tilelang 0.1.14 / torch / B300(sm_103a)。

import torch
import tilelang
import tilelang.language as T

FP8 = "float8_e4m3fn"
E8M0 = "float8_e8m0fnu"
MPAD = 16  # mma m16n8k32 要求 M 能被 16 整除；激活不足则 pad


@tilelang.jit
def proj_fp8_partial(N, K, bN, ks, threads=128, ns=3):
    """K-split 分片：grid (N/bN, ks)，每块算一段 K 的 partial，写入 P[kp]。

    ks 规则（mirror 设计文档 §3.3）：把块数从 N/bN 抬到 (N/bN)*ks 以填满 SM。
    """
    Kc = K // ks

    @T.prim_func
    def main(A: T.Tensor((MPAD, K), FP8),
             ASC: T.Tensor((MPAD, K // 32), "float32"),
             W: T.Tensor((N, K), FP8),
             WSC: T.Tensor((N // 32, K // 32), E8M0),
             P: T.Tensor((ks, MPAD, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), ks, threads=threads) as (bx, kp):
            A_sh = T.alloc_shared((MPAD, 32), FP8)
            W_sh = T.alloc_shared((bN, 32), FP8)
            C_l = T.alloc_fragment((MPAD, bN), "float32")
            C_p = T.alloc_fragment((MPAD, bN), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(Kc // 32, num_stages=ns):
                gko = kp * (Kc // 32) + ko
                T.copy(A[0, gko * 32], A_sh)
                T.copy(W[bx * bN, gko * 32], W_sh)
                # raw fp8 乘积和（mma），再按 32-K 块的 scale 累加。
                # 注意：scale 直读全局（不要自建共享缓冲——pipelined loop 不给它多缓冲，
                # 会产生跨 stage 覆盖的静默错值；见 doc §3.1 踩坑）。
                T.gemm(A_sh, W_sh, C_p, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(MPAD, bN):
                    C_l[i, j] += C_p[i, j] * ASC[i, gko] * T.cast(
                        WSC[bx * (bN // 32) + j // 32, gko], "float32")
            T.copy(C_l, P[kp, 0, bx * bN])

    return main


@tilelang.jit
def proj_fp8_reduce(ks, N, bN=256):
    """确定性归约：按 kp 升序求和 ks 个 partial。"""

    IDX = tuple(range(ks))

    @T.prim_func
    def main(P: T.Tensor((ks, MPAD, N), "float32"),
             C: T.Tensor((MPAD, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), threads=256) as bx:
            for i, j in T.Parallel(MPAD, bN):
                C[i, bx * bN + j] = sum([P[kq, i, bx * bN + j] for kq in IDX])

    return main


# ---------------------------------------------------------------- self-test
def _reference(A, ASC, W, WSC):
    af = A.float()
    wf = W.float()
    asp = ASC.repeat_interleave(32, dim=1)
    wsp = WSC.float().repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    return (af * asp) @ (wf * wsp).T


def main():
    torch.manual_seed(0)
    shapes = [("wkv", 512, 5120), ("wq_a", 1280, 5120),
              ("wq_b", 4096, 1280), ("wo_b", 5120, 1024)]
    bN, ks, ns, thr = 128, 8, 3, 128
    for name, N, K in shapes:
        m = 6
        A = torch.zeros(MPAD, K, device="cuda")
        A[:m] = torch.rand(m, K, device="cuda") * 2 - 1
        A = A.to(torch.float8_e4m3fn)
        W = (torch.rand(N, K, device="cuda") * 2 - 1).to(torch.float8_e4m3fn)
        ASC = torch.zeros(MPAD, K // 32, device="cuda")
        ASC[:m] = torch.rand(m, K // 32, device="cuda") * 0.5 + 0.75
        WSC = torch.randint(123, 131, (N // 32, K // 32), device="cuda") \
            .to(torch.uint8).view(torch.float8_e8m0fnu)

        kp = proj_fp8_partial(N, K, bN, ks, thr, ns)
        kr = proj_fp8_reduce(ks, N)
        P = torch.zeros(ks, MPAD, N, device="cuda")
        C = torch.zeros(MPAD, N, device="cuda")
        kp(A, ASC, W, WSC, P)
        kr(P, C)
        ref = _reference(A, ASC, W, WSC)
        err = (C[:m] - ref[:m]).abs().max().item()
        print(f"{name:6} n={N:<5} k={K:<5} max_abs_err={err:.3e}")


if __name__ == "__main__":
    main()
