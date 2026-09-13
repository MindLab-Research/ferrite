#!/usr/bin/env python3
# gen_proj_shapes_aot.py — 从五形状 TileLang 原型（proj_fp8_tilelang_phase2.py 的 route A）
# 冻结出**四个剩余形状**的 AOT CUDA 源码，供 `kernels/cuda/tilelang_gen/` 合入 ferrite 的
# nvcc 构建。wkv（第一阶段）不在本脚本内 —— 它已冻结，保持逐位不动。
#
# 用法（远端 B300，tilelang 0.1.14）：
#   /opt/dlami/nvme/dsv41_venv/bin/python gen_proj_shapes_aot.py <outdir>
# 产出（<outdir>）：
#   wq_a_partial_tl.cu / wq_a_reduce_tl.cu     # wq_a [n=1280, k=5120]
#   wq_b_partial_tl.cu / wq_b_reduce_tl.cu     # wq_b [n=4096, k=1280]
#   wo_b_partial_tl.cu / wo_b_reduce_tl.cu     # wo_b [n=5120, k=1024]
#   wo_a_g1_partial_tl.cu / wo_a_g1_reduce_tl.cu   # wo_a 分组 G=1  (verify@TP8)
#   wo_a_g8_partial_tl.cu / wo_a_g8_reduce_tl.cu   # wo_a 分组 G=8  (TP1)
#   proj_shapes_tl_config.txt                  # 冻结几何 + 签名（shim 的出处）
#
# ⚠️ GENERATED — do not edit. 重生成见 kernels/cuda/tilelang_gen/PROVENANCE.md §8。
#
# 口径与原型 / 第一阶段**逐行一致**（route A：fp8 mma + per-32 ue8m0 scale，K-split，
# epilogue 直读全局 scale；运行期 m 谓词；裸指针 ABI）：
#   a        [MPAD=16, k]       fp8 e4m3   （运行期 m 谓词：行 >= m 的 A_sh 写 0，不读 a 那一行）
#   a_scale  [MPAD, k/32]       f32
#   w        [n, k]             fp8 e4m3
#   w_scale  [n/32, k/32]       ue8m0
#   P        [ks, MPAD, n]      f32        （分片 partial，shim 持有的常驻 scratch）
#   C        [MPAD, OS]         f32        （归约输出，行 stride = OS —— 见下）
#
# ============================================================================
# 与第一阶段的唯一生成差异：输出行 stride 由 `OS` 烘进来（不是 n）
# ============================================================================
# ferrite 的 fp8 投影 ABI 里 `out_stride` 是本形状调用点真实的行距：
#   * wq_a 的 verify/eager 站点   out_stride = q_lora_rank = n            （== n）
#   * wo_b 的 verify/eager 站点   out_stride = dim        = n            （== n）
#   * wq_b **verify/eager** 站点 out_stride = nh*head_dim = 32768 ≠ nlh*hd = 4096 = n
#       （wq_b 是 ColumnParallel：本 rank 只写每行的前 nlh*hd 个元素，行距是整行宽）
#   * wo_a 分组站点                              out_stride = ol_total = 8192 ≠ G*n（G=1 时 1024）
#       （wo_a 是 ColumnParallel：本 rank 的 nlg 个组写进全局宽行，行距是 groups*o_lora）
# wkv 第一阶段能直接吃 out_stride == n，是因为它的站点恰好如此。这里把真实的 OS 作为
# **编译期常量**烘进归约内核的输出声明（`C: T.Tensor((MPAD, OS))`），保持 store 仍是
# 256-bit 向量化（`tl::store_global_256`），且**不需要第三发 launch 做 strided copy**。
# shim 的形状门据此要求 `out_stride == OS`（否则 decline，调用方保持老 kernel）。
#
# ABI 裸指针化：`pass_configs={TL_DISABLE_TMA_LOWER:1, TL_DISABLE_WARP_SPECIALIZED:1}`。

import sys

import tilelang
import tilelang.language as T

FP8 = "float8_e4m3fn"
E8M0 = "float8_e8m0fnu"
MPAD = 16  # mma m16n8k32 要求 M 能被 16 整除

# 裸指针 ABI 的两个 pass_configs（设计文档 §1.3 实验 B）。
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

BN = 128       # partial 的 N 向 tile
KS = 8         # K-split 分片数（镜像 tensorcore-proj-design.md §3.3 的 ks 规则）
THREADS = 128  # partial block
NS = 3         # num_stages
RED_BN = 256   # reduce 的 N 向 tile

# ---------------------------------------------------------------------------
# 稠密四形状（wkv 不动；这里只生成三个新增量）
#   (name, N, K, OS)
# OS = 调用点真实的 `out_stride`（见文件头）。wq_b 的 OS 取 verify/eager 站点
#      （nh*head_dim = 64*512 = 32768）；indexer 站点（idx_nh*idx_hd = 4096 == n）
#      的 out_stride 不同 ⇒ 它会被 shim 的形状门 decline，保持老 kernel。
#        dim=5120 · nh=64 · hd=512 · ql=1280（config.rs production）
# ---------------------------------------------------------------------------
DENSE = [
    ("wq_a", 1280, 5120, 1280),      # OS = q_lora_rank
    ("wq_b", 4096, 1280, 32768),     # OS = nh*head_dim
    ("wo_b", 5120, 1024, 5120),      # OS = dim
]

# ---------------------------------------------------------------------------
# wo_a 分组（代码事实：dsv41_kernels.cu:8165 / chain_dev.rs:13572）
#   (name, G, N, K, ASTRIDE, OS)
#   N = o_lora_rank = 1024；K = hpg*head_dim = (64/8)*512 = 4096
#   ASTRIDE = nlh*head_dim = (nh/world)*head_dim
#   OS = ol_total = groups*o_lora_rank = 8*1024 = 8192（与 world 无关）
#   G = nlg = o_groups/world —— verify@TP8 → 1；TP1 → 8
# ---------------------------------------------------------------------------
WOA = [
    ("wo_a_g1", 1, 1024, 4096, 4096, 8192),
    ("wo_a_g8", 8, 1024, 4096, 32768, 8192),
]


@tilelang.jit(pass_configs=PASS_CONFIGS)
def proj_fp8_partial(N, K, bN, ks, threads=128, ns=3):
    """K-split 分片：grid (N/bN, ks)，每块算一段 K 的 partial，写入 P[kp]。"""
    Kc = K // ks

    @T.prim_func
    def main(A: T.Tensor((MPAD, K), FP8),
             ASC: T.Tensor((MPAD, K // 32), "float32"),
             W: T.Tensor((N, K), FP8),
             WSC: T.Tensor((N // 32, K // 32), E8M0),
             P: T.Tensor((ks, MPAD, N), "float32"),
             m: T.int32):
        with T.Kernel(T.ceildiv(N, bN), ks, threads=threads) as (bx, kp):
            A_sh = T.alloc_shared((MPAD, 32), FP8)
            W_sh = T.alloc_shared((bN, 32), FP8)
            C_l = T.alloc_fragment((MPAD, bN), "float32")
            C_p = T.alloc_fragment((MPAD, bN), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(Kc // 32, num_stages=ns):
                gko = kp * (Kc // 32) + ko
                # 运行期 m 谓词：行 >= m 的 A_sh 写 0（不读激活那一行）。
                for i, j in T.Parallel(MPAD, 32):
                    if i < m:
                        A_sh[i, j] = A[i, gko * 32 + j]
                    else:
                        A_sh[i, j] = T.cast(0, FP8)
                T.copy(W[bx * bN, gko * 32], W_sh)
                # raw fp8 mma -> 按 32-K 块的 scale 累加；scale 直读全局（不进 smem）。
                T.gemm(A_sh, W_sh, C_p, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(MPAD, bN):
                    C_l[i, j] += C_p[i, j] * ASC[i, gko] * T.cast(
                        WSC[bx * (bN // 32) + j // 32, gko], "float32")
            T.copy(C_l, P[kp, 0, bx * bN])

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def proj_fp8_reduce(ks, N, OS, bN=256):
    """确定性归约：按 kp 升序求和 ks 个 partial，只写前 m 行（行 stride = OS）。"""
    IDX = tuple(range(ks))

    @T.prim_func
    def main(P: T.Tensor((ks, MPAD, N), "float32"),
             C: T.Tensor((MPAD, OS), "float32"),
             m: T.int32):
        with T.Kernel(T.ceildiv(N, bN), threads=256) as bx:
            for i, j in T.Parallel(MPAD, bN):
                if i < m:
                    C[i, bx * bN + j] = sum([P[kq, i, bx * bN + j] for kq in IDX])

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def wo_a_grouped_partial(G, N, K, ASTRIDE, bN, ks, threads=128, ns=3):
    """分组 K-split 分片：grid (N/bN, G, ks)。组维走 blockIdx.y（每 block 单组）。"""
    Kc = K // ks
    KELEM = K // 32

    @T.prim_func
    def main(A: T.Tensor((MPAD, ASTRIDE), FP8),
             ASC: T.Tensor((MPAD, ASTRIDE // 32), "float32"),
             W: T.Tensor((G, N, K), FP8),
             WSC: T.Tensor((G, N // 32, K // 32), E8M0),
             P: T.Tensor((ks, G, MPAD, N), "float32"),
             m: T.int32):
        with T.Kernel(T.ceildiv(N, bN), G, ks, threads=threads) as (bx, g, kp):
            A_sh = T.alloc_shared((MPAD, 32), FP8)
            W_sh = T.alloc_shared((bN, 32), FP8)
            C_l = T.alloc_fragment((MPAD, bN), "float32")
            C_p = T.alloc_fragment((MPAD, bN), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(Kc // 32, num_stages=ns):
                gko = kp * (Kc // 32) + ko
                # 第 g 组的激活段在 +g*K 字节（block-diagonal）；行距是 ASTRIDE（≠ K）。
                # 运行期 m 谓词（与稠密同）。
                for i, j in T.Parallel(MPAD, 32):
                    if i < m:
                        A_sh[i, j] = A[i, g * K + gko * 32 + j]
                    else:
                        A_sh[i, j] = T.cast(0, FP8)
                T.copy(W[g, bx * bN, gko * 32], W_sh)
                T.gemm(A_sh, W_sh, C_p, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(MPAD, bN):
                    C_l[i, j] += C_p[i, j] * ASC[i, g * KELEM + gko] * T.cast(
                        WSC[g, bx * (bN // 32) + j // 32, gko], "float32")
            T.copy(C_l, P[kp, g, 0, bx * bN])

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def wo_a_grouped_reduce(ks, G, N, OS, bN=256):
    """分组确定性归约：组 g 的 ks 个 partial 归到 OUT[:, g*N + ...]（行 stride = OS）。"""
    IDX = tuple(range(ks))

    @T.prim_func
    def main(P: T.Tensor((ks, G, MPAD, N), "float32"),
             OUT: T.Tensor((MPAD, OS), "float32"),
             m: T.int32):
        with T.Kernel(T.ceildiv(N, bN), G, threads=256) as (bx, g):
            for i, j in T.Parallel(MPAD, bN):
                if i < m:
                    OUT[i, g * N + bx * bN + j] = sum(
                        [P[kq, g, i, bx * bN + j] for kq in IDX])

    return main


def _dump(kern, path):
    src = kern.get_kernel_source()
    with open(path, "w") as f:
        f.write(src)
    return src


def _emit(outdir, name, kp, kr, lines):
    sp = _dump(kp, f"{outdir}/{name}_partial_tl.cu")
    sr = _dump(kr, f"{outdir}/{name}_reduce_tl.cu")
    lines.append(f"--- {name}_partial_tl.cu signature ---")
    lines += [l for l in sp.splitlines() if "main_kernel" in l and "__global__" in l
              and "launch_bounds" in l]
    lines.append(f"--- {name}_reduce_tl.cu signature ---")
    lines += [l for l in sr.splitlines() if "main_kernel" in l and "__global__" in l
              and "launch_bounds" in l]
    lines.append(f"{name}_partial_cu_bytes={len(sp)}")
    lines.append(f"{name}_reduce_cu_bytes={len(sr)}")
    return lines


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    lines = [
        "# proj-shapes TileLang AOT config (GENERATED by gen_proj_shapes_aot.py)",
        f"MPAD={MPAD} BN={BN} KS={KS} THREADS={THREADS} NS={NS} RED_BN={RED_BN}",
        f"# dense: name N K OS(=<out_stride at the call site>) "
        f"partial_grid=(N/BN,KS) reduce_grid=(N/RED_BN,)",
    ]
    for name, N, K, OS in DENSE:
        kp = proj_fp8_partial(N, K, BN, KS, THREADS, NS)
        kr = proj_fp8_reduce(KS, N, OS, RED_BN)
        lines.append(f"dense {name} N={N} K={K} OS={OS} "
                     f"partial_grid=({(N + BN - 1) // BN}, {KS}) "
                     f"partial_block={THREADS} "
                     f"partial_smem_bytes={NS * MPAD * 32 + NS * BN * 32} "
                     f"reduce_grid=({(N + RED_BN - 1) // RED_BN},)")
        _emit(outdir, name, kp, kr, lines)

    lines.append("# grouped wo_a: G groups in grid.y; a_stride = ASTRIDE; out row stride = OS")
    for name, G, N, K, ASTRIDE, OS in WOA:
        kp = wo_a_grouped_partial(G, N, K, ASTRIDE, BN, KS, THREADS, NS)
        kr = wo_a_grouped_reduce(KS, G, N, OS, RED_BN)
        lines.append(f"grouped {name} G={G} N={N} K={K} ASTRIDE={ASTRIDE} OS={OS} "
                     f"partial_grid=({(N + BN - 1) // BN}, {G}, {KS}) "
                     f"partial_block={THREADS} "
                     f"partial_smem_bytes={NS * MPAD * 32 + NS * BN * 32} "
                     f"reduce_grid=({(N + RED_BN - 1) // RED_BN}, {G})")
        _emit(outdir, name, kp, kr, lines)

    with open(f"{outdir}/proj_shapes_tl_config.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
