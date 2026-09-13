#!/usr/bin/env python3
# gen_head_aot.py — 从 TileLang head bf16 原型（head_bf16_tilelang.py）冻结出生产切片
# 形状的 AOT CUDA 源码，供 `kernels/cuda/tilelang_gen/head_bf16_shim.cu` 合入 ferrite 的
# nvcc 构建。
#
# =============================================================================
# ⚠️ 本脚本由**主 agent** 在远端执行（不是接线 agent）。
# =============================================================================
# 它是 AOT 生成（跑 TileLang → 产 CUDA 源码），属于 GPU/远端 python 运行；接线 agent
# 的纪律禁止其执行任何远端 GPU 操作（AGENTS.md 的铁律）。**接线只把命令写进清单。**
#
# 用法（远端 B300，tilelang 0.1.14）：
#   /opt/dlami/nvme/dsv41_venv/bin/python gen_head_aot.py <outdir>
# 产出（<outdir>，通常就是 kernels/cuda/tilelang_gen/）：
#   head_partial_tl.cu   # K-split 分片内核（裸指针 ABI）
#   head_reduce_tl.cu    # 确定性归约内核（裸指针 ABI，m + n 谓词）
#   head_tl_config.txt   # 编译期常量（grid/block/smem/ks/bN/threads）+ 签名（shim 的出处）
#
# ⚠️ GENERATED — do not edit. 重生成见 kernels/cuda/tilelang_gen/PROVENANCE.md。
#
# =============================================================================
# 形状（**以代码事实为准，非任务书**）
# =============================================================================
# ferrite 的 head 是 `head.weight` = [vocab=129280, dim=5120] bf16（weights.rs:90，
# Shard::Replicated）。**生产 draft 切片**是 vocab/world = 16160 行（world=8）——
# 本文件冻结的就是它（docs/agent/tilelang-attn-head.md §0 已判：任务书的 "16160×256"
# 里 16160 对，256 **不是** head 的 K；K = dim = 5120）。
#   N = 16160, K = 5120（=dim），MPAD = 16（mma m16n8k32 的 M 约束）
#
# =============================================================================
# 与原型（head_bf16_tilelang.py）的三处生成差异
# =============================================================================
# 1. **裸指针 ABI**：`pass_configs={TL_DISABLE_TMA_LOWER:1,
#    TL_DISABLE_WARP_SPECIALIZED:1}`（与 gen_wkv_aot.py / gen_moe_aot.py 同一路线，
#    见 docs/agent/tilelang-integration-design.md §1.3 实验 B）。原型的默认 lowering
#    把 A/W 变成 CUtensorMap + TMA + warp specialization，要求 host 侧用 driver API
#    `cuTensorMapEncodeTiled` 造描述符（build.sh 不链 -lcuda）。裸指针 ABI 让 shim 直接
#    把 ferrite 的 device 指针传进来。
# 2. **P 的 N 向容量补到 NPAD = ceildiv(N, BN)*BN = 16256**：N=16160 不整除 BN=128，
#    末块（bx=126）的 store 会写列 16128..16255 ⇒ 若不补就依赖 TileLang 的 store 谓词，
#    补上则 partial 的 store 永远在界内、reduce 读 P 也永远在界内（见下 (3) 的 n 谓词）。
#    多出的 96 列 × 16 行 × 8 片 × 4 B = 48 KiB，可忽略。
# 3. **reduce 带 `m` 运行期标量 + `col < N` 谓词**：调用方（shim）的 `out` 是 ferrite 的
#    logits_r，只有 **m 行**且行 stride == N ⇒ reduce 只写前 m 行（`if i < m`）、
#    不写超出 N 的补列（`if bx*BN+j < N`），于是**不需要 [16, N] 输出 staging、也不需要
#    回拷**。与 gen_wkv_aot.py 的 reduce 谓词同源。
#
# 数值形态（见原型 §4）：tensor-core mma 没有 bf16×f32 形态 ⇒ X 必须降到 bf16。**cast
# 在 shim 的 host 侧完成**（`tl_head_cast_kernel`），本生成物接收的是**已 cast 的 bf16
# X** —— 这正是原型实测的形态（原型也在 host 侧 cast），所以生成物与实测 M6/M1=1.00 同构。

import hashlib
import sys

import tilelang
import tilelang.language as T

# 裸指针 ABI 的两个 pass_configs（设计文档 §1.3 实验 B，与 gen_wkv_aot.py / gen_moe_aot.py 一致）。
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

MPAD = 16  # mma m16n8k32 的 M 约束

# ---- 冻结几何（照 docs/agent/tilelang-attn-head.md §2/§3 的实测最优点）--------
N, K = 16160, 5120     # 生产 draft 切片（vocab/world, dim）
BN = 128               # partial 的 N 向 tile
KS = 8                 # K-split 分片数
BK = 64                # partial 的 K 向 tile（必须整除 Kc = K // KS）
NS = 3                 # pipelined stages
THREADS = 128          # partial block
RED_BN = 256           # reduce 的 N 向 tile
RED_THREADS = 256      # reduce block

NPAD = ((N + BN - 1) // BN) * BN  # 16256 = P 的 N 向容量（见文件头差异 2）


@tilelang.jit(pass_configs=PASS_CONFIGS)
def head_bf16_partial(N_, K_, NPAD_, bN, ks, bK=64, threads=128, ns=3):
    """K-split 分片：grid (ceildiv(N, bN), ks)，每块算一段 K 的 partial，写入 P[kp]。

    head 是权重流（N×K×2 字节）主导的 GEMM：M 只有 1..6，激活 (m×K) 几乎全在 L2。
    ks 规则同投影族：把块数从 N/bN 抬到 (N/bN)*ks 以填满 148 SM。
    """
    Kc = K_ // ks
    # bK 必须整除 Kc；小 K 要自动收窄，否则 `Kc // bK == 0` ⇒ pipelined loop 空转 ⇒
    # 输出恒 0（静默错值，原型 §2 的 bK bug）。本形状 Kc=640、bK=64 ⇒ 10 步，正常。
    bK = min(bK, Kc)
    assert Kc % bK == 0, f"bK={bK} must divide Kc={Kc}"

    @T.prim_func
    def main(X: T.Tensor((MPAD, K_), "bfloat16"),
             W: T.Tensor((N_, K_), "bfloat16"),
             P: T.Tensor((ks, MPAD, NPAD_), "float32")):
        with T.Kernel(T.ceildiv(N_, bN), ks, threads=threads) as (bx, kp):
            X_sh = T.alloc_shared((MPAD, bK), "bfloat16")
            W_sh = T.alloc_shared((bN, bK), "bfloat16")
            C_l = T.alloc_fragment((MPAD, bN), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(Kc // bK, num_stages=ns):
                T.copy(X[0, kp * Kc + ko * bK], X_sh)
                T.copy(W[bx * bN, kp * Kc + ko * bK], W_sh)
                # 纯 bf16 mma，无 scale epilogue（head 是稠密 bf16 GEMM —— 这是它比投影族
                # 更简单的地方，所以 ABI 里没有 a_scale / w_scale）。
                T.gemm(X_sh, W_sh, C_l, transpose_B=True)
            T.copy(C_l, P[kp, 0, bx * bN])

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def head_bf16_reduce(ks, N_, NPAD_, bN=RED_BN, threads=RED_THREADS):
    """确定性归约：按 kp 升序求和 ks 个 partial，只写前 m 行、前 N 列（行 stride = N_）。"""
    IDX = tuple(range(ks))

    @T.prim_func
    def main(P: T.Tensor((ks, MPAD, NPAD_), "float32"),
             C: T.Tensor((MPAD, N_), "float32"),
             m: T.int32):
        with T.Kernel(T.ceildiv(N_, bN), threads=threads) as bx:
            for i, j in T.Parallel(MPAD, bN):
                if i < m:
                    if bx * bN + j < N_:
                        C[i, bx * bN + j] = sum([P[kq, i, bx * bN + j] for kq in IDX])

    return main


def _dump(kern, path):
    src = kern.get_kernel_source()
    with open(path, "w") as f:
        f.write(src)
    return src


def _sha(src):
    return hashlib.sha256(src.encode()).hexdigest()


def _banner(name, raw_sha, raw_bytes, desc):
    return (
        f"// {name} — GENERATED by kernels/tilelang/gen_head_aot.py; DO NOT EDIT.\n"
        f"// raw sha256 (this dump, before this banner) = {raw_sha}\n"
        f"// raw bytes = {raw_bytes}\n"
        f"// {desc}\n"
        f"// Regenerate: see kernels/cuda/tilelang_gen/PROVENANCE.md §5 (head).\n"
    )


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    kp = head_bf16_partial(N, K, NPAD, BN, KS, BK, THREADS, NS)
    kr = head_bf16_reduce(KS, N, NPAD, RED_BN, RED_THREADS)
    sp = _dump(kp, f"{outdir}/head_partial_tl.cu")
    sr = _dump(kr, f"{outdir}/head_reduce_tl.cu")
    sp_sha, sr_sha = _sha(sp), _sha(sr)
    # banner 写回（sha 记的是 banner 之前的原文 —— 与 gen_moe_aot.py 同规）。
    with open(f"{outdir}/head_partial_tl.cu", "w") as f:
        f.write(
            _banner(
                "head_partial_tl.cu",
                sp_sha,
                len(sp),
                f"head bf16 K-split partial: N={N} K={K} NPAD={NPAD} BN={BN} KS={KS} "
                f"BK={BK} th={THREADS} stg={NS}. X [{MPAD},{K}] bf16 / W [{N},{K}] bf16 / "
                f"P [{KS},{MPAD},{NPAD}] f32.",
            )
            + sp
        )
    with open(f"{outdir}/head_reduce_tl.cu", "w") as f:
        f.write(
            _banner(
                "head_reduce_tl.cu",
                sr_sha,
                len(sr),
                f"head bf16 deterministic reduce: KS={KS} N={N} NPAD={NPAD} BN={RED_BN} "
                f"th={RED_THREADS}. P [{KS},{MPAD},{NPAD}] f32 / C [{MPAD},{N}] f32 / m i32.",
            )
            + sr
        )

    smem = NS * (MPAD + BN) * BK * 2  # bf16 = 2 B
    lines = [
        "# head TileLang AOT config (GENERATED by gen_head_aot.py)",
        f"N={N} K={K} NPAD={NPAD} BN={BN} KS={KS} BK={BK} NS={NS} THREADS={THREADS} "
        f"MPAD={MPAD} RED_BN={RED_BN} RED_THREADS={RED_THREADS}",
        f"partial_grid=({(N + BN - 1)//BN}, {KS})",
        f"partial_block={THREADS}",
        f"partial_smem_bytes={smem}",
        f"reduce_grid=({(N + RED_BN - 1)//RED_BN},)",
        f"reduce_block={RED_THREADS}",
        f"raw_sha256(partial)={sp_sha}",
        f"raw_sha256(reduce)={sr_sha}",
        "",
        "--- head_partial_tl.cu signature ---",
        *[l for l in sp.splitlines() if "main_kernel" in l and "__global__" in l and "launch_bounds" in l],
        "--- head_reduce_tl.cu signature ---",
        *[l for l in sr.splitlines() if "main_kernel" in l and "__global__" in l and "launch_bounds" in l],
        "",
        f"partial_cu_bytes={len(sp)}",
        f"reduce_cu_bytes={len(sr)}",
    ]
    with open(f"{outdir}/head_tl_config.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
