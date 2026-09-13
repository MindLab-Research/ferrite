#!/usr/bin/env python3
"""TileLang tcgen05 BLOCK-SCALED (mxfp4: e2m1 + ue8m0) MoE-up prototype for ferrite.

Follows docs/agent/tilelang-moe-grouped.md §8 "next step 1": the fp4 path was
dead (159.8us) because it read fp4 and *expanded* it to bf16 in-kernel. The only
way to cash in fp4's 4x bandwidth is B300's native block-scaled MMA, where the
e2m1 operands and their ue8m0 scales go into the tensor core as-is:

    T.tcgen05_gemm_blockscaled(A_sh, B_sh, C_tmem, SFA_tmem, SFB_tmem, ...)
      + T.tcgen05_cp_warpx4     (smem uint32 scale words -> TMEM)
      + T.tcgen05_sf_warp_transpose
      + T.alloc_tmem / T.alloc_barrier / T.mbarrier_wait_parity

Shape (same as the mma_sync prototype):
    MoE up = [M_e, 5120] x [5120, 640], 35 segments (counts 1..2), N=640, K=5120.

Why BM=128 (not the 16 of the mma_sync arm): block-scaled tcgen05 disables
warp-specialization (`disable_ws=True` in the lowerer), so GetTCGEN5MMAMeta
accepts only M % 64 == 0 / M % 128 == 0 -- there is no M=16/32 atom on this
path. The M padding is free: this kernel is weight-bandwidth bound.

Run (remote B300, sm_103a, TileLang 0.1.14):
    scp kernels/tilelang/moe_bs_proto.py ubuntu@43.202.208.136:~/tl_proj/
    ssh ubuntu@43.202.208.136 'cd ~/tl_proj && python3 moe_bs_proto.py'
"""
import argparse
import inspect
import textwrap
import time

import torch
import tilelang
import tilelang.language as T
import tilelang.language.gemm_op as _gemm_op

# ---------------------------------------------------------------------------
# TileLang 0.1.14 API defect: T.tcgen05_gemm_blockscaled() never sets the
# `is_tcgen05` annotation that T.tcgen05_gemm() does, so
# cuda::Gemm::SelectInst (src/cuda/op/gemm.cc:363) falls through to the
# "SFA/SFB regions are defined => this must be the SM120 NVF4 mma.sync path"
# branch and hard-fails on sm_103a:
#     InternalError: T.mma_gemm_blockscaled() requires an SM120 CUDA target
# The body is otherwise correct; re-exec it with the one missing annotation.
# (Upstream main has the identical omission -- verified against
#  raw.githubusercontent.com/tile-ai/tilelang/{v0.1.14,main}/tilelang/language/gemm_op.py.)
# ---------------------------------------------------------------------------
_SRC = textwrap.dedent(inspect.getsource(_gemm_op.tcgen05_gemm_blockscaled))
_NEEDLE = 'ann["sf_b_granularity_k"] = int(sf_b_granularity_k)'
assert _NEEDLE in _SRC, "gemm_op.py changed; re-derive the tcgen05 blockscaled shim"
_NS = dict(_gemm_op.__dict__)
exec(_SRC.replace(_NEEDLE, _NEEDLE + '\n    ann["is_tcgen05"] = 1'), _NS)
tcgen05_gemm_blockscaled = _NS["tcgen05_gemm_blockscaled"]
T.tcgen05_gemm_blockscaled = tcgen05_gemm_blockscaled

# ---------------------------------------------------------------- shapes
E, DIM, INTER = 384, 5120, 320
TOPK, M_ROWS = 6, 6
NA = M_ROWS * TOPK          # 36 assignments/layer
N_UP = 2 * INTER            # w1 || w3 -> 640
DEV = "cuda"
SIMT_REF_US = 250.0         # ferrite production SIMT baseline, per layer
EPOCH = -1                  # out_idx for the C tensor

LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]     # e2m1 magnitudes
GRAN = 32                   # e8m0 scale granularity along K (MXFP4)


# ---------------------------------------------------------------- fp4 helpers
def quantize_mxfp4(x: torch.Tensor, gran: int = GRAN):
    """[R, K] f32 -> (fp4 packed u8 [R, K/2] low nibble = even k, ue8m0 u8 [R, K/gran]).

    Per-32 e8m0 scales = ferrite's native routed-expert format
    (`quant.rs`: fp4 e2m1, I8-packed, ue8m0 per row x 32-col block).
    """
    r, k = x.shape
    assert k % gran == 0 and k % 2 == 0
    xv = x.float().reshape(r, k // gran, gran)
    amax = xv.abs().amax(dim=2).clamp_min(2.0 ** -126)
    exp = torch.ceil(torch.log2(amax / 6.0))
    sf = torch.pow(2.0, exp)
    sf_b = (exp + 127).clamp(1, 254).to(torch.uint8)
    y = (xv / sf.unsqueeze(2)).reshape(r, k)
    mags = torch.tensor(LUT, device=x.device)
    idx = (y.abs().unsqueeze(-1) - mags).abs().argmin(-1)
    nib = ((y < 0).to(torch.uint8) << 3) | idx.to(torch.uint8)
    nib = nib.reshape(r, k // 2, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).to(torch.uint8), sf_b


def as_fp4(u8: torch.Tensor) -> torch.Tensor:
    """[R, K/2] uint8 -> the torch storage dtype TileLang maps float4_e2m1fn onto.

    TileLang's runtime maps the 4-bit `float4_e2m1fn` param onto torch's packed
    `float4_e2m1fn_x2` (2 nibbles per byte), so the torch-side tensor keeps the
    K/2-packed shape while the kernel sees a logical K.
    """
    return u8.view(torch.float4_e2m1fn_x2)


def pack_sf_group_major(sf_b: torch.Tensor):
    """[R, KS] u8 -> flat uint32, group-major: [KS/4, R] flattened.

    One uint32 packs 4 consecutive e8m0 scales (4 * 32 = 128 K). This is the
    layout T.tcgen05_gemm_blockscaled expects (see TileLang's
    `examples/blockscaled_gemm_sm100`), NOT ferrite's row-major byte plane --
    a real integration needs either a load-time repack or one tiny kernel.
    """
    r, ks = sf_b.shape
    assert ks % 4 == 0
    w = sf_b.to(torch.int64)
    packed = (w[:, 0::4] | (w[:, 1::4] << 8) | (w[:, 2::4] << 16) | (w[:, 3::4] << 24)).to(torch.uint32)
    return packed.T.contiguous().reshape(-1)


def dequant_fp4(packed: torch.Tensor, sf_b: torch.Tensor, k: int, gran: int = GRAN):
    """fp4-roundtrip reference: packed u8 [R, K/2] + ue8m0 [R, K/gran] -> f32 [R, K]."""
    q = packed.to(torch.int64)
    nb = torch.stack([q & 0xF, (q >> 4) & 0xF], dim=-1).reshape(packed.shape[0], -1)[:, :k]
    s, e, m = (nb >> 3) & 1, (nb >> 1) & 3, nb & 1
    mag = torch.tensor(LUT, device=packed.device)[e * 2 + m]
    val = torch.where(s == 1, -mag, mag)
    return val * torch.pow(2.0, sf_b.to(torch.float32) - 127).repeat_interleave(gran, dim=-1)


# ---------------------------------------------------------------- host moe_align
def moe_align(topk_ids):
    """SGLang moe_align_block_size semantics, host side (see tilelang-moe-grouped.md §2.1)."""
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
    return order, uniq.to(torch.int32), (torch.cumsum(counts, 0) - counts), counts, uniq.numel()


# ---------------------------------------------------------------- kernel
@tilelang.jit(out_idx=[-1])
def k_up_bs(NSEG, N, K, BM=128, BN=128, BK=128, threads=128, stages=4, gran=GRAN,
            repeat=1, diag="none", store_bn=0, packed=False):
    """Grouped block-scaled fp4 (e2m1+ue8m0) up-GEMM, tcgen05 1-CTA, explicit async.

    A  : [NSEG*BM, K]  float4_e2m1fn   (segment rows gathered + BM-padded)
    W  : [NSEG, N, K]  float4_e2m1fn   (per-segment expert weight, de-duplicated)
    SFA: [K/128 * NSEG*BM] uint32      group-major packed e8m0 (A)
    SFW: [K/128 * NSEG*N ] uint32      group-major packed e8m0 (W)
    C  : [NSEG*BM, N]  float32

    `gran` = K elements covered by ONE ue8m0 scale byte (32 = MXFP4 / ferrite's
    native per-(row,32) expert format; 128 = a coarser "recent supported" form).
    """
    assert K % (gran * 4) == 0, "K must be a multiple of one packed SF word (4*gran)"
    assert K % (gran * 4 * 4) == 0 or BK % gran == 0
    sf_words = K // (gran * 4)          # uint32 words per row
    k_iters = K // BK
    sf_period = gran * 4 // BK          # k-iters covered by one uint32
    assert sf_period >= 1 and k_iters % sf_period == 0
    assert BK % gran == 0 and BK % 32 == 0
    assert N % 128 == 0 and BN % 128 == 0
    M = NSEG * BM

    @T.prim_func
    def main(A: T.Tensor((M, K), T.float4_e2m1fn),
             W: T.Tensor((NSEG, N, K), T.float4_e2m1fn),
             SFA: T.Tensor((sf_words * M,), T.uint32),
             SFW: T.Tensor((sf_words * NSEG * N,), T.uint32),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(N // BN, NSEG, threads=threads) as (bx, by):
            _sm = T.float4_e2m1fn if packed else T.float4_e2m1_unpacked
            A_sh = T.alloc_shared((stages, BM, BK), _sm)
            B_sh = T.alloc_shared((stages, BN, BK), _sm)
            SFA_sh = T.alloc_shared((stages, BM), "uint32")
            SFW_sh = T.alloc_shared((stages, BN), "uint32")

            C_tmem = T.alloc_tmem([BM, BN], "float32")
            SFA_tmem = T.alloc_tmem([BM, 4], "uint32")
            SFW_tmem = T.alloc_tmem([BM, BN // 128 * 4], "uint32")

            C_l = T.alloc_fragment((BM, BN), "float32")
            sbn = store_bn if store_bn else BN
            C_sh = T.alloc_shared((BM, sbn), "float32")

            loaded = T.alloc_barrier([32] * stages)      # TMA completion
            sf_full = T.alloc_barrier([32] * stages)     # SF transposed + fenced
            consumed = T.alloc_barrier([1] * stages)     # UMMA consumed the stage
            tmem_full = T.alloc_barrier([1])             # accumulator ready

            tx = T.get_thread_binding()

            if tx < 32:
                # warp 0: TMA producer (operands + packed scale words)
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(consumed[st], ph ^ 1)
                    T.tma_copy(A[by * BM:(by + 1) * BM, k * BK:(k + 1) * BK], A_sh[st, :, :],
                               barrier=loaded[st])
                    T.tma_copy(W[by, bx * BN:(bx + 1) * BN, k * BK:(k + 1) * BK], B_sh[st, :, :],
                               barrier=loaded[st])
                    if k % sf_period == 0:
                        g = k // sf_period
                        T.tma_copy(SFA[g * M + by * BM:g * M + (by + 1) * BM], SFA_sh[st, :],
                                   barrier=loaded[st])
                        T.tma_copy(SFW[g * NSEG * N + by * N + bx * BN:
                                       g * NSEG * N + by * N + (bx + 1) * BN], SFW_sh[st, :],
                                   barrier=loaded[st])
                    T.mbarrier_arrive(loaded[st])

            elif tx < 64:
                # warp 1: scale factors smem -> TMEM, then issue the block-scaled UMMA
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(loaded[st], ph)
                    T.mbarrier_wait_parity(sf_full[st], ph)
                    if k % sf_period == 0:
                        T.tcgen05_cp_warpx4(SFA_sh[st, :], SFA_tmem)
                        T.tcgen05_cp_warpx4(SFW_sh[st, :], SFW_tmem)
                    if diag == "tma":
                        # ablation: no UMMA at all -- pure TMA+SF throughput floor
                        if k == k_iters - 1:
                            T.tcgen05_gemm_blockscaled(
                                A_sh[st, :, :], B_sh[st, :, :], C_tmem, SFA_tmem, SFW_tmem,
                                transpose_B=True, mbar=consumed[st], clear_accum=True,
                                k_start=k * BK,
                                sf_a_granularity_k=gran, sf_b_granularity_k=gran,
                            )
                        else:
                            T.tcgen05_mma_arrive(consumed[st])
                    elif diag == "nosf":
                        # ablation: SF hoisted out of the loop (numerics INVALID)
                        for rep in T.serial(repeat):
                            T.tcgen05_gemm_blockscaled(
                                A_sh[st, :, :], B_sh[st, :, :], C_tmem, SFA_tmem, SFW_tmem,
                                transpose_B=True, mbar=consumed[st],
                                clear_accum=(k == 0) and (rep == 0),
                                k_start=k * BK,
                                sf_a_granularity_k=gran, sf_b_granularity_k=gran,
                            )
                    else:
                        for rep in T.serial(repeat):
                            T.tcgen05_gemm_blockscaled(
                                A_sh[st, :, :], B_sh[st, :, :], C_tmem, SFA_tmem, SFW_tmem,
                                transpose_B=True,
                                mbar=consumed[st],
                                clear_accum=(k == 0) and (rep == 0),
                                k_start=k * BK,
                                sf_a_granularity_k=gran,
                                sf_b_granularity_k=gran,
                            )
                T.tcgen05_mma_arrive(tmem_full)

            elif tx < 96:
                # warp 2: tcgen05.cp 32x128b.warpx4 needs the scale words transposed
                for k in T.serial(k_iters):
                    st = k % stages
                    ph = (k // stages) & 1
                    T.mbarrier_wait_parity(loaded[st], ph)
                    if k % sf_period == 0:
                        T.tcgen05_sf_warp_transpose(SFA_sh[st, :])
                        T.tcgen05_sf_warp_transpose(SFW_sh[st, :])
                        T.fence_proxy_async()
                    T.mbarrier_arrive(sf_full[st])

            # epilogue: every warp
            T.mbarrier_wait_parity(tmem_full, 0)
            T.sync_threads()
            T.copy(C_tmem, C_l)
            if sbn == BN:
                T.copy(C_l, C_sh)
                T.copy(C_sh, C[by * BM, bx * BN])
            else:
                for i in T.serial(BN // sbn):
                    T.copy(C_l[:, i * sbn:(i + 1) * sbn], C_sh)
                    T.copy(C_sh, C[by * BM, bx * BN + i * sbn])

    return main


# ---------------------------------------------------------------- driver
def bench(f, it=400):
    for _ in range(25):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e6


def pad_segments(src, order, counts, nseg, BM, K):
    P = torch.zeros(nseg * BM, K, device=src.device, dtype=src.dtype)
    st = 0
    for s in range(nseg):
        c = int(counts[s])
        P[s * BM:s * BM + c] = src[order[st:st + c]]
        st += c
    return P


def run_case(nseg, counts, BM, K, N, cfg, seed=7, ref_rows=None):
    torch.manual_seed(seed)
    gran = cfg.get("gran", GRAN)
    A_f = torch.randn(nseg * BM, K, device=DEV, dtype=torch.float32) * 0.2
    W_f = torch.randn(nseg, N, K, device=DEV, dtype=torch.float32) * 0.05
    if ref_rows is not None:                       # zero out the BM padding rows
        mask = torch.zeros(nseg * BM, device=DEV, dtype=torch.bool)
        for s in range(nseg):
            mask[s * BM:s * BM + int(counts[s])] = True
        A_f = A_f * mask[:, None]

    Aq, Asf = quantize_mxfp4(A_f, gran)
    Wq, Wsf = quantize_mxfp4(W_f.reshape(nseg * N, K), gran)
    sfa = pack_sf_group_major(Asf)
    sfw = pack_sf_group_major(Wsf)
    Wq = Wq.view(nseg, N, K // 2)

    ker = k_up_bs(nseg, N, K, BM=BM, BN=cfg["bn"], BK=cfg["bk"],
                  threads=cfg["threads"], stages=cfg["stages"], gran=gran,
                  repeat=cfg.get("repeat", 1), diag=cfg.get("diag", "none"),
                  store_bn=cfg.get("store_bn", 0), packed=cfg.get("packed", False))
    C = ker(as_fp4(Aq), as_fp4(Wq), sfa, sfw)

    Adq = dequant_fp4(Aq, Asf, K, gran)
    Wdq = dequant_fp4(Wq.reshape(nseg * N, K // 2), Wsf, K, gran)
    ref = torch.einsum("smk,snk->smn", Adq.view(nseg, BM, K), Wdq.view(nseg, N, K))
    got = C.view(nseg, BM, N).float()
    err = (got - ref).abs().max().item()

    us = bench(lambda: ker(as_fp4(Aq), as_fp4(Wq), sfa, sfw))
    return us, err, ker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--gran", type=int, default=GRAN)
    ap.add_argument("--dense-only", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  tilelang {tilelang.__version__}  "
          f"dev {torch.cuda.get_device_name(0)}")
    base = dict(bn=args.bn, bk=args.bk, threads=args.threads, stages=args.stages,
                gran=args.gran)

    torch.manual_seed(7)
    topk_ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M_ROWS)]).to(torch.int32).cuda()
    order, seg_e, seg_s, counts, nseg = moe_align(topk_ids)
    print(f"MoE: nseg={nseg}, counts {counts.min().item()}..{counts.max().item()}, "
          f"{NA} assignments")

    if args.sweep:
        print(f"\n{'cfg':38s} {'us':>8s} {'TB/s':>7s}  max|err|")
        wb = nseg * N_UP * DIM * 0.5 / 1e6
        for diag, repeat, stages in (("tma", 1, 4), ("tma", 1, 6),
                                     ("none", 1, 4), ("none", 2, 4), ("none", 4, 4)):
            cfg = dict(bn=128, bk=128, threads=128, stages=stages, gran=32,
                       repeat=repeat, diag=diag)
            try:
                us, err, _ = run_case(nseg, counts, 128, DIM, N_UP, cfg, ref_rows=True)
                print(f"{f'diag={diag} rep={repeat} stg={stages}':38s} "
                      f"{us:8.1f} {wb / (us * 1e-6) / 1e3:7.2f}  {err:.5f}")
            except Exception as e:                       # noqa: BLE001
                print(f"{f'diag={diag} rep={repeat} stg={stages}':38s} "
                      f"FAIL {type(e).__name__}: {str(e)[:60]}")
        return

    # --- arm 0: dense sanity (M=BM=128 all real rows, N=640, K=5120) ---------
    us, err, ker = run_case(1, torch.tensor([128]), 128, DIM, N_UP, base, ref_rows=None)
    print(f"\n[dense 1x128]  BN={base['bn']} BK={base['bk']} stg={base['stages']} "
          f"gran={base['gran']} -> {us:8.1f} us   max|err| = {err:.5f}")
    print("  grid=1 CTA (launch+single-SM bound, not a perf number)")

    if args.dense_only:
        return

    # --- arm 1: ferrite MoE-up shape (35 segments, counts 1..2) -------------
    us, err, ker = run_case(nseg, counts, 128, DIM, N_UP, base, ref_rows=True)
    wbytes = nseg * N_UP * DIM * 0.5 / 1e6
    print(f"\n[moe 35x128]  gran={base['gran']} -> {us:8.1f} us   max|err| = {err:.5f}")
    print(f"  W bytes = {wbytes:.1f} MB -> {wbytes / (us * 1e-6) / 1e3:.2f} TB/s ; "
          f"{100 * us / SIMT_REF_US:.1f}% of SIMT {SIMT_REF_US:.0f}us")

    src = ker.get_kernel_source()
    n_bs = len(__import__("re").findall(r"blockscaled|block_scale|mxf8f6f4", src))
    print(f"\nAOT: lines={src.count(chr(10))} blockscaled_refs={n_bs}")


if __name__ == "__main__":
    main()
