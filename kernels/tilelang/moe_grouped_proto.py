#!/usr/bin/env python3
"""TileLang MoE grouped-GEMM prototype for ferrite verify MoE (36 assignments/layer).

Reproduces the numbers in docs/agent/tilelang-moe-grouped.md.

Run (remote B300, sm_103a, TileLang 0.1.14):
    scp kernels/tilelang/moe_grouped_proto.py ubuntu@43.202.208.136:~/tl_proj/
    ssh ubuntu@43.202.208.136 'cd ~/tl_proj && python3 moe_grouped_proto.py'

What it does
------------
1. host-side `moe_align`  : topk_ids [m=6, topk=6] -> expert-grouped segments
                            (same semantics as SGLang moe_align_block_size).
2. grouped GEMM prototype : one T.gemm per (expert segment, N-tile);
                            M = that expert's assignment count (1..6), padded to BM=16.
    arm A  bf16 weights     : weight streamed in the MMA's own dtype.
    arm B  fp4 e2m1+ue8m0   : fp4 read + in-kernel dequant to bf16 -> MMA.
3. SIMT 36-sweep proxy    : same shapes, no T.gemm (per-output serial K loop).
4. AOT export             : .cu device source + .so via Kernel.export_library().
"""
import os, re, struct, time
import torch
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------- shapes
E, DIM, INTER = 384, 5120, 320
TOPK, M_ROWS = 6, 6
NA = M_ROWS * TOPK          # 36 assignments/layer
N_UP = 2 * INTER            # w1 || w3  -> 640
BM = 16                     # MMA minimum M (per expert segment)
DEV = "cuda"
SIMT_REF_US = 250.0         # ferrite production SIMT baseline, per layer (40 layers @ ~10ms/step)

LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]     # e2m1 magnitudes


# ---------------------------------------------------------------- fp4 helpers
def bf16_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16


E2M1_BITS = [bf16_bits(v) for v in LUT] + [bf16_bits(-v) for v in LUT]   # nibble -> bf16 bits


def pair_lut():
    """256-entry uint32: one packed fp4 byte -> two bf16 (lo nibble in low half).

    This collapses the fp4->bf16 dequant to ONE shared-memory lookup per 2 weights.
    """
    return [(E2M1_BITS[b & 0xF] | (E2M1_BITS[(b >> 4) & 0xF] << 16)) & 0xFFFFFFFF
            for b in range(256)]


def pack_fp4(w):
    a = w.to(torch.float32)
    lut = torch.tensor(LUT, device=a.device)
    idx = (a.abs().unsqueeze(-1) - lut).abs().argmin(-1).to(torch.uint8)
    n = ((a < 0).to(torch.uint8) * 8 + idx).reshape(*a.shape[:-1], a.shape[-1] // 2, 2)
    return (n[..., 0] | (n[..., 1] << 4)).to(torch.uint8)          # low nibble = even k


def pack_ue8m0(s):
    return torch.clamp(torch.round(torch.log2(s.to(torch.float32))) + 127, 1, 254).to(torch.uint8)


def fp4_dequant_ref(Wq_e, Ws_e, dev):
    """fp4-roundtrip reference for one expert: [N, K] float32."""
    q = Wq_e.to(torch.int64)
    nb = torch.stack([q & 0xF, (q >> 4) & 0xF], dim=-1).reshape(Wq_e.shape[0], -1)
    s, e, m = (nb >> 3) & 1, (nb >> 1) & 3, nb & 1
    mag = torch.tensor(LUT, device=dev)[e * 2 + m]
    val = torch.where(s == 1, -mag, mag)
    return val * torch.pow(2.0, Ws_e.to(torch.float32) - 127).repeat_interleave(32, dim=-1)


# ---------------------------------------------------------------- host moe_align
def moe_align(topk_ids):
    """SGLang moe_align_block_size semantics, host side.

    Sort assignments by (expert asc, flat assignment index asc) so every expert owns a
    CONTIGUOUS run of rows -> one dense operand block per expert (expert-centric GEMM).
    Returns (order, seg_expert, seg_start, counts, nseg); no atomics, pure function of
    the routing table -> bit-identical across CUDA-graph replays.
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
    return order, uniq.to(torch.int32), (torch.cumsum(counts, 0) - counts), counts, uniq.numel()


# ---------------------------------------------------------------- kernels
@tilelang.jit(out_idx=[-1])
def k_bf16(NSEG, BM_, N, K, BN, BK, E_, threads=128, stages=3):
    @T.prim_func
    def main(A: T.Tensor((NSEG * BM_, K), "bfloat16"),
             W: T.Tensor((E_, N, K), "bfloat16"),
             Eid: T.Tensor((NSEG,), "int32"),
             C: T.Tensor((NSEG * BM_, N), "float32")):
        with T.Kernel(T.ceildiv(N, BN), NSEG, threads=threads) as (bx, by):
            A_sh = T.alloc_shared((BM_, BK), "bfloat16")
            B_sh = T.alloc_shared((BN, BK), "bfloat16")
            C_l = T.alloc_fragment((BM_, BN), "float32")
            e = Eid[by]
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[by * BM_, k * BK], A_sh)
                T.copy(W[e, bx * BN, k * BK], B_sh)
                T.gemm(A_sh, B_sh, C_l, transpose_B=True)
            T.copy(C_l, C[by * BM_, bx * BN])
    return main


@tilelang.jit(out_idx=[-1])
def k_fp4(NSEG, BM_, N, K, BN, BK, E_, threads=128, stages=3):
    K2, KS = K // 2, K // 32

    @T.prim_func
    def main(A: T.Tensor((NSEG * BM_, K), "bfloat16"),
             Wq: T.Tensor((E_, N, K2), "uint8"),
             Ws: T.Tensor((E_, N, KS), "uint8"),
             Lut: T.Tensor((256,), "uint32"),
             Eid: T.Tensor((NSEG,), "int32"),
             C: T.Tensor((NSEG * BM_, N), "float32")):
        with T.Kernel(T.ceildiv(N, BN), NSEG, threads=threads) as (bx, by):
            A_sh = T.alloc_shared((BM_, BK), "bfloat16")
            Bq = T.alloc_shared((BN, BK // 2), "uint8")
            Bs = T.alloc_shared((BN, BK // 32), "uint8")
            B_sh = T.alloc_shared((BN, BK), "bfloat16")
            C_l = T.alloc_fragment((BM_, BN), "float32")
            e = Eid[by]
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[by * BM_, k * BK], A_sh)
                T.copy(Wq[e, bx * BN, (k * BK) // 2], Bq)
                T.copy(Ws[e, bx * BN, (k * BK) // 32], Bs)
                for i, j in T.Parallel(BN, BK // 2):
                    p = Lut[Bq[i, j]]                     # one byte -> two bf16
                    scb = T.reinterpret("bfloat16",
                                        T.Cast("uint16", T.Cast("int32", Bs[i, j // 16]) << 7))
                    B_sh[i, 2 * j] = T.reinterpret("bfloat16", T.Cast("uint16", p & 0xFFFF)) * scb
                    B_sh[i, 2 * j + 1] = T.reinterpret("bfloat16", T.Cast("uint16", p >> 16)) * scb
                T.gemm(A_sh, B_sh, C_l, transpose_B=True)
            T.copy(C_l, C[by * BM_, bx * BN])
    return main


@tilelang.jit(out_idx=[-1])
def k_simt_fp4(NA_, N, K, BN, E_, threads=256):
    """Naive SIMT 36-sweep GEMV: one assignment per blockIdx.y, serial K loop, fp4 dequant."""
    K2, KS = K // 2, K // 32

    @T.prim_func
    def main(A: T.Tensor((NA_, K), "bfloat16"),
             Wq: T.Tensor((E_, N, K2), "uint8"),
             Ws: T.Tensor((E_, N, KS), "uint8"),
             Eid: T.Tensor((NA_,), "int32"),
             C: T.Tensor((NA_, N), "float32")):
        with T.Kernel(T.ceildiv(N, BN), NA_, threads=threads) as (bx, by):
            Wq_sh = T.alloc_shared((BN, K2), "uint8")
            Ws_sh = T.alloc_shared((BN, KS), "uint8")
            A_sh = T.alloc_shared((K,), "bfloat16")
            acc = T.alloc_fragment((BN,), "float32")
            e = Eid[by]
            T.copy(Wq[e, bx * BN, 0], Wq_sh)
            T.copy(Ws[e, bx * BN, 0], Ws_sh)
            T.copy(A[by, 0], A_sh)
            T.clear(acc)
            for j in T.Parallel(BN):
                for k in T.serial(K):
                    nb = T.Cast("int32", (Wq_sh[j, k // 2]
                                          >> T.Cast("uint8", (k % 2) * 4)) & T.Cast("uint8", 15))
                    s, ex, m = (nb >> 3) & 1, (nb >> 1) & 3, nb & 1
                    pos = T.if_then_else(ex == 0, m * 0x3F00, ((ex + 126) << 7) | (m << 6))
                    val = T.reinterpret("bfloat16", T.Cast("uint16", pos | (s << 15)))
                    scb = T.reinterpret("bfloat16", T.Cast("uint16", T.Cast("int32", Ws_sh[j, k // 32]) << 7))
                    acc[j] += T.Cast("float32", val * scb) * T.Cast("float32", A_sh[k])
            T.copy(acc, C[by, bx * BN])
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


def maxerr(C, Apad, Wdq, nseg, counts, seg_s, seg_e):
    mx = 0.0
    for s in range(nseg):
        c, st, e = int(counts[s]), int(seg_s[s]), int(seg_e[s])
        ref = Apad[s * BM:s * BM + c].float() @ Wdq[e].T
        mx = max(mx, (C.float()[s * BM:s * BM + c] - ref).abs().max().item())
    return mx


def main():
    torch.manual_seed(7)
    topk_ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M_ROWS)]).to(torch.int32).cuda()
    order, seg_e, seg_s, counts, nseg = moe_align(topk_ids)
    print(f"nseg={nseg} (distinct experts / {NA} assignments), counts {counts.min().item()}..{counts.max().item()}")

    def pad_rows(src, K):
        P = torch.zeros(nseg * BM, K, device=DEV, dtype=torch.bfloat16)
        for s in range(nseg):
            c, st = int(counts[s]), int(seg_s[s])
            P[s * BM:s * BM + c] = src[st:st + c]
        return P

    x = torch.randn(M_ROWS, DIM, device=DEV, dtype=torch.bfloat16) * 0.2
    A_up = pad_rows(x[order // TOPK], DIM)
    h = torch.randn(NA, INTER, device=DEV, dtype=torch.bfloat16) * 0.2
    A_dn = pad_rows(h, INTER)
    W13 = (torch.randn(E, N_UP, DIM, device=DEV) * 0.05).to(torch.bfloat16)
    W2 = (torch.randn(E, DIM, INTER, device=DEV) * 0.05).to(torch.bfloat16)
    W13q = pack_fp4(W13.cpu()).cuda()
    W13s = pack_ue8m0(torch.full((E, N_UP, DIM // 32), 2.0 ** -3, device=DEV).cpu()).cuda()
    W2q = pack_fp4(W2.cpu()).cuda()
    W2s = pack_ue8m0(torch.full((E, DIM, INTER // 32), 2.0 ** -3, device=DEV).cpu()).cuda()
    LutT = torch.tensor(pair_lut(), dtype=torch.int64).to(torch.uint32).cuda()
    eid = seg_e.cuda()
    used = sorted({int(v) for v in seg_e.cpu()})

    rows, ker = [], {}
    up_bf = k_bf16(nseg, BM, N_UP, DIM, 256, 64, E, threads=256, stages=3)
    C = up_bf(A_up, W13, eid)
    rows.append(("up", "bf16", "(BN=256,BK=64,th=256,stg=3)",
                 bench(lambda: up_bf(A_up, W13, eid)), maxerr(C, A_up, {e: W13[e].float() for e in used}, nseg, counts, seg_s, seg_e)))
    dn_bf = k_bf16(nseg, BM, DIM, INTER, 512, 64, E, threads=256, stages=2)
    C = dn_bf(A_dn, W2, eid)
    rows.append(("dn", "bf16", "(BN=512,BK=64,th=256,stg=2)",
                 bench(lambda: dn_bf(A_dn, W2, eid)), maxerr(C, A_dn, {e: W2[e].float() for e in used}, nseg, counts, seg_s, seg_e)))
    up_f4 = k_fp4(nseg, BM, N_UP, DIM, 256, 64, E, threads=256, stages=3)
    C = up_f4(A_up, W13q, W13s, LutT, eid)
    rows.append(("up", "fp4-LUT", "(BN=256,BK=64,th=256,stg=3)",
                 bench(lambda: up_f4(A_up, W13q, W13s, LutT, eid)),
                 maxerr(C, A_up, {e: fp4_dequant_ref(W13q[e], W13s[e], DEV) for e in used}, nseg, counts, seg_s, seg_e)))
    dn_f4 = k_fp4(nseg, BM, DIM, INTER, 256, 64, E, threads=256, stages=2)
    C = dn_f4(A_dn, W2q, W2s, LutT, eid)
    rows.append(("dn", "fp4-LUT", "(BN=256,BK=64,th=256,stg=2)",
                 bench(lambda: dn_f4(A_dn, W2q, W2s, LutT, eid)),
                 maxerr(C, A_dn, {e: fp4_dequant_ref(W2q[e], W2s[e], DEV) for e in used}, nseg, counts, seg_s, seg_e)))
    ker.update(up_bf16=up_bf, dn_bf16=dn_bf, up_fp4=up_f4, dn_fp4=dn_f4)

    print("\n=== per-layer (us) ===")
    for r in rows:
        print(f"  {r[0]:3s} {r[1]:8s} {r[2]:28s} {r[3]:8.1f}   max|err|={r[4]:.5f}")
    tu = [r[3] for r in rows if r[0] == "up" and r[1] == "bf16"][0]
    td = [r[3] for r in rows if r[0] == "dn" and r[1] == "bf16"][0]
    fu = [r[3] for r in rows if r[0] == "up" and r[1].startswith("fp4")][0]
    fd = [r[3] for r in rows if r[0] == "dn" and r[1].startswith("fp4")][0]
    print(f"  bf16 total {tu + td:7.1f} us = {100 * (tu + td) / SIMT_REF_US:.1f}% of SIMT {SIMT_REF_US:.0f}us")
    print(f"  fp4  total {fu + fd:7.1f} us = {100 * (fu + fd) / SIMT_REF_US:.1f}% of SIMT {SIMT_REF_US:.0f}us")

    ass_eid = seg_e.repeat_interleave(counts).cuda()
    simt = k_simt_fp4(NA, N_UP, DIM, 64, E, threads=256)
    print(f"  SIMT 36-sweep fp4 GEMV proxy (BN=64): {bench(lambda: simt(x[order // TOPK].contiguous(), W13q, W13s, ass_eid), it=30):.1f} us")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aot")
    os.makedirs(out, exist_ok=True)
    for name, k in ker.items():
        open(f"{out}/{name}.cu", "w").write(k.get_kernel_source())
        k.export_library(f"{out}/{name}.so")
    src = ker["up_bf16"].get_kernel_source()
    print(f"\nAOT -> {out}  [up_bf16.cu] lines={src.count(chr(10))} "
          f"mma_sync={len(re.findall(r'tl::mma_sync', src))} tma_load={len(re.findall(r'tl::tma_load', src))}")


if __name__ == "__main__":
    main()
