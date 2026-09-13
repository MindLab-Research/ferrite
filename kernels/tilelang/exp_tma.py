#!/usr/bin/env python3
"""Pure-TMA microbenchmark: what does one CTA actually get from the fp4-unpack TMA?

The MoE prototype plateaus at ~20-25 GB/s per CTA regardless of tile shape or
stage count, and a TMA-only ablation (no UMMA, no SF) is the same speed -- so
the wall is the TMA path, not the tensor core. This probes whether that wall is
the *global access pattern* (128 discontiguous 64-byte segments per tile,
2560-byte row pitch) or the smem->TMEM issue chain, by varying the inner box.

  row layout:  A[R, K] float4_e2m1fn  (K contiguous, 0.5 B/elt)
  inner box :  BKI fp4 elts -> BKI/2 bytes of global traffic per row
  outer box :  BM rows, stride K/2 bytes
"""
import argparse
import time

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def tma_sweep(R, K, BM, BKI, iters, stages, threads=128):
    """One CTA streams [BM, BKI] fp4 tiles in a stages-deep TMA pipeline."""
    @T.prim_func
    def main(A: T.Tensor((R, K), T.float4_e2m1fn),
             Out: T.Tensor((R // BM,), "float32")):
        with T.Kernel(R // BM, threads=threads) as bx:
            Sh = T.alloc_shared((stages, BM, BKI), T.float4_e2m1_unpacked)
            loaded = T.alloc_barrier([32] * stages)
            consumed = T.alloc_barrier([1] * stages)
            tx = T.get_thread_binding()

            if tx < 32:
                for k in T.serial(iters):
                    st = k % stages
                    T.mbarrier_wait_parity(consumed[st], ((k // stages) & 1) ^ 1)
                    T.tma_copy(A[bx * BM:(bx + 1) * BM,
                                 (k * BKI) % K:((k * BKI) % K) + BKI],
                               Sh[st, :, :], barrier=loaded[st])
                    T.mbarrier_arrive(loaded[st])
            elif tx < 64:
                for k in T.serial(iters):
                    st = k % stages
                    T.mbarrier_wait_parity(loaded[st], (k // stages) & 1)
                    T.mbarrier_arrive(consumed[st])

            T.sync_threads()
            if tx == 0:
                Out[bx] = T.Cast("float32", Sh[0, 0, 0])

    return main


def bench(f, it=100, warm=20):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncta", type=int, default=148)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  tilelang {tilelang.__version__}  "
          f"{torch.cuda.get_device_name(0)}")
    print(f"ncta={args.ncta} iters={args.iters}  (unpack TMA: 0.5 B/elt global, 1 B/elt smem)")
    print(f"\n{'BKI':>5s} {'outer B':>8s} {'inner B':>8s} {'stg':>4s} {'us':>9s} "
          f"{'GB/s/tile':>10s} {'agg TB/s':>9s}")

    K = 5120
    for BKI in (128, 256, 512, 1024, 2560):
        for stages in (2, 4, 6):
            BM = 128
            R = args.ncta * BM
            try:
                ker = tma_sweep(R, K, BM, BKI, args.iters, stages)
                A = torch.zeros(R, K // 2, dtype=torch.uint8, device="cuda").view(
                    torch.float4_e2m1fn_x2)
                us = bench(lambda: ker(A))
                gb = args.ncta * args.iters * BM * BKI * 0.5 / 1e9
                print(f"{BKI:5d} {BM * K // 2:8d} {BKI // 2:8d} {stages:4d} {us:9.1f} "
                      f"{gb * 1e3 / (us * 1e-6) / 1e3 / args.ncta:10.2f} "
                      f"{gb / (us * 1e-6) / 1e3:9.2f}")
            except Exception as e:                      # noqa: BLE001
                print(f"{BKI:5d} {BM * K // 2:8d} {BKI // 2:8d} {stages:4d}   FAIL "
                      f"{type(e).__name__}: {str(e)[:40]}")


if __name__ == "__main__":
    main()
