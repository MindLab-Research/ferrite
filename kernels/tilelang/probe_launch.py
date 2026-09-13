#!/usr/bin/env python3
"""Split 'fixed per-launch cost' from 'real GPU time' for the tcgen05 blockscaled prototype."""
import time

import torch
import tilelang
import tilelang.language as T

import moe_bs_proto as m


@tilelang.jit(out_idx=[-1])
def tiny(N):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=32) as bx:
            for i in T.Parallel(N):
                Y[i] = X[i] + 1.0

    return main


def graph_bench(f, it=200, warm=10, reps=4):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it):
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / it * 1e6


def host_bench(f, it=400, warm=25):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e6


def main():
    tk = tiny(64)
    x = torch.zeros(64, device="cuda")
    print(f"tiny kernel (1 CTA, 64 elems): host-bench {host_bench(lambda: tk(x)):7.1f} us")

    torch.manual_seed(7)
    topk = torch.stack([torch.randperm(m.E)[:m.TOPK] for _ in range(m.M_ROWS)]).to(torch.int32).cuda()
    order, seg_e, seg_s, counts, nseg = m.moe_align(topk)

    print(f"\n{'nseg':>5s} {'stg':>4s} {'host us':>9s} {'graph us':>9s} {'kern us':>9s} "
          f"{'TB/s(graph)':>11s} {'err':>9s}")
    for ns in (8, 24, 35):
        cnt = torch.full((ns,), 2, device="cuda") if ns > 1 else torch.tensor([128])
        for stg in (4, 6):
            cfg = dict(bn=128, bk=128, threads=128, stages=stg, gran=32)
            us, err, ker = m.run_case(ns, cnt, 128, m.DIM, m.N_UP, cfg, ref_rows=(ns > 1))
            # rebuild the exact call args for graph timing
            A_f = torch.randn(ns * 128, m.DIM, device="cuda") * 0.2
            W_f = torch.randn(ns, m.N_UP, m.DIM, device="cuda") * 0.05
            Aq, Asf = m.quantize_mxfp4(A_f)
            Wq, Wsf = m.quantize_mxfp4(W_f.reshape(ns * m.N_UP, m.DIM))
            sfa = m.pack_sf_group_major(Asf)
            sfw = m.pack_sf_group_major(Wsf)
            Wq = Wq.view(ns, m.N_UP, m.DIM // 2)
            args = (m.as_fp4(Aq), m.as_fp4(Wq), sfa, sfw)
            gus = graph_bench(lambda: ker(*args))
            wb = ns * m.N_UP * m.DIM * 0.5 / 1e6
            print(f"{ns:5d} {stg:4d} {us:9.1f} {gus:9.1f} {'' :>9s} "
                  f"{wb / (gus * 1e-6) / 1e3:11.2f} {err:9.5f}")


if __name__ == "__main__":
    main()
