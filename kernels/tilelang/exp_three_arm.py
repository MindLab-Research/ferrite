#!/usr/bin/env python3
"""Three-arm same-instant comparison: bf16 grouped MMA vs fp4 mma_sync vs fp4 blockscaled.

The remote B300 is a shared box (other tenants hold all 8 GPUs at ~100% at times),
so absolute micro-bench numbers drift by ~2.6x between sessions. Running the three
arms back-to-back in one process keeps the *ratios* meaningful even when the box is
contended; the doc reports both a quiet-session and a contended-session table.
"""
import time

import torch

import moe_bs_proto as bsm                      # tcgen05 block-scaled fp4
from moe_grouped_proto import (                 # bf16 + fp4 mma_sync arms
    BM,
    DEV,
    DIM,
    E,
    INTER,
    M_ROWS,
    N_UP,
    TOPK,
    fp4_dequant_ref,
    k_bf16,
    k_fp4,
    moe_align,
    pack_fp4,
    pack_ue8m0,
    pair_lut,
)


def bench(f, it=400):
    for _ in range(25):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e6


def main():
    torch.manual_seed(7)
    topk_ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M_ROWS)]).to(torch.int32).cuda()
    order, seg_e, seg_s, counts, nseg = moe_align(topk_ids)
    print(f"nseg={nseg}, counts {counts.min().item()}..{counts.max().item()}, "
          f"{M_ROWS * TOPK} assignments; {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"device SMs={props.multi_processor_count}")

    x = torch.randn(M_ROWS, DIM, device=DEV, dtype=torch.bfloat16) * 0.2
    Apad = torch.zeros(nseg * BM, DIM, device=DEV, dtype=torch.bfloat16)
    st = 0
    for s in range(nseg):
        c = int(counts[s])
        Apad[s * BM:s * BM + c] = x[order[st:st + c] // TOPK]
        st += c
    W13 = (torch.randn(E, N_UP, DIM, device=DEV) * 0.05).to(torch.bfloat16)
    W13q = pack_fp4(W13.cpu()).cuda()
    W13s = pack_ue8m0(torch.full((E, N_UP, DIM // 32), 2.0 ** -3, device=DEV).cpu()).cuda()
    LutT = torch.tensor(pair_lut(), dtype=torch.int64).to(torch.uint32).cuda()
    eid = seg_e.cuda()

    rows = []
    k1 = k_bf16(nseg, BM, N_UP, DIM, 256, 64, E, threads=256, stages=3)
    k1(Apad, W13, eid)
    rows.append(("bf16 grouped MMA (T.gemm, mma.sync)",
                 bench(lambda: k1(Apad, W13, eid))))
    k2 = k_fp4(nseg, BM, N_UP, DIM, 256, 64, E, threads=256, stages=3)
    k2(Apad, W13q, W13s, LutT, eid)
    rows.append(("fp4 in-kernel dequant + mma.sync",
                 bench(lambda: k2(Apad, W13q, W13s, LutT, eid))))

    for gran, stg in ((32, 6), (32, 4), (128, 6)):
        cfg = dict(bn=128, bk=128, threads=128, stages=stg, gran=gran)
        us, err, _ = bsm.run_case(nseg, counts, 128, DIM, N_UP, cfg, ref_rows=True)
        rows.append((f"fp4 tcgen05 blockscaled (BN=128 stg={stg} gran={gran})",
                     us))

    for name, us in rows:
        print(f"  {name:46s} {us:8.1f} us")

    # ratio view (same instant, same contention)
    d = dict(rows)
    bf = [v for k, v in d.items() if k.startswith("bf16")][0]
    print("\n  ratios vs bf16 grouped MMA:")
    for name, us in rows:
        print(f"    {name:46s} {us / bf:6.2f}x")


if __name__ == "__main__":
    main()
