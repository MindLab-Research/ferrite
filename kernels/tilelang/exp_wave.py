#!/usr/bin/env python3
"""Wave-size experiment: fit the whole MoE-up in ONE wave (<=148 CTAs) -> ~34us.

The nseg scan showed a hard ~32-36us plateau for nseg<=24 (<=120 CTAs) and a
2x step at 175 CTAs: the kernel is wave-quantised, not bandwidth-bound.
BN must be a multiple of 128 (tcgen05.cp 32x128b.warpx4 granularity), so a
one-wave grid needs N padded past 640.
"""
import torch

import moe_bs_proto as m

torch.manual_seed(7)
topk = torch.stack([torch.randperm(m.E)[:m.TOPK] for _ in range(m.M_ROWS)]).to(torch.int32).cuda()
order, seg_e, seg_s, counts, nseg = m.moe_align(topk)
print(f"nseg={nseg}")

CASES = [
    (640, 128, 6, 0),
    (640, 128, 6, 32),
    (768, 256, 4, 64),
    (768, 256, 3, 64),
    (768, 384, 3, 64),
    (768, 128, 6, 64),
    (1024, 256, 4, 64),
]
print(f"{'N':>5s} {'BN':>4s} {'stg':>4s} {'CTAs':>5s} {'us':>8s} {'GB/s':>8s} {'err':>10s}")
for N, BN, stg, sbn in CASES:
    cfg = dict(bn=BN, bk=128, threads=128, stages=stg, gran=32, store_bn=sbn)
    ctas = nseg * (N // BN)
    try:
        us, err, _ = m.run_case(nseg, counts, 128, m.DIM, N, cfg, ref_rows=True)
        eff = nseg * m.N_UP * m.DIM * 0.5 / 1e6        # useful bytes only
        print(f"{N:5d} {BN:4d} {stg:4d} {ctas:5d} {us:8.1f} {eff / (us * 1e-6) / 1e3:8.2f} "
              f"{err:10.2e}")
    except Exception as e:                              # noqa: BLE001
        print(f"{N:5d} {BN:4d} {stg:4d} {ctas:5d}   FAIL {type(e).__name__}: {str(e)[:50]}")
