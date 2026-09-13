#!/usr/bin/env python3
"""Stage check for the routed MoE's swiglu boundary, against the official reference.

Consumes two dumps from the SAME first MoE call (the gate|up one and the new post-swiglu one) and
reproduces what `model.py:841-851` does between them:

    gate = w1(x).float(); up = w3(x).float()          # our dumps are the raw pair
    if swiglu_limit > 0: up = clamp(up, -L, L); gate = clamp(gate, max=L)
    x = F.silu(gate) * up

with the reference's dtype boundary: the gate|up GEMM output is bf16 (`fp4_gemm`'s out_dtype), which
ferrite applies through `DSV41_BF16_TRUNCATE` (bf16_snap on the flat gate|up block) BEFORE the
swiglu. So the expected value is one of two variants, and this script reports both:

    A) bf16(gate|up) -> clamp -> silu*up            (BF16_TRUNCATE=1, the arm's COMMON)
    B) gate|up (f32)  -> clamp -> silu*up           (BF16_TRUNCATE=0)

A mismatch against BOTH would mean our swiglu differs from the official in a way neither boundary
choice explains; agreement with A validates the whole boundary chain.

Usage: swiglu_check.py --gateup DIR/gateup.f32 --swiglu DIR/swiglu.f32 [--topk 6] [--act-slot 640]
                       [--inter 320] [--limit 10.0]
"""
import argparse
import os
import sys

import numpy as np


def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    return r


def silu(x):
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateup", required=True)
    ap.add_argument("--swiglu", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--act-slot", type=int, default=640)
    ap.add_argument("--inter", type=int, default=320)
    ap.add_argument("--limit", type=float, default=10.0)
    a = ap.parse_args()

    if not os.path.exists(a.gateup) or not os.path.exists(a.swiglu):
        print(f"missing dump: {a.gateup} or {a.swiglu}"); return 2
    gu = np.fromfile(a.gateup, dtype="<f4")
    sw = np.fromfile(a.swiglu, dtype="<f4")
    n = min(gu.size, sw.size)
    gu, sw = gu[:n], sw[:n]
    rows = n // (a.topk * a.act_slot)
    if rows == 0:
        print(f"dumps too small ({n}) for topk={a.topk} act_slot={a.act_slot}"); return 2
    gu = gu.reshape(rows, a.topk, a.act_slot)
    sw = sw.reshape(rows, a.topk, a.act_slot)
    gate = gu[:, :, :a.inter]
    up = gu[:, :, a.inter:2 * a.inter]
    got = sw[:, :, :a.inter]           # the swiglu writes in place over the gate half
    print(f"rows={rows} topk={a.topk} inter={a.inter} limit={a.limit} "
          f"live={got.size} values; tail of each slot holds the stale up half (never read)")

    for tag, base in (("A bf16 -> clamp -> silu*up", bf16(gu)),
                      ("B f32  -> clamp -> silu*up", gu)):
        g = base[:, :, :a.inter]
        u = base[:, :, a.inter:2 * a.inter]
        if a.limit > 0:
            g = np.minimum(g, a.limit)
            u = np.clip(u, -a.limit, a.limit)
        exp = (silu(g) * u).astype(np.float32)
        den = np.maximum(np.abs(exp), 1e-6)
        rel = np.abs(got - exp) / den
        ok = int((rel <= 5e-2).sum())
        print(f"  {tag}: max|d|={np.abs(got-exp).max():.6g} median rel={np.median(rel):.4g} "
              f"within 5% = {ok}/{got.size}  corr={np.corrcoef(got.ravel(), exp.ravel())[0,1]:+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
