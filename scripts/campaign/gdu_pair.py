#!/usr/bin/env python3
"""Structural discriminators on the gate|up A/B dumps.

Q1  Is the block-scaled arm's slot i actually SOME OTHER expert's output (a segment/slot
    permutation)?  Test every (bs slot, old slot) pairing by correlation.
Q2  Is the content right but the per-32-K-block SCALE wrong (the SF hypothesis)?  Look at the
    per-32-block norm ratio bs/old: a wrong block scale shows up as a *tight* ratio per block
    that varies block to block, while a wrong weight/activation shows up as an arbitrary ratio.

Usage: gdu_pair.py <bs.f32> <old.f32> [topk] [act_slot]
"""
import sys
import numpy as np

topk = int(sys.argv[3]) if len(sys.argv) > 3 else 6
act = int(sys.argv[4]) if len(sys.argv) > 4 else 640
bs = np.fromfile(sys.argv[1], dtype="<f4")
old = np.fromfile(sys.argv[2], dtype="<f4")
n = min(bs.size, old.size)
bs = bs[:n].reshape(-1, act)
old = old[:n].reshape(-1, act)
print(f"slots={bs.shape[0]} act_slot={act}")

print("\n== Q1: pairwise |corr| matrix (rows = bs slot, cols = old slot) ==")
print("        " + "".join(f"{j:>10d}" for j in range(min(topk, old.shape[0]))))
best = []
for i in range(min(topk, bs.shape[0])):
    row = [float(np.corrcoef(bs[i], old[j])[0, 1]) for j in range(min(topk, old.shape[0]))]
    print(f"bs[{i}] " + "".join(f"{c:>10.4f}" for c in row))
    j = int(np.nanargmax(np.abs(row)))
    best.append((i, j, row[j]))
print("best |corr| per bs slot (bs slot -> old slot, corr):", [(i, j, round(c, 4)) for i, j, c in best])
print("diagonal (same slot) corr:", [round(float(np.corrcoef(bs[i], old[i])[0, 1]), 4) for i in range(min(topk, bs.shape[0]))])

print("\n== Q2: per-32-K-block norm ratio bs/old (per slot) ==")
for i in range(min(topk, bs.shape[0])):
    r = bs[i].reshape(-1, 32)
    o = old[i].reshape(-1, 32)
    rat = np.linalg.norm(r, axis=1) / np.maximum(np.linalg.norm(o, axis=1), 1e-12)
    print(f"slot {i}: ratio min={rat.min():.4g} p25={np.percentile(rat,25):.4g} "
          f"median={np.median(rat):.4g} p75={np.percentile(rat,75):.4g} max={rat.max():.4g}")

print("\n== Q2b: same, but comparing bs against old REVERSED inside each block (byte-order) ==")
for i in range(min(2, bs.shape[0])):
    r = bs[i].reshape(-1, 32)
    o = old[i].reshape(-1, 32)
    for tag, oo in (("as-is", o), ("block-reversed", o[:, ::-1]), ("blocks-reversed", o[::-1, :])):
        c = float(np.corrcoef(r.ravel(), oo.ravel())[0, 1])
        print(f"slot {i} [{tag}]: corr={c:.4f}")
