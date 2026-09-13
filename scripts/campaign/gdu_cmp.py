#!/usr/bin/env python3
"""Offline comparator for `DSV41_GATEUP_DUMP` (routed gate|up, raw f32, [topk][act_slot]).

Ground truth = the proven per-slot path (`dsv41_experts_mxf4.cu`), whose end-to-end text has
been verified. Both dumps are taken at the FIRST `moe()` call of the same prompt, i.e. layer 0,
so the two arrays see byte-identical inputs and are directly comparable element by element.

Usage: gu_cmp.py <bs.f32> <old.f32> [topk] [act_slot]
"""
import sys
import numpy as np

bs_p, old_p = sys.argv[1], sys.argv[2]
topk = int(sys.argv[3]) if len(sys.argv) > 3 else 6
act_slot = int(sys.argv[4]) if len(sys.argv) > 4 else 640
inter = act_slot // 2

bs = np.fromfile(bs_p, dtype="<f4")
old = np.fromfile(old_p, dtype="<f4")
print(f"bs={bs_p} n={bs.size}  old={old_p} n={old.size}")
n = min(bs.size, old.size)
if n == 0:
    print("EMPTY dump — the arm never reached the gate/up dump site")
    sys.exit(0)
bs, old = bs[:n], old[:n]
den = np.maximum(np.abs(old), 1e-6)
rel = np.abs(bs - old) / den
print(f"max|bs|={np.abs(bs).max():.6g}  max|old|={np.abs(old).max():.6g}  "
      f"nan_bs={int(np.isnan(bs).sum())} nan_old={int(np.isnan(old).sum())}")
print(f"overall: max|d|={np.abs(bs-old).max():.6g}  median rel={np.median(rel):.3g}  "
      f"frac rel>0.05={float((rel > 0.05).mean()):.4f}  corr={np.corrcoef(bs, old)[0,1]:.6f}")

for s in range(min(topk, n // act_slot)):
    b = bs[s * act_slot:(s + 1) * act_slot]
    o = old[s * act_slot:(s + 1) * act_slot]
    d = np.abs(b - o)
    # a gate<->up swap shows up as a small cross-half difference
    dswap = min(np.abs(b - o[::-1]).max(), np.abs(np.concatenate([b[inter:], b[:inter]]) - o).max())
    print(f"slot {s}: max|d|={d.max():.6g} gate={d[:inter].max():.6g} up={d[inter:].max():.6g} "
          f"|cross-half-swap best max|d|={dswap:.6g} | bs[0:3]={np.round(b[:3],4)} old[0:3]={np.round(o[:3],4)}")

mag = np.abs(old) > 0.05 * np.abs(old).max()
if mag.sum() > 0:
    r = bs[mag] / np.where(old[mag] == 0, 1e-30, old[mag])
    print(f"ratio bs/old on the {int(mag.sum())} large-magnitude elements: "
          f"median={np.median(r):.4f} p10={np.percentile(r,10):.4f} p90={np.percentile(r,90):.4f} "
          f"-> uniform mis-scale if the spread is tight and != 1")
