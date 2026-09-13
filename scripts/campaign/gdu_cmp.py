#!/usr/bin/env python3
"""Offline comparator for `DSV41_GATEUP_DUMP` (routed gate|up, raw f32, [topk][act_slot]).

Ground truth = the proven per-slot path (`dsv41_experts_mxf4.cu`), whose end-to-end text has
been verified. Both dumps are taken at the FIRST `moe()` call of the same prompt, i.e. layer 0,
so the two arrays see byte-identical inputs and are directly comparable element by element.

⚠️ CLAMP AWARENESS (from the output-layout audit): the proven path clamps INSIDE its kernel
(`g = min(g, limit)`, `u = clamp(u, -limit, limit)`, `dsv41_experts_mxf4.cu:1980-1983`) while the
block-scaled arm writes the raw pair. With `swiglu_limit = 10.0` the raw bytes therefore differ
on `{g > 10} ∪ {|u| > 10}` BY CONSTRUCTION, and the downstream swiglu re-applies the same clamp,
so the two are only REQUIRED to agree after clamping. This script reports both views, so a clamp
difference is never mistaken for the defect.

Usage: gu_cmp.py <bs.f32> <old.f32> [topk] [act_slot] [swiglu_limit]
"""
import sys
import numpy as np

bs_p, old_p = sys.argv[1], sys.argv[2]
topk = int(sys.argv[3]) if len(sys.argv) > 3 else 6
act_slot = int(sys.argv[4]) if len(sys.argv) > 4 else 640
limit = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
inter = act_slot // 2

bs = np.fromfile(bs_p, dtype="<f4")
old = np.fromfile(old_p, dtype="<f4")
print(f"bs={bs_p} n={bs.size}  old={old_p} n={old.size}  (swiglu_limit={limit})")
n = min(bs.size, old.size)
if n == 0:
    print("EMPTY dump — the arm never reached the gate/up dump site")
    sys.exit(0)
bs, old = bs[:n], old[:n]


def clamp_pair(a):
    """The reference's clamp on a raw [topk][gate|up] block: gate from above only, up both sides."""
    a = a.reshape(-1, 2, inter).copy()
    a[:, 0, :] = np.minimum(a[:, 0, :], limit)          # gate: max only
    a[:, 1, :] = np.clip(a[:, 1, :], -limit, limit)     # up: both sides
    return a.reshape(-1)


bs_c, old_c = clamp_pair(bs), clamp_pair(old)


def stats(tag, x, y):
    den = np.maximum(np.abs(y), 1e-6)
    rel = np.abs(x - y) / den
    print(f"{tag}: max|d|={np.abs(x-y).max():.6g}  median rel={np.median(rel):.3g}  "
          f"frac rel>0.05={float((rel > 0.05).mean()):.4f}  corr={np.corrcoef(x, y)[0,1]:.6f}")


print(f"max|bs|={np.abs(bs).max():.6g}  max|old|={np.abs(old).max():.6g}  "
      f"nan_bs={int(np.isnan(bs).sum())} nan_old={int(np.isnan(old).sum())}")
stats("RAW          ", bs, old)
stats("AFTER CLAMP  ", bs_c, old_c)
raw_bad = (np.abs(bs - old) / np.maximum(np.abs(old), 1e-6)) > 0.05
cl_bad = (np.abs(bs_c - old_c) / np.maximum(np.abs(old_c), 1e-6)) > 0.05
print(f"elements failing the 5% test: raw={int(raw_bad.sum())} clamped={int(cl_bad.sum())} "
      f"(explained by the clamp: {int(raw_bad.sum()) - int(cl_bad.sum())})")

for s in range(min(topk, n // act_slot)):
    b = bs[s*act_slot:(s+1)*act_slot]; o = old[s*act_slot:(s+1)*act_slot]
    bc = bs_c[s*act_slot:(s+1)*act_slot]; oc = old_c[s*act_slot:(s+1)*act_slot]
    d, dc = np.abs(b - o), np.abs(bc - oc)
    dswap = min(np.abs(b - o[::-1]).max(),
                np.abs(np.concatenate([b[inter:], b[:inter]]) - o).max())
    print(f"slot {s}: raw max|d|={d.max():.6g} (gate {d[:inter].max():.6g} / up {d[inter:].max():.6g})"
          f"  clamped max|d|={dc.max():.6g}  cross-half-swap best={dswap:.6g}"
          f"  bs[0:3]={np.round(b[:3],4)} old[0:3]={np.round(o[:3],4)}")

mag = np.abs(old_c) > 0.05 * max(np.abs(old_c).max(), 1e-9)
if mag.sum() > 0:
    r = bs_c[mag] / np.where(old_c[mag] == 0, 1e-30, old_c[mag])
    print(f"ratio bs/old on the {int(mag.sum())} large-magnitude (clamped) elements: "
          f"median={np.median(r):.4f} p10={np.percentile(r,10):.4f} p90={np.percentile(r,90):.4f}")
