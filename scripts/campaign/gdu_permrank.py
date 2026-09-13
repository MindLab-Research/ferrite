#!/usr/bin/env python3
"""Recover the CHANNEL permutation of the block-scaled arm's gate|up output.

Finding to explain: `sorted(bs)` matches `sorted(old)` to a median of ~0.006 (typical magnitude
~5), i.e. the block-scaled arm computes the SAME multiset of values as the proven path but puts
them in DIFFERENT channels. This script recovers the mapping `bs channel n -> old channel m` by
rank matching and prints its structure, then tests the concrete layout hypotheses a TMEM/epilogue
mapping error produces (4/8/16/32-way interleave, 64-block swaps, bit-reversal).

Usage: gdu_permrank.py <bs.f32> <old.f32> [slot] [act_slot]
"""
import sys
import numpy as np

slot = int(sys.argv[3]) if len(sys.argv) > 3 else 1
act = int(sys.argv[4]) if len(sys.argv) > 4 else 640
bs = np.fromfile(sys.argv[1], dtype="<f4").reshape(-1, act)[slot]
old = np.fromfile(sys.argv[2], dtype="<f4").reshape(-1, act)[slot]

# rank-based permutation: perm[n] = the old channel whose value equals bs[n]'s
r_bs = np.argsort(np.argsort(bs, kind="stable"), kind="stable")
old_by_rank = np.argsort(old, kind="stable")
perm = old_by_rank[r_bs]
print(f"slot {slot}: |bs|={np.linalg.norm(bs):.5g} |old|={np.linalg.norm(old):.5g}")
print("perm[0:64] =", perm[:64].tolist())
print("perm[64:128] =", perm[64:128].tolist())
print("perm[128:192] =", perm[128:192].tolist())
print("identity fraction:", float((perm == np.arange(act)).mean()))

N = act
cands = {
    "identity": np.arange(N),
    "4-way interleave (n//4 + (n%4)*32)": (np.arange(N) // 4) + (np.arange(N) % 4) * 32,
    "4-way de-interleave ((n%32)*4 + n//32)": (np.arange(N) % 32) * 4 + (np.arange(N) // 32),
    "32-block reversal": (np.arange(N) // 32) * 32 + (31 - np.arange(N) % 32),
    "64-block swap within 128": (np.arange(N) % 128) // 64 * 64 + np.arange(N) % 64
                                + (np.arange(N) // 128) * 128
                                + (1 - 2 * ((np.arange(N) % 128) // 64)) * 0,
    "even-odd split": (np.arange(N) // 2) + (np.arange(N) % 2) * (N // 2),
    "8-way: (n//8)%8 blocks": ((np.arange(N) // 8) % 8) * (N // 8) + (np.arange(N) // 64) * 8
                              + np.arange(N) % 8,
}
for name, c in cands.items():
    if c.shape != (N,) or np.unique(c).size != N:
        print(f"  {name}: n/a"); continue
    # agreement with the recovered permutation (only where perm is trustworthy)
    agree = float((c == perm).mean())
    m = np.corrcoef(bs[c], old)[0, 1] if np.unique(c).size == N else np.nan
    print(f"  {name}: perm-agreement={agree:.3f}  corr(bs[perm-like], old)={m:.4f}")

# block-level structure: for each 32-wide bs block, which old block shares its multiset?
print("\nblock-level match (each bs 32-block -> old 32-block by sorted-value distance):")
nb = N // 32
for b in range(nb):
    seg = np.sort(bs[b * 32:(b + 1) * 32])
    best, bd = -1, 1e30
    for b2 in range(nb):
        d = np.abs(np.sort(old[b2 * 32:(b2 + 1) * 32]) - seg).max()
        if d < bd:
            bd, best = d, b2
    print(f"  bs block {b:>2} -> old block {best:>2} (maxdiff {bd:.4g})", end="\n" if b % 2 else "   ")
print()
