#!/usr/bin/env python3
"""Permutation test on the gate|up A/B dumps.

Signature to explain: the per-32-K-block energy of the block-scaled arm matches the proven
path's (median ratio ~0.95-1.00) while the element-wise correlation is ~0. That is what a
CHANNEL (N) permutation looks like: the same multiset of values, in a different order.

Tests, per slot:
  1. sorted(bs) vs sorted(old)  -> if they agree, bs is a permutation of old.
  2. recover the mapping n -> m (nearest value) and print its structure: is it a bijection, and
     does it look like a stride / bit-reversal / block swap (the shapes a smem-layout error
     makes) or random?
  3. the same on the gate half and the up half separately (a B-tile row-order error permutes the
     two 64-channel groups independently).

Usage: gdu_perm.py <bs.f32> <old.f32> [topk] [act_slot]
"""
import sys
import numpy as np

topk = int(sys.argv[3]) if len(sys.argv) > 3 else 6
act = int(sys.argv[4]) if len(sys.argv) > 4 else 640
inter = act // 2
bs = np.fromfile(sys.argv[1], dtype="<f4")
old = np.fromfile(sys.argv[2], dtype="<f4")
n = min(bs.size, old.size)
bs = bs[:n].reshape(-1, act)
old = old[:n].reshape(-1, act)


def describe(perm, tag):
    uniq = len(set(perm.tolist()))
    print(f"  [{tag}] bijection={uniq == perm.size} (unique {uniq}/{perm.size})")
    print(f"  [{tag}] first 32: {perm[:32].tolist()}")
    d = np.diff(perm.astype(np.int64))
    print(f"  [{tag}] diff(first 32): {d[:31].tolist()}")
    print(f"  [{tag}] is identity on first 32: {bool((perm[:32] == np.arange(32)).all())}")


for s in range(bs.shape[0]):
    b, o = bs[s], old[s]
    ds = np.abs(np.sort(b) - np.sort(o))
    print(f"slot {s}: sorted-diff median={np.median(ds):.4g} p90={np.percentile(ds,90):.4g} "
          f"max={ds.max():.4g}  |bs|={np.linalg.norm(b):.5g} |old|={np.linalg.norm(o):.5g} "
          f"ratio={np.linalg.norm(b)/np.linalg.norm(o):.4f}")
    for tag, bb, oo in (("all640", b, o), ("gate", b[:inter], o[:inter]), ("up", b[inter:], o[inter:])):
        perm = np.array([int(np.argmin(np.abs(oo - v))) for v in bb])
        describe(perm, tag)
    print()
