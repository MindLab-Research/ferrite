#!/usr/bin/env python3
"""Replay the block-scaled arm's OWN staged operands through the official fp4/fp8 semantics and
compare the result with what the arm actually wrote out.

This is the assumption-free tie-breaker. It consumes only instruments that already exist:

  * `<sfdump>/a.u8`    [36][128][128]  the staged A tile at K-stage 0, read back semantically
  * `<sfdump>/b.u8`    [36][5][128][64] the staged B tile (packed fp4)
  * `<sfdump>/sfa.u32` [36][128]        the activation SF words the MMA consumed
  * `<sfdump>/sfb.u32` [36][5][128]     the weight SF words
  * `<sfdump>/meta.txt`                 nseg / topk / dim / eid / counts / order
  * `<gu>/gateup.f32`  [m*topk][640]    the arm's raw output (the scatter's result)

and answers ONE question: is the arm's output equal to the product of the operands it staged?

  * PASS  => the MMA + epilogue + scatter are faithful to the staged data, so any remaining
             difference from the proven path (or from the official) must be in the STAGED CONTENT
             itself (then `sfdump_check.py` says exactly which byte is wrong);
  * FAIL  => the kernel computed something other than its own staged operands' product (the
             staging/delivery/descriptor side), independent of what those operands are.

The reference mirrors `ref_inference/kernel.py`'s fp4_gemm structure: per 32-K block a f32 partial
product of the e4m3 activation and the e2m1 weight, multiplied by that block's (sa * sb), summed in
K order. E4M3 contributes its scale (act_quant's per-32 scale), so the A side is byte x xsc.

Usage: sfdump_replay.py --sfdump DIR --gu DIR [--row -1] [--tol 0.02]
"""
import argparse
import os
import re
import sys

import numpy as np

SEG, BM, BK, NT, NPB = 36, 128, 128, 5, 64


def e2m1(c):
    mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    v = mag[c & 0x7]
    return np.where(c & 0x8, -v, v).astype(np.float32)


def e4m3(b):
    b = b.astype(np.uint32)
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    v = np.where(e == 0, np.ldexp(m.astype(np.float32), -9),
                 np.ldexp(1.0 + m.astype(np.float32) / 8.0, e.astype(np.int32) - 7))
    return np.where(s == 1, -v, v).astype(np.float32)


def e8m0(word_bytes):
    return np.ldexp(np.float32(1.0), word_bytes.astype(np.int32) - 127).astype(np.float32)


def parse_meta(path):
    txt = open(path).read()
    out = {}
    for key in ("nseg", "topk", "dim"):
        m = re.search(rf"{key}=(-?\d+)", txt)
        if m:
            out[key] = int(m.group(1))
    for key in ("eid", "counts", "order"):
        m = re.search(rf"{key}=\[(.*?)\]", txt)
        if m:
            out[key] = [int(x) for x in m.group(1).split(",") if x.strip() != ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfdump", required=True)
    ap.add_argument("--gu", required=True)
    ap.add_argument("--row", type=int, default=-1, help="which row of a multi-row dump (-1 = last)")
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()

    meta = parse_meta(os.path.join(a.sfdump, "meta.txt"))
    nseg, topk, dim = meta["nseg"], meta["topk"], meta["dim"]
    order = meta["order"]
    eid = meta["eid"]
    a_bytes = np.fromfile(os.path.join(a.sfdump, "a.u8"), dtype="u1").reshape(SEG, BM, BK)
    b_bytes = np.fromfile(os.path.join(a.sfdump, "b.u8"), dtype="u1").reshape(SEG, NT, BM, 64)
    sfa = np.fromfile(os.path.join(a.sfdump, "sfa.u32"), dtype="<u4").reshape(SEG, BM)
    sfb = np.fromfile(os.path.join(a.sfdump, "sfb.u32"), dtype="<u4").reshape(SEG, NT, BM)
    gu = np.fromfile(os.path.join(a.gu, "gateup.f32"), dtype="<f4")
    act = 2 * (dim // 16) if dim else 640
    # the dump's layout is [m][topk][act_slot]; act_slot = 2*inter_local, taken from the file size
    m_rows = gu.size // (topk * 640)
    print(f"meta: nseg={nseg} topk={topk} dim={dim}  gu has {m_rows} row(s); eid={eid[:nseg]}")
    row = m_rows - 1 if a.row < 0 else a.row
    gu = gu.reshape(m_rows, topk, 640)[row]

    blocks = dim // 32
    worst = 0.0
    n_bad = 0
    n_cmp = 0
    for seg in range(nseg):
        for r in range(BM):
            fi = seg * BM + r
            if fi >= len(order) or order[fi] < 0:
                continue
            flat = order[fi]
            av = e4m3(a_bytes[seg, r])                       # the 128 K values of this stage
            sa_bytes = np.asarray([sfa[seg, r]], dtype="<u4").view(np.uint8)  # 4 K-block scales
            for nt in range(NT):
                sb_all = np.asarray([sfb[seg, nt]], dtype="<u4").view(np.uint8).reshape(BM, 4)
                for half in (0, 1):
                    rl = half * 64 + np.arange(64)           # rows 0..63 = W1, 64..127 = W3
                    bpack = b_bytes[seg, nt, rl]             # packed fp4 for this 128-K stage
                    wv = np.empty((64, 128), dtype=np.float32)
                    wv[:, 0::2] = e2m1(bpack & 0x0F)         # low nibble = even K
                    wv[:, 1::2] = e2m1((bpack >> 4) & 0x0F)
                    acc = np.zeros(64, dtype=np.float32)
                    for blk in range(4):
                        k0 = blk * 32
                        part = (av[k0:k0 + 32][None, :] * wv[:, k0:k0 + 32]).sum(axis=1)
                        acc += part * float(e8m0(np.array([sa_bytes[blk]], dtype=np.uint8))[0]) \
                                     * e8m0(sb_all[rl, blk])
                    # the scatter's column mapping: gate half -> nt*64 + j, up half -> 320 + nt*64 + j
                    n = (nt * 64 + np.arange(64)) if half == 0 else (320 + nt * 64 + np.arange(64))
                    got = gu[flat % topk][n]
                    den = np.maximum(np.abs(acc), 1e-6)
                    rel = np.abs(got - acc) / den
                    worst = max(worst, float(np.median(rel)))
                    n_bad += int((rel > a.tol).sum())
                    n_cmp += rel.size
                    if n_cmp <= 64:
                        print(f"  seg{seg} r{r} nt{nt} half{half} flat={flat} "
                              f"acc[0:3]={np.round(acc[:3],4)} got[0:3]={np.round(got[:3],4)} "
                              f"median rel={np.median(rel):.4g}")
    print(f"compared {n_cmp} values; worst per-case median rel={worst:.4g}; "
          f"values over tol={n_bad}")
    print("VERDICT: " + ("PASS — the arm's output equals its own staged operands' product"
                         if n_bad == 0 else
                         "FAIL — the MMA/epilogue/scatter is NOT faithful to the staged operands"))
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
