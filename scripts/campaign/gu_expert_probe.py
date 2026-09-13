#!/usr/bin/env python3
"""Which expert's weights did the block-scaled arm actually use?

The arm's gate|up is wrong in a very specific way (right magnitude, element-wise uncorrelated with
the truth, not a permutation, and NOT explained by any of the layout/scale/delivery checks that all
pass). One mechanism produces exactly that: reading another expert's weight block. This script
computes the OFFICIAL gate|up (ref_inference/kernel.py's fp4_gemm structure, the same reference that
reproduces the proven path bit-exactly) for a candidate set of experts on the SAME activation, and
scores each candidate against what the arm wrote to each slot.

Candidates default to: the routed ids, every id +-1, the segment indices 0..nseg-1 (the "used the
segment index instead of the expert id" hypothesis), and 0..7.

Usage: gu_expert_probe.py --in-dir DIR --bs FILE.f32 [--cands 0,1,2] [--ckpt DIR] [--inter-rank 0]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

DIM, INTER_FULL, TOPK, ACT = 5120, 2304, 6, 640


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


def e8m0(bytes_):
    return np.ldexp(np.float32(1.0), bytes_.astype(np.int32) - 127).astype(np.float32)


def parse_meta(txt):
    out = {}
    for key in ("eid", "counts", "order"):
        m = re.search(rf"{key}=\[(.*?)\]", txt)
        if m:
            out[key] = [int(x) for x in m.group(1).split(",") if x.strip() != ""]
    m = re.search(r"topk=(\d+)", txt)
    if m:
        out["topk"] = int(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="a gate|up dump dir with x.f32/ids.i32/xq4.u8/xsc4.f32")
    ap.add_argument("--bs", required=True, help="the arm's gateup.f32 to explain")
    ap.add_argument("--cands", default="")
    ap.add_argument("--ckpt", default="/opt/dlami/nvme/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--inter-rank", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()

    x = np.fromfile(f"{a.in_dir}/x.f32", dtype="<f4")
    ids = np.fromfile(f"{a.in_dir}/ids.i32", dtype="<i4")[:TOPK]
    xq = np.fromfile(f"{a.in_dir}/xq4.u8", dtype="u1")[:DIM]
    xs = np.fromfile(f"{a.in_dir}/xsc4.f32", dtype="<f4")[: DIM // 32]
    bs = np.fromfile(a.bs, dtype="<f4")[: TOPK * ACT].reshape(TOPK, ACT)
    print(f"ids={ids.tolist()}  xq4={xq.size} xsc4={xs.size}  bs shape={bs.shape}")
    print(f"bs per-slot norms: {[round(float(np.linalg.norm(bs[i])), 3) for i in range(TOPK)]}")

    if a.cands:
        cands = [int(v) for v in a.cands.split(",")]
    else:
        cands = sorted({int(v) for v in ids} | {int(v) + d for v in ids for d in (-1, 1)} | set(range(8)))
    print(f"candidates ({len(cands)}): {cands}")

    idx_path = os.path.join(a.ckpt, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"]
    import torch
    from safetensors import safe_open

    av = e4m3(xq)                                  # the activation codes, per K
    blocks = DIM // 32
    shard = INTER_FULL // 8
    r0 = a.inter_rank * shard
    L = a.layer
    ref = np.zeros((len(cands), ACT), dtype=np.float32)
    for ci, e in enumerate(cands):
        for half, which in ((0, "w1"), (1, "w3")):
            k = f"layers.{L}.ffn.experts.{e}.{which}.weight"
            ks = f"layers.{L}.ffn.experts.{e}.{which}.scale"
            if k not in wmap:
                print(f"  expert {e}: missing {k}")
                continue
            with safe_open(os.path.join(a.ckpt, wmap[k]), framework="pt") as f:
                wt = f.get_tensor(k).view(torch.uint8).numpy()
                ws = f.get_tensor(ks).view(torch.uint8).numpy()
            sw = wt[r0:r0 + shard]
            ss = ws[r0:r0 + shard]
            sw = np.concatenate([sw, np.zeros((320 - sw.shape[0], sw.shape[1]), np.uint8)])[:320]
            ss = np.concatenate([ss, np.zeros((320 - ss.shape[0], ss.shape[1]), np.uint8)])[:320]
            wv = np.empty((sw.shape[0], DIM), dtype=np.float32)
            wv[:, 0::2] = e2m1(sw & 0x0F)
            wv[:, 1::2] = e2m1((sw >> 4) & 0x0F)
            acc = np.zeros(sw.shape[0], dtype=np.float32)
            for b in range(blocks):
                part = (av[b * 32:(b + 1) * 32][None, :] * wv[:, b * 32:(b + 1) * 32]).sum(axis=1)
                acc += part * xs[b] * e8m0(ss[:, b])
            ref[ci, half * 320:(half + 1) * 320] = acc

    print("\nscore matrix: rows = bs slot, cols = candidate expert (corr | norm ratio)")
    hdr = "      " + "".join(f"{e:>8d}" for e in cands)
    print(hdr)
    best = []
    for i in range(TOPK):
        row = []
        for ci in range(len(cands)):
            c = float(np.corrcoef(bs[i], ref[ci])[0, 1])
            row.append(c)
        j = int(np.nanargmax(np.abs(row)))
        best.append((i, cands[j], row[j], float(np.linalg.norm(bs[i]) / max(np.linalg.norm(ref[j]), 1e-9))))
        print(f"bs[{i}] " + "".join(f"{v:>8.3f}" for v in row))
    print("\nbest match per slot (slot, expert, corr, norm ratio):")
    for i, e, c, nr in best:
        selfs = ""
        if e == int(ids[i]):
            selfs = "  <- its OWN routed id"
        print(f"  slot {i}: expert {e:>4}  corr={c:+.4f}  norm_ratio={nr:.3f}{selfs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
