#!/usr/bin/env python3
"""Reproduce the OFFICIAL sparse-attention kernel (kernel.py:328-389) bit-exactly
from its own dumped inputs, then walk toward OUR output to find the first
diverging sub-step.

Official structure (per KV block, flash-style, all f32 accumators):
    acc_s = q @ kv_block^T          (bf16 x bf16 -> f32)
    acc_s *= softmax_scale
    reduce_max(acc_s, running max, clear=False)
    acc_s = exp(acc_s - max)
    reduce_sum(acc_s, running sum)
    acc_s_cast = bf16(acc_s)
    acc_o *= exp(max_prev - max_new)         <-- online rescale
    acc_o += acc_s_cast @ kv_block           (bf16 x bf16 -> f32)
  finally: acc_o /= sum_exp

Usage: offattn.py [block]
"""
import struct
import sys

import numpy as np

S, H, D = 15, 8, 512
SCALE = 0.04419417382415922
SINK = np.array([-0.09726219, -0.01387484, 0.07686966, -0.04399407,
                 -0.19732477, 0.12044567, -0.25218663, -0.03711864], dtype=np.float32)


def load_ref(path):
    d = open(path, "rb").read()
    o = 0
    out = {}
    while o + 24 <= len(d):
        n, k, sz = struct.unpack_from("<qqq", d, o)
        o += 24
        out[k] = np.frombuffer(d, dtype="<f4", count=sz, offset=o).copy()
        o += 4 * sz
    return out


def bf16(x):
    """round f32 -> bfloat16 -> f32 (round-to-nearest-even, as torch does)."""
    return (x.astype(np.float32).view(np.uint32).astype(np.uint64)
            + np.uint64(0x8000) + ((x.view(np.uint32) >> 16) & 1)
            ).astype(np.uint32).astype(np.uint32).view(np.float32)


def bf16_np(x):
    # torch's .to(torch.bfloat16).float(): round to nearest even on 16-bit truncation
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) & np.uint64(0xFFFF0000)
    return rounded.astype(np.uint32).view(np.float32)


def bits_equal(a, b):
    return (a.view(np.uint32) == b.view(np.uint32)).sum(), a.size


def main():
    block = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    R = load_ref("/home/ubuntu/ref_attn.bin")
    q = R[1].reshape(S, H, D)
    kv = R[2].reshape(S, D)
    idxs = R[3].reshape(S, S).astype(np.int32)
    out = R[4].reshape(S, H, D)

    for pos in (0, 1, 2, 3):
        ix = [int(t) for t in idxs[pos] if t >= 0]
        if not ix:
            continue
        k = kv[ix]                       # [n, D]
        # official: block-wise flash over the n keys
        acc_o = np.zeros((H, D), dtype=np.float32)
        m = np.full((H, 1), -np.inf, dtype=np.float32)
        se = np.zeros((H, 1), dtype=np.float32)
        for b0 in range(0, len(ix), block):
            kb = k[b0:b0 + block]
            s = (q[pos].astype(np.float32) @ kb.astype(np.float32).T) * np.float32(SCALE)
            m_new = np.maximum(m, s.max(axis=-1, keepdims=True))
            alpha = np.exp(m - m_new).astype(np.float32)
            alpha = np.where(np.isfinite(m), alpha, 1.0).astype(np.float32)
            p = np.exp((s - m_new).astype(np.float32)).astype(np.float32)
            acc_o = acc_o * alpha
            pc = bf16_np(p)                                   # the bf16 prob cast
            acc_o = acc_o + (pc.astype(np.float32) @ bf16_np(kb).astype(np.float32))
            se = se * alpha + p.sum(axis=-1, keepdims=True).astype(np.float32)
            m = m_new
        # attention sink: an extra key with no value, logit = sink
        se = se + np.exp((SINK.reshape(H, 1) - m).astype(np.float32)).astype(np.float32)
        oo = (acc_o / se).astype(np.float32)
        same, n = bits_equal(oo.reshape(-1), out[pos].reshape(-1).astype(np.float32).copy())
        print("pos%d block=%d  vs OFFICIAL out: bit-equal %d/%d   scale=%.6f"
              % (pos, block, same, n,
                 float((oo * out[pos]).sum() / (out[pos].astype(np.float64) ** 2).sum())))


if __name__ == "__main__":
    main()
