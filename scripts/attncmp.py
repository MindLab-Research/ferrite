#!/usr/bin/env python3
"""Compare the OFFICIAL sparse-attention operator against ours, per position.

Input A: ~/ref_attn.bin -- written by ~/ref_attn.py with EXPLICIT headers:
          [8B n][8B kind][8B size][size x 4B f32]  (repeated)
          kind 1 = q, 2 = kv, 3 = topk_idxs, 4 = out   (all for the prefill call,
          i.e. every position 0..P-1 at once)
Input B: our dump (DSV41_GT_XDUMP) -- fixed size table, kind 50 = q post-rope,
          kind 51 = attention output pre-wo_a. Ours holds THIS RANK's 16 heads
          (nlh*hd = 4096); the official holds all 128 heads (32768), so the
          comparable slice is heads [0, nlh) if the TP shard is contiguous.

Usage: attncmp.py ref_attn.bin our.bin [pos ...]
"""
import struct
import sys

OUR_SIZES = {
    0: 5120, 1: 5120, 2: 5120, 3: 6, 4: 6, 5: 512, 6: 512, 7: 160,
    8: 320, 9: 32, 10: 64, 11: 64, 12: 16, 13: 16, 20: 40, 21: 4,
    22: 384, 23: 5120, 40: 5120, 41: 5120, 42: 16160,
    50: 4096, 51: 4096, 52: 1024, 53: 5120,
}
for _k in range(14, 20):
    OUR_SIZES[_k] = 32


def load_ref(path):
    """{(call_n, kind): [f32...]} with the explicit header."""
    data = open(path, "rb").read()
    off = 0
    out = {}
    while off + 24 <= len(data):
        n, kind, size = struct.unpack_from("<qqq", data, off)
        off += 24
        vals = struct.unpack_from("<%df" % size, data, off)
        off += 4 * size
        out[(n, kind)] = list(vals)
    return out


def load_ours(path, kind):
    data = open(path, "rb").read()
    off = 0
    out = {}
    while off + 16 <= len(data):
        pos, k = struct.unpack_from("<qq", data, off)
        off += 16
        n = OUR_SIZES.get(k)
        if n is None:
            break
        if k == kind:
            out[pos] = list(struct.unpack_from("<%df" % n, data, off))
        off += 4 * n
    return out


def cmp(a, b, label):
    n = min(len(a), len(b))
    wa = struct.unpack_from("<%dI" % n, struct.pack("<%df" % n, *a[:n]), 0)
    wb = struct.unpack_from("<%dI" % n, struct.pack("<%df" % n, *b[:n]), 0)
    bd = sum(1 for x, y in zip(wa, wb) if x != y)
    sbb = sum(y * y for y in b[:n])
    sc = sum(x * y for x, y in zip(a[:n], b[:n])) / sbb if sbb else float("nan")
    print("   %-22s n=%-6d bitdiff=%-6d scale=%.6f%s"
          % (label, n, bd, sc, "  BIT-EXACT" if bd == 0 else ""))


def main():
    ref = load_ref(sys.argv[1])
    ours = sys.argv[2]
    poss = [int(x) for x in sys.argv[3:]] or list(range(0, 6))
    keys = sorted({k for _, k in ref})
    print("ref kinds:", keys)
    # shapes: infer h/d from the official's q (b, m, h, d)
    if (0, 1) in ref:
        print("ref q size=%d kv size=%d idxs size=%d out size=%d"
              % (len(ref[(0, 1)]), len(ref[(0, 2)]), len(ref[(0, 3)]), len(ref[(0, 4)])))
    q_ours = load_ours(ours, 50)
    o_ours = load_ours(ours, 51)
    for pos in poss:
        print("pos %d:" % pos)
        if (0, 4) in ref and pos in o_ours:
            cmp(o_ours[pos], ref[(0, 4)], "attn out (ours vs ref)")


if __name__ == "__main__":
    main()
