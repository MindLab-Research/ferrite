#!/usr/bin/env python3
"""Bit-exact comparison of the dsv41 dump streams (ours vs the reference).

Record layout: [8B pos][8B kind][n x 4B payload], n from the table below.

BIT-EXACTNESS RULES (the trap that produced a wrong "routing is all zero"
reading earlier):
  * kind 3 is written by `gt_dump_u32` on our side (the payload IS int32 ids)
    while the reference writes the same ids as f32 values. Each side must be
    decoded in its OWN type; comparing raw words across encodings is nonsense.
  * every other kind is f32 on both sides, so compare RAW 4-byte words
    (bit equality), not parsed floats.

Usage: dumpcmp.py ours.bin ref.bin [kind ...]
"""
import struct
import sys

SIZES = {
    0: 5120, 1: 5120, 2: 5120, 3: 6, 4: 6, 5: 512, 6: 512, 7: 160,
    8: 320, 9: 32, 10: 64, 11: 64, 12: 16, 13: 16, 20: 40, 21: 4,
    40: 5120, 41: 5120, 42: 16160, 50: 4096, 51: 4096, 52: 1024, 53: 5120,
}
for _k in range(14, 20):
    SIZES[_k] = 32


def load(path):
    """{(pos, kind): raw bytes} -- the stream is truncated at the first unknown kind."""
    data = open(path, "rb").read()
    off = 0
    out = {}
    while off + 16 <= len(data):
        pos, kind = struct.unpack_from("<qq", data, off)
        off += 16
        n = SIZES.get(kind)
        if n is None:
            break
        out[(pos, kind)] = data[off:off + 4 * n]
        off += 4 * n
    return out


def words(raw):
    return struct.unpack_from("<%dI" % (len(raw) // 4), raw, 0)


def ids_of(raw, is_int):
    if is_int:
        return list(struct.unpack_from("<%di" % (len(raw) // 4), raw, 0))
    return [int(round(v)) for v in struct.unpack_from("<%df" % (len(raw) // 4), raw, 0)]


def main():
    ours, ref = sys.argv[1], sys.argv[2]
    only = set(int(x) for x in sys.argv[3:]) or None
    A, B = load(ours), load(ref)
    keys = sorted(set(A) | set(B), key=lambda kv: (kv[1], kv[0]))
    print("ours records=%d  ref records=%d" % (len(A), len(B)))
    kinds = sorted({k for _, k in keys})
    print("kinds ours=%s" % sorted({k for _, k in A}))
    print("kinds ref =%s" % sorted({k for _, k in B}))
    print()
    print("%-5s %-5s %8s %8s %12s %10s %10s" %
          ("kind", "pos", "words", "bitdiff", "maxabs", "scale", "note"))
    for kind in kinds:
        if only and kind not in only:
            continue
        for pos in sorted({p for p, k in keys if k == kind}):
            a, b = A.get((pos, kind)), B.get((pos, kind))
            if a is None or b is None:
                print("%-5d %-5d %8s %8s %12s %10s %10s" %
                      (kind, pos, "-" if a is None else len(a) // 4,
                       "-" if b is None else len(b) // 4, "-", "-",
                       "ours-missing" if a is None else "ref-missing"))
                continue
            n = min(len(a), len(b))
            if kind == 3:
                ai, bi = ids_of(a[:n], True), ids_of(b[:n], True)
                # the reference encodes ids as float values; verify that reading
                # them back as int32 would be nonsense (sanity, not a compare).
                bf = struct.unpack_from("<%df" % (n // 4), b, 0)
                fint = [int(round(v)) for v in bf]
                bitdiff = sum(1 for x, y in zip(ai, fint) if x != y)
                print("%-5d %-5d %8d %8d %12s %10s %10s" %
                      (kind, pos, n // 4, bitdiff, "-", "-",
                       "ids equal" if bitdiff == 0 else "IDS DIFFER %s vs %s" % (ai, fint)))
                continue
            wa, wb = words(a[:n]), words(b[:n])
            bitdiff = sum(1 for x, y in zip(wa, wb) if x != y)
            fa = struct.unpack_from("<%df" % (n // 4), a, 0)
            fb = struct.unpack_from("<%df" % (n // 4), b, 0)
            maxabs = max(abs(x - y) for x, y in zip(fa, fb))
            sab = sum(x * y for x, y in zip(fa, fb))
            sbb = sum(y * y for y in fb)
            sc = sab / sbb if sbb else float("nan")
            print("%-5d %-5d %8d %8d %12.6g %10.6f %10s" %
                  (kind, pos, n // 4, bitdiff, maxabs, sc,
                   "BIT-EXACT" if bitdiff == 0 else ""))


if __name__ == "__main__":
    main()
