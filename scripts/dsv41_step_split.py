#!/usr/bin/env python3
"""Steady-state kernel split out of an nsys sqlite export for the DSV4.1 decode step.

Why this exists: `nsys stats --report cuda_gpu_kern_sum` aggregates the WHOLE
run, and for this workload the weight-deserialization kernels (dequant /
bf16_to_f32) dominate that aggregate. Quoting the top rows of that report
therefore says nothing about the decode step. This tool takes the trailing
window of GPU activity instead, which is decode-only once the load and the
prefill are behind it, and reports per-kernel median / count / total / share.

Usage:
    nsys export --type sqlite -o /tmp/prof.sqlite /tmp/prof/many.nsys-rep
    python3 scripts/dsv41_step_split.py /tmp/prof.sqlite 320

The window is in milliseconds of trailing GPU time (default 320 ms); pick it
several times larger than one step so the split is stable.
"""
import sqlite3
import statistics
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    db = sys.argv[1]
    win_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 320.0

    con = sqlite3.connect(db)
    cur = con.cursor()
    tabs = [r[0] for r in cur.execute("select name from sqlite_master where type='table'")]
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tabs:
        print("no CUPTI_ACTIVITY_KIND_KERNEL table; tables present:", tabs[:16])
        return 1
    cols = [d[1] for d in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    name_col = "demangledName" if "demangledName" in cols else "name"
    rows = list(
        cur.execute(
            "select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
            "join StringIds s on s.id = k." + name_col
        )
    )
    if not rows:
        print("kernel table is empty:", db)
        return 1

    tmax = max(r[2] for r in rows)
    cut = tmax - int(win_ms * 1e6)
    sel = [(n, e - s) for n, s, e in rows if s >= cut]
    tot_ns = sum(d for _, d in sel)
    if not sel:
        print("no kernels in the trailing window; try a larger window")
        return 1

    agg = {}
    for n, d in sel:
        agg.setdefault(n, []).append(d / 1000.0)
    print(
        "window = last %.2fs -> %d kernels, GPU busy %.2f ms"
        % (win_ms / 1000.0, len(sel), tot_ns / 1e6)
    )
    print("%-54s %6s %9s %10s %7s" % ("kernel", "n", "med us", "total ms", "share"))
    for name, samples in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        print(
            "%-54s %6d %9.2f %10.3f %6.1f%%"
            % (
                name[:54],
                len(samples),
                statistics.median(samples),
                sum(samples) / 1000.0,
                100.0 * sum(samples) / (tot_ns / 1000.0),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
