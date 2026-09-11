#!/usr/bin/env bash
# In-graph NODE-GAP measurement for the DSV4.1 whole-step CUDA graph (--tp 1 only).
#
# WHY THIS EXISTS (the 0.2 vs 1.5 us question, see STATUS.md "图 gap 的口径风险")
#   The per-step kernel-launch floor is ~1500-2000 launches. Under host launches
#   that costs ~3.05 us each (~5-6 ms/step, 13-16%). Once the whole-step graph is
#   replayed, the launch is recorded in the graph and the per-node cost is the
#   SLOT GAP between consecutive graph nodes. Two figures are on the table and
#   they differ by 7x:
#       ~0.2 us/node  (measured in the GLM graph)  -> floor is ~0.16 ms (2.5%)
#       ~1.5 us/node  (audited ramp-down residual) -> floor is ~1.2 ms  (18%)
#   The verdict decides whether "node reduction" (fusion / fewer launches) is a
#   headline lever or a rounding error, so it has to be MEASURED, not inferred.
#
# WHY --tp 1 (and why the other configs cannot be used)
#   * The device-side AR v5 (default ON, and FORCED on whenever the graph is on:
#     tp.rs ar_v5() = `graph || env`) PUBLISH-SPINS waiting for the peers' stamps.
#     Under nsys's `--cuda-graph-trace=node` that spin is amplified ~300x (measured:
#     240 s of wall clock for 69 steps). At --tp 8 this is fatal to profiling.
#   * At --tp 1 the single-device path is taken (dsv41-run.rs:162 `if tp > 1`) and
#     that path constructs DevChain WITHOUT a comm (chain_dev.rs:199), so there is
#     no peer to wait for: world=1 means the publish chain polls its OWN stamp, so
#     nothing is amplified. The graph still captures (chain_dev.rs:1716 wants it,
#     :1720 gates on decode_steps >= 1), so we get the real in-graph node sequence.
#   * Because there is no comm at --tp 1 there should be no dsv41_ar_* rows at all;
#     the ar/event predecessor exclusion below is belt-and-braces for the case where
#     one appears (e.g. a future tp=1 collective), and it is what keeps a spin gap
#     out of the median if a build ever emits one.
#
# HOW THE GAP IS DEFINED
#   Group the traced kernels by (deviceId, streamId), sort by start. For each
#   consecutive pair the gap is `next.start - prev.end`. Negative gaps (an overlap,
#   or a clock artifact) are dropped; a pair whose PREDECESSOR is an ar/event-class
#   kernel is dropped too (see EXCLUDE_RE), because that gap is a wait for a
#   collective, not a graph node slot. The headline number is the MEDIAN of the
#   remaining gaps on the BUSIEST stream (the one with the most kernels).
#
# CRITICAL: the gap is only meaningful if the rows really ARE graph nodes.
#   With `--cuda-graph-trace=node` nsys stamps each expanded node with graphId /
#   graphNodeId. We filter on `graphNodeId >= 0` and treat ZERO in-graph rows as a
#   HARD FAILURE: falling back to the time window there would silently measure the
#   HOST launch gaps (~2-5 us), which look plausible and would answer the 0.2 vs
#   1.5 us question with the wrong number. That is exactly the "zero rows must fail
#   loudly" lesson from dsv41_profile.sh. (Older nsys builds that do not export the
#   graph columns at all get a loud WARNING and an "UNRELIABLE" verdict instead.)
#
# Usage:  scripts/dsv41_node_gap.sh [N_TOKENS] [OUTDIR]
#           N_TOKENS default 31, OUTDIR default /tmp/dsv41-node-gap
# Env overrides:
#           NSYS, BIN, DSV41_MODEL_DIR, DSV41_KERNELS, CUDA_VISIBLE_DEVICES, PROMPT
#           GAP_WINDOW_NS  trailing window when the graph columns are missing (def 30e9)
#           GAP_SPLIT_NS   inter-step split for the cross-check (def 100000 = 100 us)
#           GAP_EXCLUDE_RE predecessor exclusion regex (def ar/event/nccl/spin class)
set -euo pipefail

N_TOKENS="${1:-31}"
OUT="${2:-/tmp/dsv41-node-gap}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/target/release/dsv41-run}"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
KERNELS="${DSV41_KERNELS:-$ROOT/kernels/cuda/libferrite_kernels.so}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
PROMPT="${PROMPT:-1+1=}"
WIN_NS="${GAP_WINDOW_NS:-30000000000}"
SPLIT_NS="${GAP_SPLIT_NS:-100000}"
EXCLUDE_RE="${GAP_EXCLUDE_RE:-ar_reduce|ar_publish|_ar_|nccl|spin|barrier|stamp|event}"

if [ -z "${NSYS:-}" ]; then
  for c in /usr/local/cuda-13.2/bin/nsys /usr/local/cuda/bin/nsys "$(command -v nsys 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then NSYS="$c"; break; fi
  done
fi
if [ -z "${NSYS:-}" ] || [ ! -x "$NSYS" ]; then
  echo "FATAL: nsys not found; set NSYS=/path/to/nsys" >&2
  exit 1
fi

mkdir -p "$OUT"

# The GPUs must be exclusive: a stray dsv41-run would share the device and pollute
# both the timeline and the step-time cross-check.
if pgrep -x dsv41-run >/dev/null 2>&1; then
  echo "refusing to start: a dsv41-run is already alive (the GPU must be exclusive)" >&2
  exit 1
fi
[ -x "$BIN" ]     || { echo "FATAL: missing binary $BIN (cargo build --release first)" >&2; exit 1; }
[ -f "$KERNELS" ] || { echo "FATAL: missing kernels .so $KERNELS" >&2; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "FATAL: missing model dir $MODEL_DIR" >&2; exit 1; }

echo "== profiling tp=1 + graph ON : $N_TOKENS tokens -> $OUT =="
# NOTE (dsv41_profile.sh cost us hours on this): a comment line inside a
# backslash-continued command has no trailing backslash and TERMINATES it - what
# ran was bare `env` and nsys profiled that, giving an empty report. All comments
# stay outside the continuation, and stderr goes to the log (never /dev/null).
timeout -s KILL 900 "$NSYS" profile --trace=cuda --cuda-graph-trace=node --sample=none \
  -o "$OUT/ng" --force-overwrite=true \
  env CUDA_VISIBLE_DEVICES="$GPU" \
      DSV41_MODEL_DIR="$MODEL_DIR" \
      DSV41_KERNELS="$KERNELS" \
      DSV41_GRAPH_STEP=1 \
      "$BIN" --prompt "$PROMPT" --max-tokens "$N_TOKENS" --tp 1 \
  >"$OUT/ng.log" 2>&1 || true

tail -5 "$OUT/ng.log" || true
if [ ! -f "$OUT/ng.nsys-rep" ]; then
  echo "FATAL: no $OUT/ng.nsys-rep - nsys did not profile the binary. Log tail:" >&2
  tail -20 "$OUT/ng.log" >&2
  exit 1
fi

# The measured decode step time is the cross-check target. The binary prints
#   [dsv41] DECODE N tokens in Xs = Y tok/s (Z ms/token)
# and Z is decode-only (timed from t_dec, dsv41-run.rs:239-249).
MS_TOK="$(grep -oE '\(([0-9.]+) ms/token\)' "$OUT/ng.log" | tail -1 | grep -oE '[0-9.]+' || true)"
if [ -z "$MS_TOK" ]; then MS_TOK=0; echo "WARNING: no 'ms/token' line in the log (cross-check disabled)" >&2; fi
echo "   measured decode step = ${MS_TOK} ms/token"

echo "== exporting sqlite =="
"$NSYS" export --type=sqlite --force-overwrite=true -o "$OUT/ng.sqlite" "$OUT/ng.nsys-rep" \
  >"$OUT/export.log" 2>&1
if [ ! -s "$OUT/ng.sqlite" ]; then
  echo "FATAL: sqlite export failed or is empty. export.log:" >&2
  cat "$OUT/export.log" >&2
  exit 1
fi

python3 - "$OUT/ng.sqlite" "$WIN_NS" "$MS_TOK" "$N_TOKENS" "$EXCLUDE_RE" "$SPLIT_NS" <<'PY'
import re
import sqlite3
import statistics
import sys

db, win_ns, measured_ms, n_tok, exclude_re, split_ns = sys.argv[1:7]
win_ns = int(float(win_ns))
measured_ms = float(measured_ms)
n_tok = int(n_tok)
split_ns = int(float(split_ns))
EX = re.compile(exclude_re, re.I)

con = sqlite3.connect(db)
cur = con.cursor()
cols = [d[1] for d in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
if not cols:
    print("FATAL: no CUPTI_ACTIVITY_KIND_KERNEL table in", db)
    sys.exit(1)
name_col = next((c for c in ("shortName", "demangledName", "name") if c in cols), None)
if name_col is None:
    print("FATAL: kernel table has no name column; cols =", cols[:16])
    sys.exit(1)

tmax = cur.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
if tmax is None:
    print("FATAL: kernel table is empty; the profile captured no CUDA kernels")
    sys.exit(1)

# Prefer the graph-node stamp: it isolates the captured nodes and EXCLUDES the
# prefill and the weight-load kernels, which would otherwise show up as giant
# inter-region gaps. Falling back to a window silently answers the wrong question,
# so a present-but-empty graph column set is fatal, not a fallback.
has_graph_cols = "graphId" in cols and "graphNodeId" in cols
n_in_graph = 0
if has_graph_cols:
    n_in_graph = cur.execute(
        "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE graphNodeId IS NOT NULL AND graphNodeId >= 0"
    ).fetchone()[0]

unreliable = False
if has_graph_cols and n_in_graph > 0:
    where = "k.graphNodeId IS NOT NULL AND k.graphNodeId >= 0"
    scope = "in-graph nodes only (graphNodeId >= 0)"
elif has_graph_cols and n_in_graph == 0:
    print("FATAL: no in-graph nodes in the trace (graphNodeId >= 0 matched 0 rows).")
    print("       The whole-step graph did NOT capture. Check: --tp 1, DSV41_GRAPH_STEP=1,")
    print("       and that decode actually ran (look for the DECODE line in ng.log).")
    print("       Refusing to fall back to the time window: its gaps are HOST launch")
    print("       gaps (~2-5 us) and would be reported as if they were node slots.")
    sys.exit(1)
else:
    # nsys build without the graph columns: only the trailing window is available.
    where = "k.start >= %d" % (tmax - win_ns)
    scope = "trailing window last %.1fs (NO graph columns in this nsys export)" % (win_ns / 1e9)
    unreliable = True
    print("WARNING: this nsys export has no graphId/graphNodeId columns (cols=%s...)." % cols[:10])
    print("         The gaps below are the trailing-window gaps and MAY be HOST launch")
    print("         gaps, not in-graph node slots. Treat the verdict as UNRELIABLE.")

rows = cur.execute(
    "SELECT k.deviceId, k.streamId, s.value, k.start, k.end "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.%s "
    "WHERE %s ORDER BY k.deviceId, k.streamId, k.start" % (name_col, where)
).fetchall()
if not rows:
    print("FATAL: no kernels matched scope =", scope)
    sys.exit(1)

groups = {}
for dev, stm, name, start, end in rows:
    groups.setdefault((dev, stm), []).append((name, start, end))

print()
print("== in-graph node gap: tp=1 + DSV41_GRAPH_STEP=1 ==")
print("scope      : %s" % scope)
print("kernels    : %d rows over %d (device, stream) groups" % (len(rows), len(groups)))

# Per-group gap statistics. gap = next.start - prev.end; drop gaps < 0 (overlap /
# clock artifact) and drop pairs whose predecessor is an ar/event-class kernel (that
# gap is a collective wait, not a graph slot).
per = []
for (dev, stm), ks in groups.items():
    ks.sort(key=lambda r: r[1])
    gaps = []
    n_excl = n_neg = 0
    for (pn, _ps, pe), (nn, ns, _ne) in zip(ks, ks[1:]):
        g = ns - pe
        if g < 0:
            n_neg += 1
            continue
        if EX.search(pn):
            n_excl += 1
            continue
        gaps.append(g)
    per.append(dict(dev=dev, stm=stm, nk=len(ks), span=ks[-1][2] - ks[0][1],
                    gaps=gaps, n_excl=n_excl, n_neg=n_neg,
                    ksum=sum(e - s for _n, s, e in ks),
                    first=ks[0][1], last=ks[-1][2]))
per.sort(key=lambda d: -d["nk"])
main = per[0]

print()
print("%-4s %-6s %7s %10s %7s %8s %8s %8s" %
      ("dev", "stream", "nodes", "span_ms", "gaps", "med_us", "mean_us", "p90_us"))
for d in per[:8]:
    g = d["gaps"]
    med = statistics.median(g) / 1000.0 if g else float("nan")
    mean = statistics.fmean(g) / 1000.0 if g else float("nan")
    p90 = (sorted(g)[int(0.9 * (len(g) - 1))] / 1000.0) if g else float("nan")
    print("%-4d %-6d %7d %10.3f %7d %8.3f %8.3f %8.3f" %
          (d["dev"], d["stm"], d["nk"], d["span"] / 1e6, len(g), med, mean, p90))

print()
print("MAIN stream = device %d stream %d (%d nodes); predecessor-excluded gaps: %d"
      % (main["dev"], main["stm"], main["nk"], main["n_excl"]))
mg = main["gaps"]
if not mg:
    print("FATAL: main stream has no usable gaps (all negative or all excluded)")
    sys.exit(1)
med_us = statistics.median(mg) / 1000.0
mean_us = statistics.fmean(mg) / 1000.0

print("  median gap = %.3f us   mean = %.3f us   n = %d" % (med_us, mean_us, len(mg)))
print()

# ---- verdict: which caliber does the measurement support? -------------------
print("== VERDICT ==")
print("  main-stream median node gap = %.3f us" % med_us)
if unreliable:
    print("  >> UNRELIABLE (no graph columns): the number may be a host launch gap.")
elif med_us <= 0.3:
    print("  >> 0.2 us CALIBER HOLDS (<= 0.3 us). In-graph launches are near-free.")
    print("     Node reduction is DOWNGRADED: ~1500-2000 nodes x %.2f us ~= %.2f ms/step."
          % (med_us, 1800 * med_us / 1000.0))
    print("     Look instead at the real per-kernel cost (the算子 work, not the launches).")
elif med_us >= 1.5:
    print("  >> 1.5 us CALIBER HOLDS (>= 1.5 us). Node slots are expensive.")
    print("     Node reduction becomes a HEADLINE lever: ~1500-2000 nodes x %.2f us ~= %.2f ms/step."
          % (med_us, 1800 * med_us / 1000.0))
    print("     Priorities: hc family merge-back, gemm_fp8 node count, act-cpasync quant.")
else:
    print("  >> IN BETWEEN (0.3 < median < 1.5 us): node reduction is MEDIUM priority.")
    print("     ~1500-2000 nodes x %.2f us ~= %.2f ms/step." % (med_us, 1800 * med_us / 1000.0))
print()

# ---- cross-check: sum(kernel) + sum(gap) == span, and span ~ measured step ---
print("== CROSS-CHECK (kernel + gap vs measured step) ==")
main_ks = sorted(groups[(main["dev"], main["stm"])], key=lambda r: r[1])
# Split the busiest stream into per-step segments on large idle gaps (between two
# graph replays the host is off doing other work, so the gap is >> a node slot).
segs = []
cur_seg = [main_ks[0]]
for a, b in zip(main_ks, main_ks[1:]):
    if b[1] - a[2] > split_ns:
        segs.append(cur_seg)
        cur_seg = [b]
    else:
        cur_seg.append(b)
segs.append(cur_seg)

kern_sum = sum(e - s for _n, s, e in main_ks) / 1e6
span_sum = (main_ks[-1][2] - main_ks[0][1]) / 1e6
print("  main stream: sum(kernel) = %.3f ms, span(first..last) = %.3f ms" % (kern_sum, span_sum))
print("  => sum(gap) = span - sum(kernel) = %.3f ms  (identity: kernels + gaps == span)"
      % (span_sum - kern_sum))
print("  segments split at gap > %d ns: %d segment(s); expected decode steps ~ %d"
      % (split_ns, len(segs), max(n_tok - 1, 1)))
spans = sorted((s[-1][2] - s[0][1]) / 1e6 for s in segs)
# The LAST segment is often a partial step (the run ends mid-graph); drop it from
# the step estimate, and drop the first if it is the capture step (which executes
# once at record time).
full = spans[:-1] if len(spans) > 1 else spans
if full:
    med_span = statistics.median(full)
    print("  per-segment span: median = %.3f ms (n=%d, excluded last/partial)"
          % (med_span, len(full)))
    if measured_ms > 0:
        ratio = med_span / measured_ms
        print("  measured step = %.3f ms/token  ->  segment/measured = %.2fx" % (measured_ms, ratio))
        if 0.6 <= ratio <= 1.4:
            print("  >> CONSISTENT: the graph segment reproduces the measured step.")
        else:
            print("  >> MISMATCH: the traced segment is %.2fx the measured step - the graph"
                  % ratio)
            print("     may not cover the whole step, or the window/scope is off.")
PY

echo
echo "artifacts: $OUT/ng.nsys-rep  $OUT/ng.sqlite  $OUT/ng.log"
echo "drill down: $NSYS stats --report cuda_gpu_kern_sum --format csv $OUT/ng.nsys-rep"
