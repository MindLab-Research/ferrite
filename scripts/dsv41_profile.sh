#!/usr/bin/env bash
# Decode-only per-kernel breakdown for DSV41, by DIFFERENCING two nsys profiles.
#
# Why differencing: an nsys kernel table averaged over a whole run includes the
# prefill, where several kernels are called with large row counts and are far
# slower than in decode. Averaging those in inflates the decode figure (this bit
# us: hc_mixes "49% of the step" was an artifact). Subtracting a 1-token run from
# an N-token run leaves the decode-only net cost.
#
# Why the host-barrier AR (DSV41_AR_V5=0): the device-side all-reduce's publish
# kernel SPINS waiting for the peers' stamps. Under nsys's per-node graph tracing
# that spin is amplified hundreds of times (measured: 240 s of wall clock for 69
# steps), so this script pins DSV41_AR_V5=0 - which it must do EXPLICITLY now that
# the device-side AR is the default - and attributes the AR separately (90 calls x
# ~8 us with v5 against the barrier's measured per-call time).
# DSV41_GRAPH_STEP=0 for the same reason: a whole-step capture records ~400 nodes,
# which is exactly what makes the node tracing expensive, and every kernel's cost is
# identical between the two paths (verified by DSV41_TOKTRACE: one request is
# bit-identical), so the per-kernel table is valid for both.
#
# Also: never parse the default table output with awk - kernel names contain
# spaces and the columns shift. Use --format csv.
#
# Usage:  scripts/dsv41_profile.sh [N_TOKENS] [OUTDIR]
#   then:  nsys stats --report cuda_gpu_kern_sum --format csv <rep> for the raw tables
set -euo pipefail

N="${1:-40}"
OUT="${2:-/tmp/dsv41-prof}"
NSYS="${NSYS:-/usr/local/cuda-13.2/bin/nsys}"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
KERNELS="${DSV41_KERNELS:-$PWD/kernels/cuda/libferrite_kernels.so}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BIN="${BIN:-./target/release/dsv41-run}"
PROMPT="${PROMPT:-请写一篇关于人工智能的短文}"

mkdir -p "$OUT"
if pgrep -x dsv41-run >/dev/null 2>&1; then
  echo "refusing to start: a dsv41-run is already alive (the GPUs must be exclusive)" >&2
  exit 1
fi

# Both pins are load-bearing and must stay: with the graph on, ar_v5 is forced on (a host
# barrier is not a CUDA call and cannot be captured), and the v5 publish spin under nsys's
# per-node graph tracing is a documented 300x pathology - 240 s for 69 steps. The per-kernel
# costs are identical in both modes, so this is the only usable attribution configuration;
# subtract ~1500-2000 launches x (3.05 us standalone - 0.2 us in-graph) to read the graph-on
# step.
#
# NOTE (2026-09-11): these comments used to sit INSIDE the backslash-continued command below.
# A comment line has no trailing backslash, so it terminated the command: what actually ran
# was `env VAR=0 VAR2=0` (which merely prints the environment - that env dump was the tell)
# and the binary then ran as a separate command with no nsys around it. nsys profiled `env`,
# so every report had zero CUDA kernels and the CSV was empty. Comments stay outside the
# continuation. Also: never swallow the stats stderr - that is what hid this for hours - and
# fail loudly on an empty CSV, because a zero-row diff silently prints "0.0 ms".
prof() { # $1 = max_tokens, $2 = output tag
  echo "== profiling max_tokens=$1 -> $OUT/$2 =="
  timeout -s KILL 900 "$NSYS" profile --trace=cuda --cuda-graph-trace=node --sample=none \
    -o "$OUT/$2" --force-overwrite=true \
    env CUDA_VISIBLE_DEVICES="$GPUS" DSV41_MODEL_DIR="$MODEL_DIR" DSV41_KERNELS="$KERNELS" \
    DSV41_AR_V5=0 DSV41_GRAPH_STEP=0 \
    "$BIN" --prompt "$PROMPT" --max-tokens "$1" --tp 8 >"$OUT/$2.log" 2>&1 || true
  grep -E "DECODE|\[dsv41\] step" "$OUT/$2.log" | tail -3 || true
  # CSV, never the table (kernel names contain spaces)
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv "$OUT/$2.nsys-rep" \
    | tail -n +2 >"$OUT/$2.csv"
  local rows; rows=$(wc -l <"$OUT/$2.csv")
  echo "   [$2] $rows kernel rows"
  if [ "$rows" -lt 5 ]; then
    echo "FATAL: $OUT/$2.csv has $rows rows - the profile did not capture the binary." >&2
    echo "       did the binary actually run? see $OUT/$2.log (last lines):" >&2
    tail -5 "$OUT/$2.log" >&2
    exit 1
  fi
}

prof 1 one
prof "$N" many

python3 - "$OUT/one.csv" "$OUT/many.csv" "$N" <<'PY'
import csv, sys
one, many, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

def load(p):
    d = {}
    for r in csv.reader(open(p)):
        if len(r) < 5:
            continue
        try:
            total, inst = float(r[1]), int(r[2])
        except ValueError:
            continue
        d[" ".join(r[4:]).strip()] = (inst, total)
    return d

a, b = load(one), load(many)
rows = []
for k, (i2, t2) in b.items():
    i1, t1 = a.get(k, (0, 0.0))
    dt, di = t2 - t1, i2 - i1
    if dt > 0:
        rows.append((dt, di, k, dt / di if di else 0.0))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows) or 1.0
steps = max(n - 1, 1)
print(f"decode-only net GPU time = {tot/1e6:.1f} ms over {steps} steps / 8 ranks"
      f" = {tot/1e6/steps/8:.2f} ms per step per rank")
# nsys reports durations in NANOSECONDS, so both the total and the per-call mean are
# ns; the per-call column used to be printed raw under a "us/call" header, which read
# as absurd (hc_mixes showed 153182.6). Divide by 1000 and say what each column means.
# Note the instances and the durations are both summed over the ranks (one process
# hosts all eight here), so "calls/step" counts all ranks together and the per-call
# mean is each rank's own cost.
print(f"{'share':>7} {'calls/step':>10} {'us/call':>9}  kernel")
for dt, di, k, avg in rows[:15]:
    print(f"{dt/tot*100:6.1f}% {di/steps:10.0f} {avg/1000:9.1f}  {k[:52]}")
PY
echo
echo "note: attribute the all-reduce separately (NCCL mode here); for absolute per-call"
echo "times use an isolated repro - multi-device nsys times are not trustworthy."
