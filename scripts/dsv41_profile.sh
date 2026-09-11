#!/usr/bin/env bash
# Decode-only per-kernel breakdown for DSV41, by DIFFERENCING two nsys profiles.
#
# Why differencing: an nsys kernel table averaged over a whole run includes the
# prefill, where several kernels are called with large row counts and are far
# slower than in decode. Averaging those in inflates the decode figure (this bit
# us: hc_mixes "49% of the step" was an artifact). Subtracting a 1-token run from
# an N-token run leaves the decode-only net cost.
#
# Why NCCL mode: the P2P all-reduce's publish kernel SPINS waiting for the peers'
# stamps. Under nsys's per-node graph tracing that spin is amplified hundreds of
# times (measured: 240 s of wall clock for 69 steps), so profile with the AR in
# NCCL mode (do NOT pass FERRITE_P2P=1 / DSV41_AR_V5=1) and attribute the AR
# separately.
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

prof() { # $1 = max_tokens, $2 = output tag
  echo "== profiling max_tokens=$1 -> $OUT/$2 =="
  timeout -s KILL 900 "$NSYS" profile --trace=cuda --cuda-graph-trace=node --sample=none \
    -o "$OUT/$2" --force-overwrite=true \
    env CUDA_VISIBLE_DEVICES="$GPUS" DSV41_MODEL_DIR="$MODEL_DIR" DSV41_KERNELS="$KERNELS" \
    "$BIN" --prompt "$PROMPT" --max-tokens "$1" --tp 8 >"$OUT/$2.log" 2>&1 || true
  grep -E "DECODE|\[dsv41\] decode" "$OUT/$2.log" | tail -3 || true
  # CSV, never the table (kernel names contain spaces)
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv "$OUT/$2.nsys-rep" 2>/dev/null \
    | tail -n +2 >"$OUT/$2.csv"
}

prof 1 "$OUT/one.csv.tag" 2>/dev/null || true
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
print(f"{'share':>7} {'calls/step':>10} {'us/call':>9}  kernel")
for dt, di, k, avg in rows[:15]:
    print(f"{dt/tot*100:6.1f}% {di/steps:10.0f} {avg:9.1f}  {k[:52]}")
PY
echo
echo "note: attribute the all-reduce separately (NCCL mode here); for absolute per-call"
echo "times use an isolated repro - multi-device nsys times are not trustworthy."
