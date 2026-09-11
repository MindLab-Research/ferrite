#!/usr/bin/env bash
# One-arm driver for DSV4.1 serve experiments.
#
# Launches a serve with a fixed environment plus any extra VAR=VALUE pairs given
# on the command line, drives a fixed four-prompt set, prints the answers, the
# per-step latency percentiles (the only accepted timing basis: the per-step
# "[dsv41] step pos=N" lines, never a segment average) and the fault count, then
# tears the serve down. Run two arms back to back in one session for an A/B.
#
# Usage:  scripts/dsv41_serve_ab.sh <tag> [VAR=VALUE ...]
#   scripts/dsv41_serve_ab.sh base
#   scripts/dsv41_serve_ab.sh vec DSV41_GEMV_FP8_VEC=1
#
# The caller owns the tree: fetch and reset, rebuild BOTH products (the .so and
# the binary, since the .so's build id is embedded in the binary), then run arms.
set -uo pipefail

TAG="${1:?usage: dsv41_serve_ab.sh <tag> [VAR=VALUE ...]}"
shift || true
PORT="${DSV41_PORT:-8090}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/tmp/ab_${TAG}.log"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"

# Pre-flight same-source gate. This driver does NOT build (the caller does), but
# it must not launch a serve onto a mismatched pair: .so and .build_id are both
# gitignored/untracked, so a `git checkout` + `git clean` can leave a STALE .so
# while the caller only rebuilt the binary. The id gate in cuda.rs/devrt.rs would
# still abort at dlopen, but that costs a full serve + log read to discover.
# Fail here, with the reason, before spawning anything.
SO="$ROOT/kernels/cuda/libferrite_kernels.so"
for f in "$SO" "$ROOT/target/release/dsv41-run"; do
    [ -e "$f" ] || { echo "FATAL: missing $f - run: PHASES=0 scripts/dsv41_recovery_verify.sh"; exit 1; }
done
[ -f "$ROOT/kernels/cuda/.build_id" ] || {
    echo "FATAL: kernels/cuda/.build_id missing (git clean? build.rs would embed +cuNOSTAMP)"
    echo "       rebuild the .so FIRST: kernels/cuda/build.sh ${ARCH:-103a}, then cargo build --release"
    exit 1
}
if command -v strings >/dev/null 2>&1 && \
   ! strings "$ROOT/target/release/dsv41-run" | grep -qF "$(cat "$ROOT/kernels/cuda/.build_id")"; then
    echo "FATAL: dsv41-run does not embed the current .build_id - stale/mismatched pair."
    echo "       rebuild BOTH in order: kernels/cuda/build.sh 103a, then cargo build --release"
    exit 1
fi

# Exact-PID cleanup only: pkill -f would match the caller's own command line.
for p in $(pgrep -x dsv41-run); do kill -9 "$p"; done
sleep 8

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    DSV41_MODEL_DIR="$MODEL_DIR" \
    DSV41_KERNELS="$ROOT/kernels/cuda/libferrite_kernels.so" \
    DSV41_TIMING=1 "$@" \
    "$ROOT/target/release/dsv41-run" --serve --port "$PORT" --tp 8 >"$LOG" 2>&1 &

for _ in $(seq 1 40); do
    sleep 5
    grep -q "serving" "$LOG" 2>/dev/null && break
done

echo "### $TAG   extra-env: ${*:-<none>}"
OUT="/tmp/ab_${TAG}_out.txt"
: >"$OUT"
for P in "The capital of France is" "请背诵《静夜思》" "1+1=" "请背诵《出师表》开头"; do
    printf -- "  [%s] " "$P"
    timeout 300 curl -s -m 260 -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":48,\"temperature\":0}" \
        | tee -a "$OUT" \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); print(repr(d["choices"][0]["message"]["content"][:40]))' 2>/dev/null \
        || echo "(failed)"
done
echo "  full outputs: $OUT"

python3 - "$LOG" "$TAG" <<'PY'
import re
import sys

log, tag = sys.argv[1], sys.argv[2]
xs = sorted(
    float(m.group(1))
    for line in open(log)
    for m in [re.search(r"\[dsv41\] step pos=\d+: ([\d.]+)ms", line)]
    if m
)
if xs:
    n = len(xs)
    print(
        f"  {tag:6s} steps={n:4d} p10={xs[n // 10]:6.2f} p50={xs[n // 2]:6.2f} "
        f"p90={xs[9 * n // 10]:6.2f} -> {1000 / xs[n // 2]:5.1f} tok/s"
    )
else:
    print("  no steps")
PY

echo "  faults: $(grep -cE 'illegal|fault' "$LOG")   log: $LOG"
for p in $(pgrep -x dsv41-run); do kill -9 "$p"; done
sleep 8
