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
# If the caller did NOT (or rebuilt them out of order), the pre-flight gate below
# now SELF-HEALS: it rebuilds the pair in the one order that works instead of
# aborting. See the gate for why `cargo build` alone can never fix it.
set -uo pipefail

TAG="${1:?usage: dsv41_serve_ab.sh <tag> [VAR=VALUE ...]}"
shift || true
PORT="${DSV41_PORT:-8090}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/tmp/ab_${TAG}.log"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"

# Pre-flight same-source gate. This driver does not normally build (the caller
# does), but it must not launch a serve onto a mismatched pair: .so and .build_id
# are both gitignored/untracked, so a `git checkout` + `git clean` leaves them
# behind while the caller may only have rebuilt the binary. The id gate in
# cuda.rs/devrt.rs would still abort at dlopen, but that costs a full serve +
# log read to discover — so detect here and rebuild here, before spawning.
#
# WHY `cargo build` ALONE CANNOT HEAL IT (2026-09-11): cargo is incremental and
# the stamp is baked in by ferrite-kernel/build.rs (`rerun-if-changed` on
# .build_id + build.rs). If cargo does not observe a change it simply relinks an
# OLD FERRITE_BUILD_ID, so the mismatch is permanent no matter how many times the
# binary is rebuilt. The only working order is:
#     build.sh (rewrites .build_id) -> touch build.rs (force the rerun)
#     -> cargo build --release (bakes the fresh stamp into the binary)
#
# WHY THE CHECK USES `grep -cF` AND NOT `grep -qF`: this script runs under
# `set -o pipefail`. `strings BIN | grep -qF id` makes grep exit at its first
# match, so `strings` gets SIGPIPE (141) and pipefail reports the PIPELINE as
# failed EVEN THOUGH THE ID WAS FOUND — the gate then misfires "stale" on a
# perfectly good pair. `grep -c` reads the whole stream (no early exit, no
# SIGPIPE) and still exits 0 iff there is >=1 match.
SO="$ROOT/kernels/cuda/libferrite_kernels.so"
BIN="$ROOT/target/release/dsv41-run"
K="$ROOT/kernels/cuda"
ARCH="${ARCH:-103a}"   # B300 = sm_103a (build.sh's default 100a is stale)
for f in "$SO" "$BIN"; do
    [ -e "$f" ] || { echo "FATAL: missing $f - run: PHASES=0 scripts/dsv41_recovery_verify.sh"; exit 1; }
done

# true iff $BIN literally embeds the id the .so currently carries.
# No `strings` -> cannot verify -> assume OK (preserves the old behaviour).
embeds_id() {
    command -v strings >/dev/null 2>&1 || return 0
    [ -f "$K/.build_id" ] || return 1
    strings "$BIN" | grep -cF -- "$(cat "$K/.build_id")" >/dev/null
}

if ! embeds_id; then
    # Auto-rebuild if build_id mismatch (the .so timestamp may not have
    # changed even though content did, so cargo skips the rebuild).
    echo "WARN: build_id mismatch, auto-rebuilding (build.sh ${ARCH} -> cargo build)..."
    command -v nvcc >/dev/null 2>&1 || {
        echo "FATAL: nvcc not found - cannot rebuild the .so (toolkit needed, no GPU)"
        exit 1
    }
    ( cd "$K" && bash build.sh "$ARCH" ) || { echo "FATAL: kernels/cuda/build.sh $ARCH failed"; exit 1; }
    # build.sh always rewrites .build_id, but an identical-content rewrite can
    # land within the same mtime second - cargo would then skip build.rs and
    # relink the old stamp. Touching it forces the rerun unconditionally.
    touch "$ROOT/crates/ferrite-kernel/build.rs"
    ( cd "$ROOT" && cargo build --release ) || { echo "FATAL: cargo build --release failed"; exit 1; }
    # Re-check, never trust the rebuild blindly (post-build proof, not assumption).
    if ! embeds_id; then
        echo "FATAL: $BIN still does not embed the .build_id after a rebuild - stale/mismatched pair."
        echo "       .so/.build_id id: $(cat "$K/.build_id" 2>/dev/null || echo '<.build_id missing>')"
        echo "       rebuild manually: (cd kernels/cuda && bash build.sh $ARCH) && cargo build --release"
        exit 1
    fi
    echo "WARN: rebuild OK, pair is same-source again"
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
