#!/usr/bin/env bash
# Build the ferrite CUDA kernels. Compiling needs nvcc but NO GPU (compile
# only) — safe to run on a busy B300 node.
#
# Usage: ./build.sh [sm_arch]     default: 100a (B300 Blackwell Ultra)
#        FERRITE_OUT=libferrite_kernels.so ./build.sh
set -euo pipefail

ARCH="${1:-100a}"
OUT="${FERRITE_OUT:-libferrite_kernels.so}"
SRC="$(dirname "$0")/ferrite_kernels.cu"

NVCC="${NVCC:-nvcc}"
"$NVCC" --version >/dev/null 2>&1 || { echo "error: nvcc not found (CUDA toolkit required)"; exit 1; }

# -O3 + fPIC shared object. --use_fast_math is OPT-IN via FERRITE_FAST_MATH=1
# (2026-09-10: enforced-by-default measured a ~2x replay regression —
# 27.99ms vs 13.40ms at full 2032 MHz clock, i.e. not thermal/power).
# --use_fast_math: DEFAULT ON (matching the working config). Set
# FERRITE_NO_FAST_MATH=1 to disable for A/B. WARNING (measured 2026-09-10):
# the .so built WITHOUT it CRASHES the batched capture (faults=2, err 900) —
# fast-math shifts kernel durations and thereby the in-capture pool size-class
# requests (the known pool-size sensitivity). Keep it ON unless investigating.
FAST_MATH_FLAG="--use_fast_math"
if [ -n "${FERRITE_NO_FAST_MATH:-}" ]; then FAST_MATH_FLAG=""; fi
"$NVCC" -O3 -shared -Xcompiler -fPIC $FAST_MATH_FLAG \
    -std=c++17 \
    -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
    -o "$OUT" "$SRC"

echo "built ${OUT} for sm_${ARCH} from ${SRC}"
