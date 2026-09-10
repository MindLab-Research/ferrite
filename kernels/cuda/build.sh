#!/usr/bin/env bash
# Build the ferrite CUDA kernels. Compiling needs nvcc but NO GPU (compile
# only) — safe to run on a busy B300 node.
#
# Usage: ./build.sh [sm_arch]     default: 100a (B300 Blackwell Ultra)
#        FERRITE_OUT=libferrite_kernels.so ./build.sh
set -euo pipefail

ARCH="${1:-100a}"
OUT="${FERRITE_OUT:-libferrite_kernels.so}"
DIR="$(dirname "$0")"
# DeepSeek-V4.1-Flash kernels live in their own translation units; they are
# linked into the same .so so the engine keeps a single dlopen target.
SRCS=("$DIR/ferrite_kernels.cu")
# the tcgen05 MXFP4 expert GEMM is its own TU; tests_*.cu carry a main
# and are deliberately NOT linked into the shared object.
for f in "$DIR"/dsv41_kernels.cu "$DIR"/dsv41_experts_mxf4.cu "$DIR"/dsv41_vision.cu "$DIR"/dsv41_glue.cu "$DIR"/dsv41_route.cu; do
    [ -f "$f" ] && SRCS+=("$f")
done

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
# Build stamp: the Rust side refuses to load a .so built from another
# revision (user rule: 严禁组合不同版本). Use the git revision of THIS tree.
BUILD_ID="$(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$(dirname "$0")" status --porcelain 2>/dev/null)" ]; then
  BUILD_ID="${BUILD_ID}-dirty"
fi
# Same-source enforcement: fold the .cu content hash in. The Rust side embeds
# whatever this script last wrote to .build_id, so rebuilding only ONE of the
# two artifacts produces a mismatch and the process REFUSES TO START.
CU_HASH="$(sha256sum "${SRCS[@]}" | sha256sum | cut -c1-16)"
BUILD_ID="${BUILD_ID}+cu${CU_HASH}"
echo "$BUILD_ID" > "$(dirname "$0")/.build_id"

"$NVCC" -O3 -shared -Xcompiler -fPIC $FAST_MATH_FLAG \
    -std=c++17 \
    -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
    -DFERRITE_KERNEL_BUILD_ID="\"${BUILD_ID}\"" \
    -o "$OUT" "${SRCS[@]}"

echo "built ${OUT} for sm_${ARCH} from ${SRCS[*]} (build_id ${BUILD_ID})"
