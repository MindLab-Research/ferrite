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
for f in "$DIR"/dsv41_kernels.cu "$DIR"/dsv41_experts_mxf4.cu "$DIR"/dsv41_vision.cu; do
    [ -f "$f" ] && SRCS+=("$f")
done

NVCC="${NVCC:-nvcc}"
"$NVCC" --version >/dev/null 2>&1 || { echo "error: nvcc not found (CUDA toolkit required)"; exit 1; }

# -O3 + fPIC shared object; --use_fast_math disabled (CPU-golden parity first,
# numerics tuning comes after the B300 correctness harness passes).
"$NVCC" -O3 -shared -Xcompiler -fPIC \
    -std=c++17 \
    -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
    -o "$OUT" "${SRCS[@]}"

echo "built ${OUT} for sm_${ARCH} from ${SRCS[*]}"
