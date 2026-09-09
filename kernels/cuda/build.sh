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

# -O3 + fPIC shared object; --use_fast_math ENABLED 2026-09-10 (the B300
# correctness harness passed long ago — the text output has been verified
# 出师表-verbatim through dozens of runs). The main win is -ftz=true (flush
# denormals to zero) — the softmax exp(-large) values in sparse_attn/kpool
# produce denormals that cost ~10x normal float ops on some paths.
"$NVCC" -O3 -shared -Xcompiler -fPIC --use_fast_math \
    -std=c++17 \
    -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
    -o "$OUT" "$SRC"

echo "built ${OUT} for sm_${ARCH} from ${SRC}"
