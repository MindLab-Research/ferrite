#!/usr/bin/env bash
# Isolated kernel timing for the DSV4.1 decode hot spots, against the PRODUCTION
# .so at production shapes. This is the fast loop for kernel work: it answers
# "did the change help?" in ~3 seconds instead of two ~1.5-minute serve A/Bs.
#
#   scripts/dsv41_microbench.sh [gemv|indexer|all]
#
# Notes learned the hard way:
#   * nvcc does NOT accept -Wl,-rpath here; run with LD_LIBRARY_PATH instead.
#   * the benches need <cstdint> (uint8_t) and -std=c++17.
#   * the .so is picked up at run time, so only rebuild the bench when its own
#     source changes - a fresh `bash build.sh 103a` is enough for the kernels.
set -euo pipefail
WHAT="${1:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
K="$ROOT/kernels/cuda"
export LD_LIBRARY_PATH="$K${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if pgrep -x dsv41-run >/dev/null 2>&1; then
  echo "refusing to start: a dsv41-run is alive (the GPU must be free)" >&2
  exit 1
fi
bench() { # $1 = source basename
  local src="$ROOT/scripts/$1.cu" bin="/tmp/$1"
  nvcc -O2 -std=c++17 -o "$bin" "$src" -L"$K" -lferrite_kernels
  "$bin"
}
case "$WHAT" in
  gemv)    bench dsv41_gemv_bench ;;
  indexer) bench dsv41_indexer_bench ;;
  all)     bench dsv41_gemv_bench; bench dsv41_indexer_bench ;;
  *) echo "usage: $0 [gemv|indexer|all]" >&2; exit 2 ;;
esac
