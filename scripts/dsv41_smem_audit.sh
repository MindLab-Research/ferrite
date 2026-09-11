#!/usr/bin/env bash
# Cross-check dynamic shared memory: every kernel that declares
# `extern __shared__` needs its launcher's THIRD launch argument to carry the
# size. Getting that wrong is a silent memory-corruption bug - the staging
# pointer lands at the base of a zero-byte dynamic allocation and the writes
# walk off it. It cost this project two rounds (gemv_bf16, then gemv_f32) and
# the symptom looked like a model regression (4 GPU faults, empty outputs).
#
# Run before committing any kernel that gained a staging buffer.
set -euo pipefail
cd "$(dirname "$0")/../kernels/cuda"
echo "== kernels declaring dynamic shared memory =="
grep -Hn "extern __shared__" dsv41_*.cu
echo
echo "== launch sites with a THIRD argument that is not 0 =="
grep -HnE "<<<" dsv41_*.cu | grep -E "<<<[^>]*, *(\(size_t\)|sizeof|.*\* *sizeof)" | head -40
echo
echo "manual step: for each kernel above, confirm a launch site exists whose third"
echo "argument is the staging size (not 0)."
