#!/usr/bin/env bash
# Compile-check the DeepSeek-V4.1-Flash CUDA kernels WITHOUT running anything.
#
# Compiling needs nvcc but no GPU, so this is safe on a busy B300 node and is
# the agreed verification level for this work ("先别上 b300 实验" = no serving /
# no GPU inference; a compile check is not an experiment).
#
# Usage (local):
#   bash scripts/dsv41_compile_check.sh
# It rsync-free copies the sources to the remote, runs nvcc, and prints the
# per-kernel register usage.
set -euo pipefail

REMOTE="${REMOTE:-ubuntu@43.202.208.136}"
ARCH="${ARCH:-103a}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
FILES=($(cd "$DIR" && ls kernels/cuda/dsv41_*.cu kernels/cuda/tests_*.cu 2>/dev/null))

echo "== local syntax sanity (headers/ABI in the crate) =="
(cd "$DIR" && cargo test -p ferrite-dsv41 --quiet 2>&1 | tail -3)

echo "== copy sources to ${REMOTE}:/tmp/dsv41_cc =="
ssh -o BatchMode=yes "$REMOTE" 'mkdir -p /tmp/dsv41_cc'
found=0
for f in "${FILES[@]}"; do
    if [ -f "$DIR/$f" ]; then
        scp -q -o BatchMode=yes "$DIR/$f" "$REMOTE:/tmp/dsv41_cc/"
        found=1
    else
        echo "   (missing, skipped: $f)"
    fi
done
if [ "$found" = "0" ]; then
    echo "no kernel sources yet — nothing to compile"
    exit 0
fi

echo "== nvcc -gencode arch=compute_${ARCH},code=sm_${ARCH} (compile only) =="
ssh -o BatchMode=yes "$REMOTE" "cd /tmp/dsv41_cc && for f in *.cu; do
  echo \"--- \$f ---\"
  nvcc -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 -std=c++17 -Xptxas -v \\
       -c \"\$f\" -o \"\${f%.cu}.o\" 2>&1 | grep -E 'error|warning|Used [0-9]+ registers|registers' | head -30 || true
done"
echo "== self-test programs: compile only (run them yourself with ./t_*) =="
ssh -o BatchMode=yes "$REMOTE" "cd /tmp/dsv41_cc && for f in tests_*.cu; do
  [ -f \"\$f\" ] || continue
  printf '%-26s ' \"\$f\"
  nvcc -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 -std=c++17 -o /tmp/dsv41_cc/\${f%.cu}.bin \"\$f\" 2>&1 | grep -cE 'error'
done"
echo "== done (compile only; no GPU work was started) =="
