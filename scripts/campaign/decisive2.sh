#!/bin/bash
# decisive2.sh — the second decisive window: rebuild (to pick up the deferred [NC] probe and
# the D1 visibility fix), then run the differential test (BS arm vs the known-correct old
# per-slot GEMV path) plus the numeric probe.
set -euo pipefail
cd ~/ferrite || exit 1
echo "########## back-to-back rebuild (no writers during this) ##########"
cd kernels/cuda && bash build.sh 103a 2>&1 | tail -1 && cd ~/ferrite && source ~/.cargo/env \
  && cargo build --release 2>&1 | tail -1 && echo BUILD_PAIR_OK || { echo "BUILD FAILED"; exit 1; }
echo "########## differential: BS arm ON vs the known-correct old path ##########"
bash ~/bs_vs_old.sh 2>&1 | tail -30
echo "########## [NC] numeric probe (deferred to a non-captured call) ##########"
for L in ~/armrun_DIFF_bs_on.log ~/armrun_T1.log; do
  [ -f "$L" ] || continue
  echo "-- $(basename $L)"
  grep -E "\[NC\]|^\[NC\]" "$L" | head -12
done
echo "########## decisive2 DONE ##########"
