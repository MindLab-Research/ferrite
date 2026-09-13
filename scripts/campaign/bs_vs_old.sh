#!/bin/bash
# bs_vs_old.sh [extra env...] — the decisive differential test for the BS arm.
#   arm A: the BS (tcgen05 fp4 MoE) arm ON
#   arm B: the SAME configuration with the BS gates removed -> the old per-slot GEMV path,
#          whose output is known correct (this whole campaign's ground truth)
# Then wq_check compares A against B: identical text (or identical degradation) means the BS
# arm reproduces the working path; a divergence localises the defect to the BS path.
# This exists because the [NC] numeric readback is structurally blocked during graph
# recording (§73), so the differential is the strongest available instrument.
set -euo pipefail
cd ~/ferrite || exit 1
EXTRA="$*"
echo "########## arm A: BS ON (tcgen05 fp4 MoE, correct layout by default) ##########"
bash ~/arm_run.sh DIFF_bs_on DSV41_MOE_BS_SWAPAB=1 $EXTRA 2>&1 | tee ~/armrun_DIFF_bs_on.txt | grep -E "OUT:|ERR_COUNT|DONE|NC\]"
echo "########## arm B: BS OFF (old per-slot GEMV path = ground truth) ##########"
# strip the two BS gates so the MoE falls back to the known-correct path
bash ~/arm_run_fast.sh DIFF_old $(echo "$EXTRA" | tr ' ' '\n' | grep -v 'MOE_TILELANG_BS\|MOE_BS_' | tr '\n' ' ') 2>&1 \
  | tee ~/armrun_DIFF_old.txt | grep -E "OUT:|ERR_COUNT|DONE"
echo "########## differential verdict (identical = BS arm reproduces the working path) ##########"
python3 ~/wq_check.py --log ~/armrun_DIFF_bs_on.txt --eager-file <(sed -n "s/^\[DIFF_old\] OUT: '\(.*\)'$/\1/p" ~/armrun_DIFF_old.txt | head -1)
echo "--- side by side ---"
printf 'BS ON : '; sed -n "s/^\[DIFF_bs_on\] OUT: '\(.*\)'$/\1/p" ~/armrun_DIFF_bs_on.txt | head -1 | cut -c1-160; echo
printf 'OLD   : '; sed -n "s/^\[DIFF_old\] OUT: '\(.*\)'$/\1/p" ~/armrun_DIFF_old.txt | head -1 | cut -c1-160; echo
echo "########## bs_vs_old DONE ##########"
