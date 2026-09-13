#!/bin/bash
# §141 follow-up: localise the SECOND BS-arm defect (the segment tables are now correct,
# but the decoded text is still garbage).
#
#   1) GENK: run the OFFICIAL TileLang-generated kernel through OUR shim and host wiring
#            (DSV41_MOE_BS_HANDWRITTEN=0). Correct text => the host wiring (tables, packed-SF
#            pool, TMA descriptors, gathered activation, C layout) is fine and the defect is
#            inside the hand-written kernel; garbage => the shared host side is still wrong.
#   2) NC2 : the host-double numeric probe on real data. With the tables fixed this probe has
#            a real chance of completing: ALL MATCH => the MMA's own output is correct and the
#            defect is downstream of g_c (scatter / swiglu / down); MISMATCH => the MMA's
#            staged inputs are wrong.
set -uo pipefail
cd "$HOME/ferrite"

run () {  # run <name> [env...]
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WARN: completions|WATCHDOG" | head -5
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

run GENK DSV41_MOE_BS_HANDWRITTEN=0
run NC2 DSV41_MOE_BS_NUMCHECK=1

echo "=== NC verdict ==="
grep -a "NC\] WORST" "$HOME/armrun_NC2.log" | head -3
grep -a "NC\] seg=" "$HOME/armrun_NC2.log" | head -8
echo "=== MMA-DIAG g_c ==="
grep -a "MMA-DIAG" "$HOME/armrun_GENK.log" "$HOME/armrun_NC2.log" | head -4
echo "=== GENK armed notices ==="
grep -aE "moe-bs\] ARMED|DECLINE|moe-bs-diag" "$HOME/armrun_GENK.log" | head -4
