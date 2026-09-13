#!/bin/bash
# §141 follow-up batch 3: three cheap liveness/error probes on the hand-written BS arm.
#
#   SY : DSV41_MOE_BS_SYNC_DIAG=1  — per-kernel sync verdicts (gather / HANDWRITTEN MMA /
#        scatter "done: OK?"). A CUDA error inside the MMA (bad descriptor / OOB) shows up here
#        instead of being swallowed. Also proves the MMA actually COMPLETES.
#   ZA : DSV41_MOE_BS_ZERO_A=1     — zero the gathered activations AFTER the gather. If the arm's
#        output is really consumed, gate|up must collapse and the text must change materially;
#        if the text is byte-identical to the baseline arm, the arm's output is being ignored
#        (a wiring defect, not a kernel defect).
#   ZE : DSV41_MOE_BS_ZERO_EID=1   — force every segment's Eid to 0 (all experts read expert 0).
#        Same signal as ZA but for the segment tables.
set -uo pipefail
cd "$HOME/ferrite"

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WARN: completions|WATCHDOG" | head -4
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

run SY DSV41_MOE_BS_SYNC_DIAG=1
run ZA DSV41_MOE_BS_ZERO_A=1
run ZE DSV41_MOE_BS_ZERO_EID=1

echo "=== SYNC-DIAG verdicts (SY) ==="
grep -a "SYNC-DIAG" "$HOME/armrun_SY.log" | head -8
echo "=== baseline for comparison (FIXA) ==="
grep -a "OUT:" "$HOME/armrun_FIXA.txt" | head -1
