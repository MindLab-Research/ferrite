#!/bin/bash
# Batch 5: collect the STAGED-operand semantic dump (DSV41_MOE_BS_SFDUMP) plus the gate|up dump,
# for the block-scaled arm and for the proven per-slot path (the latter as the control: its staged
# data must be the ground truth the checker compares against).
#
# One arm per path, no rebuild needed if the tree is already built (it verifies the mtime and only
# rebuilds when the binary is older than HEAD).
set -uo pipefail
cd "$HOME/ferrite"

echo "=== tree state ==="; git log --oneline -1
stat -c "%y %n" kernels/cuda/libferrite_kernels.so target/release/ferrite-serve

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "bs-sfdump\|gateup-dump.*done" "$HOME/armrun_${name}.log" | head -3
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

rm -rf /tmp/sfd_bs /tmp/sfd_old
run SFD_BS  DSV41_MOE_BS_SFDUMP=/tmp/sfd_bs  DSV41_GATEUP_DUMP=/tmp/gu_in_SFD_BS
run SFD_OLD DSV41_MOE_BS_SFDUMP=/tmp/sfd_old DSV41_GATEUP_DUMP=/tmp/gu_in_SFD_OLD \
            DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0

echo "=== dumps ==="; ls -la /tmp/sfd_bs /tmp/sfd_old 2>/dev/null
echo "=== staged-dump checker (writes its own verdict) ==="
if [ -f "$HOME/sfdump_check.py" ]; then
  for d in /tmp/sfd_bs /tmp/sfd_old; do
    echo "--- $d"; timeout 600 python3 "$HOME/sfdump_check.py" --in-dir "$d" \
        --gateup-dir /tmp/gu_in_$(basename $d | sed 's/sfd_/SFD_/') 2>&1 | tail -25
  done
else
  echo "(sfdump_check.py not present yet — dump collected, checker pending)"
fi
