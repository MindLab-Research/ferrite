#!/bin/bash
# arm_summary.sh <arm-name>...  — unified one-glance summary of finished arms.
# Prints, per arm: the switch receipt, the generated text, the error count, the [NC]
# numeric verdict (if present), and the step line — so a round can be judged without
# re-grepping logs and without tail(1) truncating the decisive line (§25).
for a in "$@"; do
  LOG=$HOME/armrun_$a.log
  echo "=================== $a ==================="
  if [ ! -f "$LOG" ]; then echo "  (no log)"; continue; fi
  printf '  switches: '; grep -oE "packed fp4 staging = [0-9]+|pack geometry = [0-9]+|swapAB = [0-9]+|canon layout = [0-9]+|scale_vec::1X = [0-9]+|sf byte order reversed = [0-9]+|\[NC\] entered" "$LOG" | sort -u | tr '\n' ' '; echo
  printf '  OUT     : '; grep -m1 "OUT:" "$LOG" | cut -c1-160; echo
  printf '  errors  : '; grep -cE "error|ERR_COUNT: [1-9]" "$LOG"
  printf '  [NC]    : '; (grep -m1 "\[NC\] entered" "$LOG" || echo -n "not-entered ") ; grep -m1 "\[NC\] WORST" "$LOG" | sed 's/.*\[NC\] //' || true; echo
  printf '  step    : '; grep -m1 "\[dsv41\] step pos" "$LOG" | sed 's/.*step pos/step pos/'; echo
  printf '  decline : '; grep -c "DECLINE during graph capture" "$LOG"
done
echo "==========================================="
echo "judge order: [NC] WORST rel (numeric) > text > step time (graphs are OFF in these arms,"
echo "             so the step time is roughly 10x slow and NOT a performance number)."
