#!/bin/bash
# endgame.sh — run the moment the BS arm is confirmed correct.
#   (1) full-gate regression + the 400 push (push400_hw_test.sh, the shipping env set)
#   (2) mechanised red-line verdicts on whatever logs that run produced
#   (3) a compact summary: real p50 step time + the acceptance verdicts
# Reads nothing destructive; the only heavy step is push400_hw_test.sh itself.
set -euo pipefail
cd ~/ferrite || exit 1
echo "########## (1) full-gate regression + 400 push (shipping env configuration) ##########"
if [ -x ~/push400_hw_test.sh ]; then
  # tee the whole regression to a file: push400_hw_test.sh prints the generated text to
  # stdout (no tee of its own), and wq_check.py can only judge what it can read.
  bash ~/push400_hw_test.sh 2>&1 | tee ~/armrun_ENDGAME_regression.txt | tail -45
else
  echo "(~/push400_hw_test.sh missing)"; exit 1
fi
echo "########## (2) mechanised red-line verdicts ##########"
# every arm log this run touched, newest first, capped to the recent few
mapfile -t LOGS < <(ls -t ~/armrun_*.log ~/armrun_ENDGAME_regression.txt 2>/dev/null | head -8)
if [ "${#LOGS[@]}" -gt 0 ]; then
  python3 ~/wq_check.py --log "${LOGS[@]}" || true
else
  echo "(no armrun_*.log found — check how push400_hw_test.sh names its logs)"
fi
echo "########## (3) step-time summary (REAL p50 only; never from a graphs-off diagnostic arm) ##########"
for L in "${LOGS[@]}"; do
  printf '%-28s ' "$(basename "$L")"
  grep -oE "\[dsv41\] step pos=[0-9]+: [0-9.]+ms \([0-9.]+ tok/s\)" "$L" | tail -1 || echo "(no step line)"
done
echo "########## endgame DONE — promote precision gates via ~/promote_precision.sh, one at a time ##########"
