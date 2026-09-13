#!/bin/bash
# Batch 10 (fix confirmation): if the mbarrier fix is right, the SAME measurements that exposed the
# race must now come out clean —
#   1. determinism: two runs must be bit-identical;
#   2. ZERO_A: with the A operand actually consumed, zeroing g_a must give an output of EXACTLY 0;
#   3. correctness: the arm vs the official-semantics oracle (full K);
#   4. the text red line (the counting prompt must come back).
#
# Usage: batch10_fixconfirm.sh [MBAR_RING|MBAR_PERSTAGE|MBAR_RING+MBAR_PERSTAGE] [label]
set -uo pipefail
cd "$HOME/ferrite"
MODE="${1:-MBAR_RING}"
LABEL="${2:-${MODE//+/_}}"
GATE=""
for m in ${MODE//+/ }; do GATE="$GATE DSV41_MOE_BS_$m=1"; done
echo "=== mode: $MODE  (gate:$GATE) label: $LABEL ==="

echo "=== rebuild BOTH artefacts + freshness gate ==="
(cd kernels/cuda && set -o pipefail; bash build.sh 103a 2>&1 | tail -2; echo KERNEL_RC=${PIPESTATUS[0]})
source "$HOME/.cargo/env"
(set -o pipefail; cargo build --release 2>&1 | tail -2; echo CARGO_RC=${PIPESTATUS[0]})
bash "$HOME/check_artifacts.sh" || { echo "ARTIFACTS_STALE — aborting"; exit 1; }
git log --oneline -1

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "mbarrier ring" "$HOME/armrun_${name}.log" | head -1
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

rm -rf /tmp/rn_a /tmp/rn_b /tmp/rn_za
run RN_A  DSV41_GATEUP_DUMP=/tmp/rn_a  $GATE
run RN_B  DSV41_GATEUP_DUMP=/tmp/rn_b  $GATE
run RN_ZA DSV41_GATEUP_DUMP=/tmp/rn_za $GATE DSV41_MOE_BS_ZERO_A=1

echo "=== 1. determinism (must be bit-identical) ==="
cmp -s /tmp/rn_a/eager/gateup.f32 /tmp/rn_b/eager/gateup.f32 \
  && echo "DETERMINISTIC ✓ (the race is gone)" || echo "STILL NON-DETERMINISTIC ✗"

echo "=== 2. ZERO_A (must be exactly 0) ==="
python3 - <<'PY'
import numpy as np, os
p = '/tmp/rn_za/eager/gateup.f32'
if not os.path.exists(p):
    print('no ZERO_A dump'); raise SystemExit
a = np.fromfile(p, dtype='<f4')
print(f"ZERO_A under the ring: max|.|={np.abs(a).max():.6g} nonzero={int((a!=0).sum())} "
      f"-> " + ("A OPERAND IS NOW CONSUMED AS STAGED ✓" if float(np.abs(a).max()) == 0.0
                else "still not consuming the staged A ✗"))
PY

echo "=== 3. correctness vs the official-semantics oracle ==="
for d in /tmp/rn_a; do
  e=$d; [ -d "$d/eager" ] && e=$d/eager
  python3 "$HOME/gdu_cmp.py" "$e/gateup.f32" /tmp/gu_bs_new.f32 6 640 10.0 2>&1 | head -6
done

echo "=== 4. text red line (the counting prompt) ==="
python3 "$HOME/wq_check.py" --log "$HOME/armrun_RN_A.txt" 2>&1 | tail -6 || true
grep -a "OUT:" "$HOME/armrun_RN_A.txt" | head -2
