#!/bin/bash
# Gate|up A/B numeric comparator: the proven per-slot path vs the block-scaled arm.
#   * Both runs dump `DSV41_GATEUP_DUMP` at the FIRST `moe()` call (= layer 0), so the two
#     arrays see byte-identical inputs and the diff pattern names the defect:
#     one column half only => B-side row mapping; uniform ratio => scale; permutation =>
#     smem/descriptor layout; all zeros => staging; NaN => bad descriptor.
#   * The OLD arm disables ONLY the block-scaled dispatch (DSV41_MOE_TILELANG_BS=0 +
#     DSV41_MOE_BS_HANDWRITTEN=0). DSV41_EXPERT_ACT_E4M3=1 stays on in both arms, so both
#     consume the same e4m3 activation bytes and the same fp4 weight pool.
set -uo pipefail
cd "$HOME/ferrite"

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "gateup-dump" "$HOME/armrun_${name}.log" | head -2
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

rm -f /tmp/gu_bs.f32 /tmp/gu_old.f32
run GD_BS  DSV41_GATEUP_DUMP=/tmp/gu_bs.f32
run GD_OLD DSV41_GATEUP_DUMP=/tmp/gu_old.f32 DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0

echo "=== dumps ==="
ls -la /tmp/gu_*.f32 2>/dev/null
echo "=== offline comparison (old path = ground truth) ==="
python3 "$HOME/gdu_cmp.py" /tmp/gu_bs.f32 /tmp/gu_old.f32 6 640 || true
