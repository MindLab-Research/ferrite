#!/bin/bash
# gap_probe.sh [TAG] — the probe for the counting-gap defect, which bisect_probe.sh cannot see
# (that one runs stage_probe.sh, i.e. the hang probe).
#
# Two facts define this bug and both must be measured:
#   * are there gaps at all (1..100 counting -> the byte-exact reply's integer list), and
#   * are the gaps the SAME on two consecutive runs (a race would differ).
# The minimal eager configuration is deliberate: SPEC/DSPARK/E4M3/BF16/ILV and both graphs off, BS arm
# off, so nothing but the base model forward is under test.
set -uo pipefail
TAG=${1:-main}
cd "$HOME/ferrite"
echo "=== gap_probe on $(git log --oneline -1 --no-decorate | head -1) (tag=$TAG) ==="
report () {
  python3 - "$1" <<'PY'
import re, sys
t = open("/home/ubuntu/num100_last.txt", errors="ignore").read()
nums = [int(x) for x in re.findall(r"\d+", t)]
gaps = [(a, b) for a, b in zip(nums, nums[1:]) if b - a > 1]
print("  run %s: numbers=%d  gaps=%s  all1..100=%s"
      % (sys.argv[1], len(nums), gaps[:8], set(range(1, 101)) <= set(nums)))
print("  GAPSIG %s" % gaps)
PY
}
for r in 1 2; do
  bash "$HOME/num100.sh" GAP_${TAG}_$r \
    DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 DSV41_SPEC=0 DSV41_DSPARK=0 \
    DSV41_EXPERT_ACT_E4M3=0 DSV41_BF16_TRUNCATE=0 DSV41_EXPERT_ILV=0 \
    DSV41_GRAPH_STEP=0 DSV41_VERIFY_GRAPH=0 >/dev/null 2>&1
  report "$r"
done
echo "=== interpretation: identical GAPSIG on both runs => deterministic (a code path); different => a race ==="
