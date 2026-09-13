#!/bin/bash
# gap_bisect.sh <commit> [--restore] — deterministic bisection for the counting-gap defect.
#
# WHY THIS AND NOT bisect_probe.sh: that one runs stage_probe.sh (the HANG probe) and so cannot see
# gaps. This runs one minimal-eager counting run and prints GAPSIG. The defect was just shown to be
# DETERMINISTIC (two identical runs give byte-identical gap signatures), so ONE run per commit is a
# sound verdict — which makes a 116-commit range bisectable in ~7 rebuilds.
#
# The checkout touches kernels/crates only, exactly like bisect_probe.sh, and --restore returns the
# tree to origin/main and rebuilds it, so history is never rewritten (the AGENTS rule forbids revert,
# not a bisection experiment on a test rig).
set -uo pipefail
REV=${1:?usage: gap_bisect.sh <commit> [--restore]}
RESTORE=${2:-}
cd "$HOME/ferrite" || exit 1
echo "=== gap_bisect: checking kernels+crates out at $REV ==="
git checkout -q "$REV" -- kernels crates 2>&1 | head -3
echo "=== rebuild pair (pipefail guarded) ==="
if ! (cd kernels/cuda && bash build.sh 103a 2>&1 | tail -1); then echo "BUILD_SH FAILED"; exit 1; fi
if ! (source "$HOME/.cargo/env" && cargo build --release 2>&1 | tail -1); then echo "CARGO FAILED"; exit 1; fi
echo "=== one minimal-eager counting run (the verdict) ==="
bash "$HOME/num100.sh" GAPB_$REV \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 DSV41_SPEC=0 DSV41_DSPARK=0 \
  DSV41_EXPERT_ACT_E4M3=0 DSV41_BF16_TRUNCATE=0 DSV41_EXPERT_ILV=0 \
  DSV41_GRAPH_STEP=0 DSV41_VERIFY_GRAPH=0 >/dev/null 2>&1
python3 - "$REV" <<'PY'
import re, sys
t = open("/home/ubuntu/num100_last.txt", errors="ignore").read()
nums = [int(x) for x in re.findall(r"\d+", t)]
gaps = [(a, b) for a, b in zip(nums, nums[1:]) if b - a > 1]
clean = set(range(1, 101)) <= set(nums)
print("  %s: numbers=%d clean=%s GAPSIG %s" % (sys.argv[1], len(nums), clean, gaps))
print("  VERDICT:", "CLEAN at this commit (defect is NEWER)" if clean else "GAPPY at this commit (defect is OLDER or here)")
PY
if [ "$RESTORE" = "--restore" ]; then
  echo "=== restoring origin/main ==="
  git checkout -q origin/main -- kernels crates
  (cd kernels/cuda && bash build.sh 103a 2>&1 | tail -1)
  (source "$HOME/.cargo/env" && cargo build --release 2>&1 | tail -1)
  echo "RESTORED"
fi
