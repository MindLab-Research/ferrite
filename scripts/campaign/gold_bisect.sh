#!/bin/bash
# gold_bisect.sh <commit> [--restore] — bisect the shared-forward regression with the GOLDEN arbiter.
#
# WHY THIS IS THE RIGHT ARBITER (and why it is cheap):
#   /tmp/gu_in_GD4_OLD/{x.f32,ids.i32} are BYTE-IDENTICAL to today's dump inputs (`same x: True,
#   max|d|=0`, same ids), and /tmp/gu_in_GD4_OLD/gateup.f32 matches the self-tested official oracle
#   bit-exactly (max|d|=0, corr=1.0). So that file IS the official answer for inputs we reproduce on
#   every run. One minimal-eager run + one numpy diff therefore gives a binary verdict per commit --
#   no counting prompt, no gap analysis, no ambiguity.
#
# The checkout touches kernels/crates only (like bisect_probe.sh); --restore returns to origin/main.
set -uo pipefail
REV=${1:?usage: gold_bisect.sh <commit> [--restore]}
RESTORE=${2:-}
cd "$HOME/ferrite" || exit 1
echo "=== gold_bisect: kernels+crates at $REV ==="
git checkout -q "$REV" -- kernels crates 2>&1 | head -3
if ! (cd kernels/cuda && bash build.sh 103a 2>&1 | tail -1); then echo "BUILD_SH FAILED"; exit 1; fi
if ! (source "$HOME/.cargo/env" && cargo build --release 2>&1 | tail -1); then echo "CARGO FAILED"; exit 1; fi
echo "=== one minimal-eager run, dumping the first MoE call ==="
rm -rf /tmp/goldchk
bash "$HOME/num100.sh" GOLDB_$REV \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 DSV41_SPEC=0 DSV41_DSPARK=0 \
  DSV41_EXPERT_ACT_E4M3=0 DSV41_BF16_TRUNCATE=0 DSV41_EXPERT_ILV=0 \
  DSV41_GRAPH_STEP=0 DSV41_VERIFY_GRAPH=0 DSV41_GATEUP_DUMP=/tmp/goldchk >/dev/null 2>&1
python3 - "$REV" <<'PY'
import os, sys, numpy as np
def rd(p, dt='<f4'):
    return np.fromfile(p, dtype=dt) if os.path.exists(p) else None
new = rd("/tmp/goldchk/eager/gateup.f32")
gold = rd("/tmp/gu_in_GD4_OLD/gateup.f32")
new_ids = rd("/tmp/goldchk/eager/ids.i32", '<i4'); gold_ids = rd("/tmp/gu_in_GD4_OLD/ids.i32", '<i4')
new_x = rd("/tmp/goldchk/eager/x.f32"); gold_x = rd("/tmp/gu_in_GD4_OLD/x.f32")
same_in = (new_x is not None and gold_x is not None and np.array_equal(new_x, gold_x)
           and new_ids is not None and gold_ids is not None and np.array_equal(new_ids, gold_ids))
print("  inputs identical to the golden run:", same_in)
if new is None:
    print("  VERDICT: no dump produced"); raise SystemExit
if not same_in:
    print("  WARNING: inputs differ -- this commit changed the INPUT, so the gate|up diff is confounded")
den = np.maximum(np.abs(gold), 1e-6); rel = np.abs(new - gold) / den
exact = np.array_equal(new, gold)
print("  %s vs GOLDEN: max|d|=%.6g med_rel=%.4g frac>5%%=%.4f corr=%+.5f exact=%s"
      % (sys.argv[1], np.abs(new-gold).max(), np.median(rel), float((rel > 0.05).mean()),
         np.corrcoef(np.nan_to_num(new), gold)[0, 1], exact))
print("  VERDICT:", "MATCHES the golden (= official) at this commit" if exact
      else "DIVERGES at this commit (regression present)")
PY
if [ "$RESTORE" = "--restore" ]; then
  echo "=== restoring origin/main ==="
  git checkout -q origin/main -- kernels crates
  (cd kernels/cuda && bash build.sh 103a 2>&1 | tail -1)
  (source "$HOME/.cargo/env" && cargo build --release 2>&1 | tail -1)
  echo RESTORED
fi
