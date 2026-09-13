#!/bin/bash
# STEP-TIME arm (the goal is a step-time problem: 450 tok/s at real acc 2.2 needs step ~7.2 ms).
#
# Deliberately NOT a correctness arm and NOT a baseline re-run: the correctness work was re-scoped
# (the BS arm is one optimisation among several, and the dominant term, per the project's own ruling,
# is verify's 4.45x non-amortisation = 28.17 ms of a 32.5 ms step). This measures what the
# already-landed-but-default-off optimisations do to the step, on the path that is already correct.
#
# One arm, one question: how much of the step does the landed launch-reduction set remove?
#   DRAFT_P3LITE_*  : draft-segment fusions (3 independent switches)
#   ATTN_MROWS / COMPRESSOR_PROJ_MROWS / ENGRAM_PROJ_MROWS / ENGRAM_GATHER_MROWS : the MROWS family
#   MOE_DOWN_BS     : the down-arm block-scaled kernel (wired, reviewed, default OFF)
#   MOE_BS_*        : the gate/up arm itself (kept OFF here: not yet correct)
#
# Judged on: step p50 of the LAST 200 steps (never a throughput back-calculation), plus the accept
# length, so the number is comparable to the blog's per-GPU 50.8 -> 218.4 ms framing.
set -uo pipefail
cd "$HOME/ferrite"
echo "=== build (hash-gated) ==="
bash "$HOME/ensure_built.sh" || { echo "=== BUILD/ARTEFACT GATE FAILED — aborting ==="; exit 1; }
git log --oneline -1

# ONE arm: the landed optimisations together. If it wins, bisect later; if it loses, the step time
# still tells us which term dominates (the arm prints the draft/verify/commit split when available).
NAME=STEP_ON
echo "########## $NAME : landed launch-reduction set ##########"
bash "$HOME/arm_run.sh" "$NAME" \
  DSV41_DRAFT_P3LITE_SEED=1 DSV41_DRAFT_P3LITE_KV=1 DSV41_DRAFT_P3LITE_ATTN=1 \
  DSV41_ATTN_MROWS=1 DSV41_COMPRESSOR_PROJ_MROWS=1 \
  DSV41_ENGRAM_PROJ_MROWS=1 DSV41_ENGRAM_GATHER_MROWS=1 \
  DSV41_MOE_DOWN_BS=1 \
  2>&1 | tee "$HOME/armrun_${NAME}.txt" | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3

echo "=== the honest step picture ==="
python3 - <<'PY'
import os, re, statistics
p = os.path.expanduser('~/armrun_STEP_ON.log')
if not os.path.exists(p):
    print("no log"); raise SystemExit
txt = open(p, errors='ignore').read()
vals = [float(m) for m in re.findall(r'\[dsv41\] step pos=\d+: ([0-9.]+)ms', txt)]
tail = vals[-200:]
if tail:
    med = statistics.median(tail)
    print(f"step p50 (last {len(tail)}) = {med:.2f} ms  ->  {1000/med:.1f} tok/s at acc 1.0")
    print(f"  p10={sorted(tail)[len(tail)//10]:.2f}ms  p90={sorted(tail)[9*len(tail)//10]:.2f}ms")
# the per-arm breakdown lines, if this build prints them
for pat in (r'\[dspark\] steps=.*', r'\[dsv41\] mean-k.*', r'\[dspark\] draft=.*'):
    for m in re.findall(pat, txt)[-1:]:
        print(" ", m[:300])
PY
echo "=== the build's landed-optimisation回执 ==="
grep -a "MROWS\|P3LITE\|DOWN_BS\|armed\|draft=" "$HOME/armrun_STEP_ON.log" | head -6
