#!/bin/bash
# draft_side.sh — two NEW levers from the blog mapping that have never been tried here, one per arm.
#
# 1. DSV41_DRAFT_GRAPH=1 (dspark_dev.rs:249, default OFF): the blog's step 12 analogue on the DRAFT
#    side — the draft is 3.87 ms of our step and the mapping calls this its biggest lever.
# 2. DSV41_ATTN_PROJ_ALIGN=1 (dspark_dev.rs:159, default OFF): an ACCEPT lever, and the code itself
#    labels it "NUMERICAL fix, not perf" — the draft's four projections run an m16 x n TILE program
#    where the official runs the same F.linear on both sides, so aligning them is a correctness
#    improvement that should also raise the accept toward the real 2.2.
#
# Base is the production-like spec config with both graphs and AR_V5=0, BS arm off (the hand-written
# fp4 arm destroys the model), and NO GATEUP_FUSE=0 (measured: 60 ms vs 52 ms).
set -uo pipefail
cd "$HOME/ferrite" || exit 1
git checkout -q origin/main -- kernels crates 2>/dev/null
git log --oneline -1
bash "$HOME/ensure_built.sh" || { echo "BUILD GATE FAILED"; exit 1; }
BASE="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0"
# VERIFY_FORK is the verify-side counterpart of the plain path's double-stream overlap (the blog's
# step 8, +23%): the mapping reports it exists, is default OFF, and has never been wired on the
# verify path ("把 layer() 的三处 side-stream fork 原样搬进 verify").
for pair in "DRAFTGRAPH:DSV41_DRAFT_GRAPH=1" "PROJALIGN:DSV41_ATTN_PROJ_ALIGN=1" \
            "VERIFYFORK:DSV41_VERIFY_FORK=1"; do
  tag=${pair%%:*}; envs=${pair#*:}
  echo "########## $tag : $envs ##########"
  pkill -9 -x ferrite-serve 2>/dev/null
  for i in $(seq 1 20); do pgrep -x ferrite-serve >/dev/null || break; sleep 2; done
  bash "$HOME/num100.sh" "DR_$tag" $BASE $envs 2>&1 | tail -2
  python3 - "$HOME/armrun_DR_$tag.log" "$tag" <<'PY'
import re, sys, statistics
t = open(sys.argv[1], errors='ignore').read()
pos = [int(x) for x in re.findall(r"\[dsv41\] step pos=(\d+)", t)]
ms  = [float(x) for x in re.findall(r"\[dsv41\] step pos=\d+: ([0-9.]+)ms", t)]
if not ms:
    print("  %s: NO STEPS" % sys.argv[2]); raise SystemExit
s = sorted(ms[-200:]); med = statistics.median(s)
acc = [b-a for a, b in zip(pos, pos[1:]) if b > a]
ma = statistics.mean(acc) if acc else 0.0
print("  %s: steps=%d p50=%.2fms mean_adv=%.2f => tok/step=%.2f => %.1f tok/s (%.1f at 3.24)"
      % (sys.argv[2], len(ms), med, ma, ma+1, 1000*(ma+1)/med, 1000*3.24/med))
PY
done
echo "=== raw reply of the last arm (first 8 lines) ==="; head -8 "$HOME/num100_last.txt" 2>/dev/null
