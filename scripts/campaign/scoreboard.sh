#!/bin/bash
# scoreboard.sh — ONE measurement on the CURRENT trunk at the production-like config, with the
# launch-reduction levers that have been audited as value-neutral (mrows-numerics / shexp-numerics
# found them "发射/布局 only", i.e. not arithmetic changes).
#
# Why these flags:
#   DSV41_SPEC=1 + DSV41_DSPARK=1   the arm must actually be armed, else each step emits ONE token
#                                   while still paying the verify cost (measured: 50.55 ms at 19.8 tok/s)
#   DSV41_VERIFY_GRAPH=1            verify is ~7000 streaming launches (~20.3 ms of the 28.17 ms);
#                                   one graph dispatches them at ~0.4 us/node (~2.8 ms)
#   DSV41_GRAPH_STEP=1              ON by default, but arm_run.sh's GRAPH_OFF forces it to 0
#   DSV41_AR_V5=0                   MANDATORY with the step graph, else the all-reduce v5 path hangs
#   DSV41_MOE_TILELANG_BS=0/_HANDWRITTEN=0   the hand-written fp4 arm destroys the model
#   the fold family                 GATE_MROWS[_ROUTE] collapses moe_rows' per-row `for r in 0..m`
#                                   gate GEMV; ATTN/COMPRESSOR/ENGRAM MROWS do the same elsewhere;
#                                   DRAFT_P3LITE_* fuse the draft segments
# No rollback, no golden comparison, no past-version restore: current trunk only.
set -uo pipefail
cd "$HOME/ferrite" || exit 1
echo "=== current trunk ==="
git checkout -q origin/main -- kernels crates 2>/dev/null
git log --oneline -1
echo "=== same-source rebuild (mandatory after any partial checkout) ==="
bash "$HOME/ensure_built.sh" || { echo "BUILD GATE FAILED"; exit 1; }

pkill -9 -x ferrite-serve 2>/dev/null
for i in $(seq 1 20); do pgrep -x ferrite-serve >/dev/null || break; sleep 2; done

NAME=SCORE
echo "########## $NAME : production-like spec + both graphs + fold family ##########"
bash "$HOME/num100.sh" "$NAME" \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 \
  DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1 \
  DSV41_ATTN_MROWS=1 DSV41_COMPRESSOR_PROJ_MROWS=1 \
  DSV41_ENGRAM_PROJ_MROWS=1 DSV41_ENGRAM_GATHER_MROWS=1 \
  DSV41_DRAFT_P3LITE_SEED=1 DSV41_DRAFT_P3LITE_KV=1 DSV41_DRAFT_P3LITE_ATTN=1 \
  2>&1 | tee "$HOME/armrun_${NAME}.txt" | grep -aE "OUT:|WATCHDOG|SERVE_FAILED" | head -3

python3 - "$HOME/armrun_${NAME}.log" <<'PY'
import re, sys, statistics
t = open(sys.argv[1], errors='ignore').read()
pos = [int(x) for x in re.findall(r"\[dsv41\] step pos=(\d+)", t)]
ms  = [float(x) for x in re.findall(r"\[dsv41\] step pos=\d+: ([0-9.]+)ms", t)]
if not ms:
    print("NO STEPS — the serve or the arm failed; check the log"); raise SystemExit
s = sorted(ms[-200:]); med = statistics.median(s)
acc = [b-a for a, b in zip(pos, pos[1:]) if b > a]
mean_acc = (statistics.mean(acc) if acc else 0.0)
print("steps=%d  p50=%.2fms  p10=%.2f  p90=%.2f" % (len(ms), med, s[len(s)//10], s[9*len(s)//10]))
print("mean pos advance per step = %.2f  => tok/step = %.2f (+1 bonus)" % (mean_acc, mean_acc+1))
for tk in (mean_acc+1, 3.24, 2.0):
    print("   if tok/step=%.2f  =>  %.1f tok/s" % (tk, 1000*tk/med))
print("TARGET: 400 tok/s needs step <= %.2f ms at tok/step 3.24" % (1000*3.24/400))
for tag in ("captured", "verify_graph", "[dsv41] mean-k"):
    for mm in re.findall(r"[^\n]*" + re.escape(tag) + r"[^\n]*", t)[-2:]:
        print("  log:", mm[:150])
PY
echo "=== the arm's raw reply (byte-exact, first 12 lines) ==="
head -12 "$HOME/num100_last.txt" 2>/dev/null
