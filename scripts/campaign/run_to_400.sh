#!/bin/bash
# The two-step run to 400 tok/s, chained so there is no idle gap between them.
#
# Why these two and not more: the code comments give the budget precisely.
#   * verify is ~7000 STREAMING LAUNCHES per verify (40 layers x ~170 nodes) at ~2.9 us each
#     (~20.3 ms) — i.e. almost the whole 28.17 ms is launch-submit time, not compute. The verify
#     graph collapses that block into ONE graph whose per-node dispatch floor is ~0.4 us
#     (~2.8 ms). That is the single biggest lever, and the comment says it is safe: the per-verify
#     inputs are refreshed on DEVICE buffers outside the capture, so no launch argument changes.
#   * the MoE gate/up+down TileLang arm measures 70.0 us/layer (up 45.5 + dn 24.5) = 28% of the
#     250 us SIMT baseline, i.e. ~7.2 ms off a 40-layer step.
#
# Both steps start from the CORRECT path (the hand-written fp4 BS arm destroys the model: the
# 1..100 probe produced nothing but repeated junk), and both are judged the same way: the p50 of the
# last 200 step lines (never back-calculated from throughput) plus the 1..100 text, whose first 61
# lines must read 1..61.
set -uo pipefail
cd "$HOME/ferrite"

judge () {
  local tag="$1" log="$2"
  echo "=================== $tag ==================="
  echo "--- text (first 40 lines of the reply) ---"
  python3 - "$log" <<'PY'
import re, sys
txt = open(sys.argv[1], errors='ignore').read()
m = re.findall(r'\[[A-Z0-9_]+\] OUT: (.*)', txt)
if m:
    body = m[-1]
    print("\n".join(body.split('\\n')[:40]))
else:
    print("(no OUT line)")
PY
  echo "--- step p50 ---"
  python3 - "$log" <<'PY'
import re, statistics, sys
txt = open(sys.argv[1], errors='ignore').read()
v = [float(m) for m in re.findall(r'\[dsv41\] step pos=\d+: ([0-9.]+)ms', txt)]
tail = v[-200:]
if not tail:
    print("no step lines"); raise SystemExit
med = statistics.median(tail)
s = sorted(tail)
print(f"n={len(tail)} p50={med:.2f}ms p10={s[len(s)//10]:.2f}ms p90={s[9*len(s)//10]:.2f}ms")
print(f"  => {1000/med:.1f} tok/s at acc 1.0 ; at acc 2.2 (tok/step = 3.24): {3.24*1000/med:.1f} tok/s")
PY
}

echo "############ P1: correct path + VERIFY GRAPH (the ~20ms -> ~3ms lever) ############"
bash "$HOME/num100.sh" STEP_P1 \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 DSV41_VERIFY_GRAPH=1
judge "P1 (verify graph)" "$HOME/armrun_STEP_P1.log"

echo "############ P2: P1 + the launch-reduction family (MoE + gate folds) ############"
bash "$HOME/num100.sh" STEP_P2 \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 DSV41_VERIFY_GRAPH=1 \
  DSV41_MOE_TILELANG=1 \
  DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1 \
  DSV41_ATTN_MROWS=1 DSV41_COMPRESSOR_PROJ_MROWS=1 \
  DSV41_ENGRAM_PROJ_MROWS=1 DSV41_ENGRAM_GATHER_MROWS=1
judge "P2 (graph + MoE + gate folds)" "$HOME/armrun_STEP_P2.log"

echo "############ the accept length actually observed (the goal's other half) ############"
grep -a "mean-k\|accept" "$HOME/armrun_STEP_P2.log" | tail -3
