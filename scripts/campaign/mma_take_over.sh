#!/bin/bash
# mma_take_over.sh — make the routed MoE ACTUALLY run the tcgen05 (MMA) e4m3 arm.
#
# Why this exists, from the arm's own warning lines (DSV41_EXPERT_TCGEN05_E4M3=1 EXPVERT_GROUPED=1 yet):
#   "the routed gate/up stays on the batched SIMT launch: the dense-tile tc5::e4x arm declined (the
#    routed tile is not the dense shape the launcher accepts (m % 128 == 0, dim % 64 == 0 and
#    2*inter % 64 == 0 are all required), and the GROUPED masked arm did not take the stage either"
#   "DSV41_EXPERT_GROUPED is set, but ... gate/up is on the FUSED swiglu shape
#    (DSV41_GATEUP_FUSE + fp4 mode 2 + dim % 512 == 0)"
# ⇒ the dense arm is arithmetically impossible at verify's m=6 (6 % 128 != 0), and the GROUPED masked
#   arm — the one designed for irregular row counts — was SHADOWED by the fused-swigu shape. So the
#   single variable here is turning the fused shape off and letting the grouped MMA take the stage.
#
# The dense single-row arm also faults with err 716 by its own documentation
# (docs/agent/tcgen05-716-e4m3-confound-verdict.md), so the GROUPED path is the one to pursue at m=6.
set -uo pipefail
cd "$HOME/ferrite" || exit 1
git checkout -q origin/main -- kernels crates 2>/dev/null
git log --oneline -1
bash "$HOME/ensure_built.sh" || { echo "BUILD GATE FAILED"; exit 1; }
pkill -9 -x ferrite-serve 2>/dev/null
for i in $(seq 1 20); do pgrep -x ferrite-serve >/dev/null || break; sleep 2; done
NAME=MMA_TAKE
bash "$HOME/num100.sh" "$NAME" \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 DSV41_MOE_TILELANG=0 \
  DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 DSV41_MOE_BATCH=1 DSV41_EXPERT_ILV=0 \
  DSV41_GATEUP_FUSE=0 \
  2>&1 | tail -3
echo "=== did the MMA arm take the stage THIS time? (the decline notices must be ABSENT) ==="
grep -acE "declined|did not take the stage|still runs the proven per-\(row, slot\)" "$HOME/armrun_${NAME}.log" 2>/dev/null
grep -aiE "tc05|tcgen05|grouped masked|took the stage|armed" "$HOME/armrun_${NAME}.log" 2>/dev/null | head -6
python3 - "$HOME/armrun_${NAME}.log" <<'PY'
import re, sys, statistics
t = open(sys.argv[1], errors='ignore').read()
pos = [int(x) for x in re.findall(r"\[dsv41\] step pos=(\d+)", t)]
ms  = [float(x) for x in re.findall(r"\[dsv41\] step pos=\d+: ([0-9.]+)ms", t)]
if not ms:
    print("NO STEPS"); raise SystemExit
s = sorted(ms[-200:]); med = statistics.median(s)
acc = [b-a for a, b in zip(pos, pos[1:]) if b > a]
ma = statistics.mean(acc) if acc else 0.0
print("steps=%d p50=%.2fms | mean pos advance=%.2f => tok/step=%.2f" % (len(ms), med, ma, ma+1))
print("   => %.1f tok/s at the observed accept; %.1f at tok/step 3.24 (target step<=%.2fms)"
      % (1000*(ma+1)/med, 1000*3.24/med, 1000*3.24/400))
PY
echo "=== raw reply (first 8 lines) ==="; head -8 "$HOME/num100_last.txt" 2>/dev/null
