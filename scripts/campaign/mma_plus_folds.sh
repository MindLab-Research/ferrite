#!/bin/bash
# mma_plus_folds.sh — the next single variable after the MMA take-over: add the audited
# value-neutral launch-reduction family on top of whatever made the grouped tcgen05 e4m3 arm take the
# stage (DSV41_GATEUP_FUSE=0 + EXPT_TCGEN05_E4M3 + EXPT_GROUPED + EXPT_ACT_E4M3 + EXPT_ILV=0).
#
# Why these: verify's cost is launch-count bound (SGLang's own lesson and our measurement: "one
# activation row per launch" plus `for r in 0..m` gate GEMV in moe_rows). mrows-numerics and
# shexp-numerics both concluded these doors change only the emission structure, not the arithmetic
# (same block=32, same ue8m0 scales, same f32 accumulation order), and P2-vs-P1 text was byte-identical
# which is the same conclusion from the behaviour side.
set -uo pipefail
cd "$HOME/ferrite" || exit 1
git checkout -q origin/main -- kernels crates 2>/dev/null
git log --oneline -1
bash "$HOME/ensure_built.sh" || { echo "BUILD GATE FAILED"; exit 1; }
pkill -9 -x ferrite-serve 2>/dev/null
for i in $(seq 1 20); do pgrep -x ferrite-serve >/dev/null || break; sleep 2; done
NAME=MMA_FOLDS
bash "$HOME/num100.sh" "$NAME" \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 DSV41_MOE_TILELANG=0 \
  DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 DSV41_MOE_BATCH=1 DSV41_EXPERT_ILV=0 \
  DSV41_GATEUP_FUSE=0 \
  DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1 \
  DSV41_ATTN_MROWS=1 DSV41_COMPRESSOR_PROJ_MROWS=1 \
  DSV41_ENGRAM_PROJ_MROWS=1 DSV41_ENGRAM_GATHER_MROWS=1 \
  DSV41_DRAFT_P3LITE_SEED=1 DSV41_DRAFT_P3LITE_KV=1 DSV41_DRAFT_P3LITE_ATTN=1 \
  2>&1 | tee "$HOME/armrun_${NAME}.txt" | grep -aE "OUT:|WATCHDOG" | head -3
echo "=== decline notices (must stay 0) ==="; grep -acE "declined|did not take the stage|stays on the batched SIMT" "$HOME/armrun_${NAME}.log" 2>/dev/null
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
print("   => %.1f tok/s at that accept; %.1f at tok/step 3.24 (target step<=%.2fms for 400)"
      % (1000*(ma+1)/med, 1000*3.24/med, 1000*3.24/400))
PY
echo "=== raw reply (first 8 lines) ==="; head -8 "$HOME/num100_last.txt" 2>/dev/null
