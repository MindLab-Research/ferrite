#!/bin/bash
# next_lever.sh — the single next variable after the scoreboard: open the shared-expert multi-row doors.
#
# WHY: the family ledger puts the shared expert at 10.4 ms / 27.9% of verify — the single biggest
# family — and its documented bottleneck is that the SAME weights are staged once PER ROW (moe_rows
# loops `for r in 0..m`), i.e. a 6x read amplification at m=6. The code documents the fix as an
# existing, default-OFF door:
#   DSV41_SH_EXP_MROWS=1  the whole (w1/w3 -> swiglu -> w2) chain as ONE multi-row pass
#                         5 launches/row  ->  1 launch/layer + quant_rows
#   DSV41_SH_PAIR_M=1     the template<M> fused form (gemm_fp8_sh_exp_fused<M>)
#   (SH_EXP_MX2, default ON, is the M=1 w1|w3 fusion already in the baseline)
# Everything else stays exactly as scoreboard.sh so this is ONE variable.
set -uo pipefail
cd "$HOME/ferrite" || exit 1
git checkout -q origin/main -- kernels crates 2>/dev/null
git log --oneline -1
bash "$HOME/ensure_built.sh" || { echo "BUILD GATE FAILED"; exit 1; }
pkill -9 -x ferrite-serve 2>/dev/null
for i in $(seq 1 20); do pgrep -x ferrite-serve >/dev/null || break; sleep 2; done
NAME=LEVER_SH
bash "$HOME/num100.sh" "$NAME" \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 \
  DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1 \
  DSV41_ATTN_MROWS=1 DSV41_COMPRESSOR_PROJ_MROWS=1 \
  DSV41_ENGRAM_PROJ_MROWS=1 DSV41_ENGRAM_GATHER_MROWS=1 \
  DSV41_DRAFT_P3LITE_SEED=1 DSV41_DRAFT_P3LITE_KV=1 DSV41_DRAFT_P3LITE_ATTN=1 \
  DSV41_SH_EXP_MROWS=1 DSV41_SH_PAIR_M=1 \
  2>&1 | tee "$HOME/armrun_${NAME}.txt" | grep -aE "OUT:|WATCHDOG|SERVE_FAILED" | head -3
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
print("steps=%d p50=%.2fms p10=%.2f p90=%.2f | mean pos advance=%.2f => tok/step=%.2f"
      % (len(ms), med, s[len(s)//10], s[9*len(s)//10], ma, ma+1))
print("   => %.1f tok/s at the observed accept; %.1f at tok/step 3.24; target step<=%.2fms"
      % (1000*(ma+1)/med, 1000*3.24/med, 1000*3.24/400))
PY
echo "=== raw reply (byte-exact, first 10 lines) ==="; head -10 "$HOME/num100_last.txt" 2>/dev/null
