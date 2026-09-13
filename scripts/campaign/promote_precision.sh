#!/bin/bash
# promote_precision.sh "<GATE_ENV=1> [DBG_ENV=1]" — the three-step promotion flow for a
# precision gate. Never promote two gates at once (one variable at a time, §63).
#   step 1  diagnostic arm with all five graph gates OFF (so the DBG readback can run, §58)
#           -> the five per-element values must match the independent host reference exactly
#   step 2  text red lines on the same arm: 1..61 counting + no repeats/garbage (wq_check.py)
#   step 3  the real performance arm (graphs ON) so the step time stays meaningful
set -uo pipefail
GATE=${1:?usage: promote_precision.sh \"DSV41_XXX=1 [DSV41_XXX_DBG=1]\"}
echo "############ gate: $GATE ############"
echo "---- step 1/3: diagnostic arm (graphs OFF) — DBG readback vs host reference ----"
bash ~/arm_run.sh PP_diag $GATE 2>&1 | tail -6
bash ~/arm_summary.sh PP_diag
echo "-- probe output (expect zero per-element difference vs the host reference) --"
grep -E "routed-down-quant|compress-latent|indexer-fp4|DBG" ~/armrun_PP_diag.log | head -20 \
  || echo "(no probe output — check the capture guard: the arm must keep all five graph gates off, §58)"
echo "---- step 2/3: text red lines (wq_check.py) ----"
python3 ~/wq_check.py --log ~/armrun_PP_diag.log || echo "(FAIL/WARN — do NOT promote)"
echo "---- step 3/3: fast arm (graphs ON) — confirm no throughput/behaviour regression ----"
bash ~/arm_run_fast.sh PP_fast $GATE 2>&1 | tail -6
bash ~/arm_summary.sh PP_fast
python3 ~/wq_check.py --log ~/armrun_PP_fast.log || true
echo "############ decision ############"
echo "promote only if: (a) the DBG difference is zero modulo 1 ulp, (b) wq_check PASS on both"
echo "arms, and (c) the fast arm's p50 shows no regression. Then add the env to"
echo "push400_hw_test.sh + verify_correct.sh (the shipping configuration) and re-run the"
echo "full-gate regression."
