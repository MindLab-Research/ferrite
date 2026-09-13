#!/bin/bash
# promote_all.sh — promote the five precision gates in ONE GPU window, one arm per gate.
# Each arm turns on exactly ONE gate (the "one variable at a time" rule is preserved: no arm
# changes two gates), runs the diagnostic arm so the gate's DBG readback can print (§58: the
# five graph gates stay off), takes the mechanised red-line verdict, then runs the fast arm
# for a real p50. Inspect each block before adding the gate to the shipping scripts.
set -uo pipefail
cd ~/ferrite || exit 1
GATES=(
  "DSV41_ROUTED_DOWN_QUANT=1 DSV41_ROUTED_DOWN_QUANT_DBG=1"
  "DSV41_WINDOW_KV_QUANT=1 DSV41_WINDOW_KV_QUANT_DBG=1"
  "DSV41_COMPRESS_LATENT_QUANT=1 DSV41_COMPRESS_LATENT_QUANT_DBG=1"
  "DSV41_INDEXER_FP4_RT=1 DSV41_INDEXER_FP4_RT_DBG=1"
  "DSV41_ATTN_P_BF16=1 DSV41_ATTN_P_BF16_DBG=1"
)
NAMES=(RQ WKV LAT IDX PB)
for i in "${!GATES[@]}"; do
  N=${NAMES[$i]}; G=${GATES[$i]}
  echo "########## ${N}: ${G} ##########"
  echo "---- diagnostic arm (graphs OFF; DBG readback expected) ----"
  bash ~/arm_run.sh "${N}_diag" $G 2>&1 | tail -6
  bash ~/arm_summary.sh "${N}_diag"
  echo "---- gate probe lines (expect zero difference vs the independent host reference) ----"
  grep -E "routed-down-quant|win-kv|window-kv|compress-latent|indexer-fp4|attn-p" ~/armrun_${N}_diag.log | head -14 \
    || echo "(no probe output — re-check the capture guard, §58/§73)"
  echo "---- mechanised red lines ----"
  python3 ~/wq_check.py --log ~/armrun_${N}_diag.log || echo "  (FAIL/WARN — do NOT promote this gate)"
  echo "---- fast arm (graphs ON) — real p50, confirm no throughput regression ----"
  bash ~/arm_run_fast.sh "${N}_fast" $G 2>&1 | tail -5
  bash ~/arm_summary.sh "${N}_fast"
  python3 ~/wq_check.py --log ~/armrun_${N}_fast.log || true
  echo
done
echo "########## promote_all DONE — promote only the gates whose probe matched and whose red lines passed ##########"
