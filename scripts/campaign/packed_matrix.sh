#!/bin/bash
# PACKED controlled matrix — run AFTER the packed geometry is chosen.
# Fixes §27's confound (all arms share one freshly built binary) and varies ONE thing per arm.
# [NC] (numeric) is the primary criterion; text is only a hint; all arms run with the graph
# OFF so the BS arm is exercised on every call.
set -uo pipefail
cd ~/ferrite
# ALL FIVE graph gates must be off: even with the whole-step graph disabled, the per-op
# captures (FERRITE_GRAPH_{LAYER,MOE,MID,DSA}) still put the BS shim's call inside a capture,
# which makes it decline and makes NUMCHECK's non-capture guard fail (that is why [NC] never
# printed). See docs/agent/moe-bs-crash-investigation.md §16/§33/§35.
GRAPH_OFF="FERRITE_GRAPH=0 FERRITE_GRAPH_LAYER=0 FERRITE_GRAPH_MOE=0 FERRITE_GRAPH_MID=0 FERRITE_GRAPH_DSA=0"
COMMON="DSV41_MOE_BS_PACKED=1 DSV41_GRAPH_STEP=0 $GRAPH_OFF"
run() { bash ~/arm_run.sh "$@" 2>&1 | tail -14; }
echo "### Q1: packed, geom0, canonical, no swapAB ###"
run Q1_geom0_canon $COMMON DSV41_MOE_BS_PACKGEOM=0 DSV41_MOE_BS_CANON=1
echo "### Q2: packed, geom0, canonical, swapAB ###"
run Q2_geom0_canon_swap $COMMON DSV41_MOE_BS_PACKGEOM=0 DSV41_MOE_BS_CANON=1 DSV41_MOE_BS_SWAPAB=1
echo "### Q3: packed, geom1, canonical, no swapAB ###"
run Q3_geom1_canon $COMMON DSV41_MOE_BS_PACKGEOM=1 DSV41_MOE_BS_CANON=1
echo "### Q4: packed, geom1, canonical, swapAB ###"
run Q4_geom1_canon_swap $COMMON DSV41_MOE_BS_PACKGEOM=1 DSV41_MOE_BS_CANON=1 DSV41_MOE_BS_SWAPAB=1
echo "### Q5: best of the above + SFREV ###"
run Q5_sfrev $COMMON DSV41_MOE_BS_PACKGEOM=1 DSV41_MOE_BS_CANON=1 DSV41_MOE_BS_SWAPAB=1 DSV41_MOE_BS_SFREV=1
pkill -9 -x ferrite-serve 2>/dev/null
echo "### SUMMARY (primary: [NC] rel; secondary: text) ###"
for a in Q1_geom0_canon Q2_geom0_canon_swap Q3_geom1_canon Q4_geom1_canon_swap Q5_sfrev; do
  printf '%-22s ' "$a"
  if grep -q "\[NC\] WORST" $HOME/armrun_$a.log 2>/dev/null; then
    grep -m1 "\[NC\] WORST" $HOME/armrun_$a.log | sed 's/^\[NC\] //'
  else
    printf '[NC] absent | '
    grep -m1 'OUT:' $HOME/armrun_$a.log 2>/dev/null | head -c 90; echo
  fi
done
echo "### PACKED MATRIX DONE ###"
