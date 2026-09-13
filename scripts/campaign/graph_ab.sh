#!/bin/bash
# Graph A/B: the production-like config measured 51 ms/step against the documented 32.5 ms baseline, so
# something I turned ON is hurting. The candidates are exactly the flags the docs' baseline did NOT
# have. One variable at a time, all with DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_AR_V5=0 (mandatory: without
# AR_V5=0 the whole-step graph hangs the all-reduce v5 path) and the correct path (BS arm off).
#
# Read the accept from the step positions, not from the printed tok/s: the serve prints 1/step, so a
# pos jump larger than 1 per step is the real evidence (23 -> 27 was seen, i.e. +4).
set -uo pipefail
cd "$HOME/ferrite"
bash "$HOME/ensure_built.sh" || { echo "[runner] build gate FAILED"; exit 1; }

judge () {
  python3 - "$1" <<'PY'
import re, sys, statistics
t = open(sys.argv[1], errors='ignore').read()
pos = [int(x) for x in re.findall(r"\[dsv41\] step pos=(\d+)", t)]
ms  = [float(x) for x in re.findall(r"\[dsv41\] step pos=\d+: ([0-9.]+)ms", t)]
if not ms:
    print("  no steps"); raise SystemExit
s = sorted(ms[-200:]); med = statistics.median(s)
accepts = [b - a for a, b in zip(pos, pos[1:])]
m = statistics.mean(accepts) if accepts else 0
print(f"  steps={len(ms)} p50={med:.2f}ms  pos-jumps: n={len(accepts)} mean={m:.2f} max={max(accepts) if accepts else 0}")
print(f"  => tok/step = mean_accept (+1 bonus) = {m+1:.2f}  => tok/s = {1000*(m+1)/med:.1f}")
for mm in re.findall(r"\[diff\][^\n]*", t)[-2:]:
    print("  diff:", mm[:200])
PY
}

for cfg in "VERIFY_GRAPH=1 GRAPH_STEP=1" "VERIFY_GRAPH=0 GRAPH_STEP=1" "VERIFY_GRAPH=1 GRAPH_STEP=0" "VERIFY_GRAPH=0 GRAPH_STEP=0"; do
  name="GAB_$(echo "$cfg" | tr ' =' '__')"
  echo "########## $name : $cfg ##########"
  envs=""
  for kv in $cfg; do envs="$envs DSV41_$kv"; done
  bash "$HOME/num100.sh" "$name" DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_AR_V5=0 \
    DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 $envs \
    2>&1 | tee "$HOME/armrun_${name}.txt" | grep -aE "OUT:" | head -1
  judge "$HOME/armrun_${name}.log"
done
