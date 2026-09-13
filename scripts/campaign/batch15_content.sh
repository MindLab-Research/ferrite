#!/bin/bash
# Batch 15: two things that were never actually tested, in one build.
#
# 1. THE STAGED-OPERAND CONTENT CHECK, FOR REAL. DSV41_MOE_BS_SFDUMP was a DEAD instrument until the
#    wiring fix: the kernel's dump block tests `g_sfdump != nullptr && g_sfdump_on && k == g_sfdump_k`,
#    and no code ever set the first two, so the scratch stayed zero and sfdump_check.py compared zeros.
#    Now the shim hands the scratch to the symbol and arms the switch, so this run is the FIRST real
#    answer to "does the staged content equal what the official checkpoint says it should be?".
#    A mismatch here names the injected byte; agreement means the content is right and the defect is
#    in the use of it (which batch14's raw-tile probe and the per-stage test then split).
#
# 2. THE SURVIVOR FIX. output-stale-audit's §5 mechanism (tcgen05.alloc does not zero TMEM + a wait
#    that can be one phase short => the epilogue reads columns this launch never wrote) is tested by
#    DSV41_MOE_BS_MBAR_PERSTAGE=1, where every K-stage gets a barrier of its own that is arrived
#    exactly once, so the wait's phase is the constant 0 and there is no accounting to desynchronise.
#    The verdict is the same triple used everywhere: oracle agreement, ZERO_A == exactly 0, and
#    bit-identical output across two runs.
set -uo pipefail
cd "$HOME/ferrite"
echo "=== build (hash-gated) ==="
bash "$HOME/ensure_built.sh" || { echo "=== BUILD/ARTEFACT GATE FAILED — aborting before any measurement ==="; exit 1; }
git log --oneline -1

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -2
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "staged-operand dump armed\|mbarrier per-stage\|mbarrier ring\|bs-sfdump\|bs-gc" \
      "$HOME/armrun_${name}.log" | head -4
}

rm -rf /tmp/b15_ps /tmp/b15_ps_b /tmp/b15_za /tmp/sfd_b15
run B15_PS   DSV41_GATEUP_DUMP=/tmp/b15_ps   DSV41_MOE_BS_SFDUMP=/tmp/sfd_b15 \
             DSV41_MOE_BS_MBAR_PERSTAGE=1
run B15_PS2  DSV41_GATEUP_DUMP=/tmp/b15_ps_b DSV41_MOE_BS_MBAR_PERSTAGE=1
run B15_PZA  DSV41_GATEUP_DUMP=/tmp/b15_za   DSV41_MOE_BS_MBAR_PERSTAGE=1 DSV41_MOE_BS_ZERO_A=1
run B15_DC   DSV41_GATEUP_DUMP=/tmp/b15_dc   DSV41_MOE_BS_DCLEAR=1 DSV41_MOE_BS_SFDUMP=/tmp/sfd_b15
run B15_DCZA DSV41_GATEUP_DUMP=/tmp/b15_dcza DSV41_MOE_BS_DCLEAR=1 DSV41_MOE_BS_ZERO_A=1

echo "=== 1. determinism of the per-stage arm ==="
cmp -s /tmp/b15_ps/eager/gateup.f32 /tmp/b15_ps_b/eager/gateup.f32 \
  && echo "IDENTICAL (deterministic)" || echo "DIFFERS (still non-deterministic)"
echo "=== 2. ZERO_A under the per-stage barriers (must be EXACTLY 0) ==="
python3 - <<'PY'
import os, numpy as np
p = '/tmp/b15_za/eager/gateup.f32'
if os.path.exists(p):
    a = np.fromfile(p, dtype='<f4')
    print(f"max|.|={np.abs(a).max():.6g} nonzero={int((a!=0).sum())} -> "
          + ("A OPERAND IS CONSUMED AS STAGED ✓" if float(np.abs(a).max()) == 0.0
             else "still not consuming the staged A"))
else:
    print("no dump")
PY
echo "=== 2b. DCLEAR (the (b)-hypothesis fix): ZERO_A must be EXACTLY 0, and the oracle must agree ==="
python3 - <<'PY2'
import os, numpy as np
p = '/tmp/b15_dcza/eager/gateup.f32'
if os.path.exists(p):
    a = np.fromfile(p, dtype='<f4')
    print(f"DCLEAR+ZERO_A: max|.|={np.abs(a).max():.6g} nonzero={int((a!=0).sum())} -> "
          + ("A IS CONSUMED AS STAGED (the accumulator clear is no longer load-bearing) OK"
             if float(np.abs(a).max()) == 0.0 else "still not consuming the staged A"))
else:
    print("DCLEAR+ZERO_A: no dump")
PY2
for e in /tmp/b15_dc/eager /tmp/b15_dc; do
  [ -f "$e/gateup.f32" ] && { python3 "$HOME/gdu_cmp.py" "$e/gateup.f32" /tmp/gu_bs_new.f32 6 640 10.0 2>&1 | head -5; break; }
done
echo "=== 3. oracle agreement (the official-semantics verdict) ==="
for d in /tmp/b15_ps; do
  e=$d; [ -d "$d/eager" ] && e=$d/eager
  [ -f "$e/gateup.f32" ] && python3 "$HOME/gdu_cmp.py" "$e/gateup.f32" /tmp/gu_bs_new.f32 6 640 10.0 2>&1 | head -6
done
echo "=== 4. THE FIRST REAL STAGED-CONTENT CHECK ==="
if [ -f "$HOME/sfdump_check.py" ]; then
  for d in /tmp/sfd_b15/eager /tmp/sfd_b15 /tmp/sfd_b15/rows; do
    [ -d "$d" ] && { echo "--- $d"; timeout 900 python3 "$HOME/sfdump_check.py" --in-dir "$d" 2>&1 | tail -16; }
  done
else
  echo "(sfdump_check.py missing)"
fi
echo "=== 5. replay: output vs its own staged operands' product ==="
if [ -f "$HOME/sfdump_replay.py" ] && [ -d /tmp/sfd_b15 ]; then
  for e in /tmp/b15_ps/eager /tmp/b15_ps; do
    [ -d "$e" ] || e=$(dirname "$e")
    [ -d "$e" ] && timeout 900 python3 "$HOME/sfdump_replay.py" --sfdump /tmp/sfd_b15 --gu "$e" 2>&1 | tail -4
  done
fi
