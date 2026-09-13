#!/bin/bash
# Batch 13 (the stale-output discriminator): pre-clear the caller's output buffer before the arm
# runs (`DSV41_MOE_BS_PRECLEAR=1` => an async memset on the same stream, no sync in the shim).
#
# READ THE RESULT LIKE THIS:
#   * pre-cleared output MATCHES the official oracle  => the garbage was STALE RESIDUE that nothing
#     in this call wrote; pre-clearing is also the fix (and it names the writer that is missing);
#   * pre-cleared output is STILL the same garbage     => the value really is computed by this call,
#     and the search stays on the arithmetic/delivery (batch12's g_c probe then splits MMA vs scatter).
set -uo pipefail
cd "$HOME/ferrite"
echo "=== build (hash-gated: no rebuild when the .cu content is unchanged) ==="
bash "$HOME/ensure_built.sh" || { echo "=== BUILD/ARTEFACT GATE FAILED — aborting before any measurement ==="; exit 1; }
git log --oneline -1

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -2
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "gateup-dump.*done\|bs-gc\|ZERO-DIAG" "$HOME/armrun_${name}.log" | head -3
}

rm -rf /tmp/pc_1 /tmp/pc_2 /tmp/guin_pc
run PC_1 DSV41_GATEUP_DUMP=/tmp/pc_1 DSV41_MOE_BS_SFDUMP=/tmp/guin_pc DSV41_MOE_BS_PRECLEAR=1
run PC_2 DSV41_GATEUP_DUMP=/tmp/pc_2 DSV41_MOE_BS_PRECLEAR=1

echo "=== determinism of the pre-cleared arm (two runs) ==="
cmp -s /tmp/pc_1/eager/gateup.f32 /tmp/pc_2/eager/gateup.f32 \
  && echo "IDENTICAL (deterministic)" || echo "DIFFERS (still non-deterministic)"

echo "=== pre-cleared arm vs the official-semantics oracle (same dumped input) ==="
if [ -f /tmp/pc_1/eager/gateup.f32 ] && [ -f /tmp/gu_bs_new.f32 ]; then
  python3 "$HOME/gdu_cmp.py" /tmp/pc_1/eager/gateup.f32 /tmp/gu_bs_new.f32 6 640 10.0 2>&1 | head -8
else
  echo "(missing dumps)"
fi

echo "=== the raw MMA tile under pre-clear (all zeros?) + its zero-operand behaviour ==="
python3 - <<'PY'
import os, numpy as np
for p in ('/tmp/guin_pc/eager/gc_seg0.f32',):
    if os.path.exists(p):
        a = np.fromfile(p, dtype='<f4')
        print(f"{p}: n={a.size} max|.|={np.abs(a).max():.6g} nonzero={int((a!=0).sum())}")
    else:
        print(f"{p}: missing (the .so may predate the g_c accessor)")
e = '/tmp/pc_1/eager/gateup.f32'
if os.path.exists(e):
    a = np.fromfile(e, dtype='<f4')
    print(f"pre-cleared ex_act_b: n={a.size} max|.|={np.abs(a).max():.6g} "
          f"nonzero={int((a!=0).sum())} -> "
          + ("NOTHING WAS WRITTEN by this call" if float(np.abs(a).max()) == 0.0 else
             "the arm did write values"))
PY
