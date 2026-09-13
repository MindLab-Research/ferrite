#!/bin/bash
# Batch 16: the three new fix/detector gates, alone and combined, judged the same way everywhere —
# oracle agreement + determinism + "no NaN" (the sentinel mode self-checks).
#
#   DCLEAR=2 : qNaN sentinel into D before the K loop. On the correct path the first (enable_d=0) MMA
#              overwrites all 128 columns, so the sentinel must NOT survive into the output. If any
#              NaN appears, some column was never written by this launch — i.e. the foreign-residue
#              mechanism is real and the sentinel just made it visible (which is the point).
#   DCLEAR=1 : the same but with zeros (the fix without the detector).
#   DRAIN=1  : one final commit+wait after the K loop — "all MMAs of this launch completed", with no
#              per-stage parity accounting involved.
#   MBAR_PERSTAGE=1 : a dedicated barrier per K-stage (arrived once, phase always 0).
#
# Verdicts: (a) oracle agreement, (b) ZERO_A must be EXACTLY 0, (c) bit-identical across two runs,
# (d) NaN census — must be 0 on the correct path, >0 proves the read-foreign-columns mechanism.
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
  grep -a "explicit D clear\|final drain\|mbarrier per-stage\|staged-operand dump armed\|bs-gc" \
      "$HOME/armrun_${name}.log" | head -4
}

rm -rf /tmp/f1 /tmp/f2 /tmp/f3 /tmp/f4 /tmp/f4b /tmp/f1za /tmp/sfdf1
run F1_DC2   DSV41_GATEUP_DUMP=/tmp/f1   DSV41_MOE_BS_SFDUMP=/tmp/sfdf1 DSV41_MOE_BS_DCLEAR=2
run F2_DRAIN DSV41_GATEUP_DUMP=/tmp/f2   DSV41_MOE_BS_DRAIN=1
run F3_PS    DSV41_GATEUP_DUMP=/tmp/f3   DSV41_MOE_BS_MBAR_PERSTAGE=1
run F4_ALL   DSV41_GATEUP_DUMP=/tmp/f4   DSV41_MOE_BS_DCLEAR=2 DSV41_MOE_BS_DRAIN=1 DSV41_MOE_BS_MBAR_PERSTAGE=1
run F4_ALL_B DSV41_GATEUP_DUMP=/tmp/f4b  DSV41_MOE_BS_DCLEAR=2 DSV41_MOE_BS_DRAIN=1 DSV41_MOE_BS_MBAR_PERSTAGE=1
run F1_DC2ZA DSV41_GATEUP_DUMP=/tmp/f1za DSV41_MOE_BS_DCLEAR=2 DSV41_MOE_BS_ZERO_A=1

echo "=== verdicts ==="
python3 - <<'PY'
import os, numpy as np
def rd(d):
    for p in (f'{d}/eager/gateup.f32', f'{d}/gateup.f32'):
        if os.path.exists(p):
            return np.fromfile(p, dtype='<f4')
    return None
ref = rd('/tmp/gu_in_GD4_OLD') if os.path.exists('/tmp/gu_in_GD4_OLD') else None
oracle = np.fromfile('/tmp/gu_bs_new.f32', dtype='<f4') if os.path.exists('/tmp/gu_bs_new.f32') else None
for tag, d in (('DCLEAR=2', '/tmp/f1'), ('DRAIN', '/tmp/f2'), ('MBAR_PERSTAGE', '/tmp/f3'),
               ('ALL THREE', '/tmp/f4'), ('ALL THREE (2nd run)', '/tmp/f4b'),
               ('DCLEAR=2 + ZERO_A', '/tmp/f1za')):
    a = rd(d)
    if a is None:
        print(f'{tag:22} no dump'); continue
    nan = int(np.isnan(a).sum()); mx = float(np.nanmax(np.abs(a))) if a.size else 0.0
    line = f'{tag:22} n={a.size:5d} max|.|={mx:.6g} nan={nan} nonzero={int((a!=0).sum()):5d}'
    if oracle is not None and a.size == oracle.size:
        den = np.maximum(np.abs(oracle), 1e-6); rel = np.abs(a - oracle) / den
        line += (f'  |  vs oracle: med rel={np.nanmedian(rel):.4g} '
                 f'corr={np.corrcoef(np.nan_to_num(a), oracle)[0,1]:+.5f}')
    print(line)
a4, b4 = rd('/tmp/f4'), rd('/tmp/f4b')
if a4 is not None and b4 is not None:
    print('determinism (ALL THREE, two runs):',
          'IDENTICAL' if np.array_equal(a4, b4) else f'DIFFERS (max|d|={np.nanmax(np.abs(a4-b4)):.6g})')
PY
echo
echo "READ: nan==0 everywhere => no column was left unwritten (the sentinel never survived)."
echo "      nan>0          => the foreign-column mechanism is REAL and now loud instead of silent."
echo "      F1_DC2ZA max|.|==0 => the staged A IS consumed (the accumulator clear is no longer load-bearing)."
