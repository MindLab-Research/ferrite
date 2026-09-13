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
# The ZERO probes contradict each other on the *old* data: ZERO_ALL gives ~0 (1.2e-36) while ZERO_A
# alone gives 0.248 and ZERO_SF alone 0.165 — impossible for a product, so at least one of those
# single-factor probes is broken (or only partially applied). The raw tile settles it: if `gc` is
# non-zero with A supposedly zeroed, A really reached the MMA (probe ineffective); if `gc` is zero
# while `ex_act_b` is not, the value appears AFTER the MMA (scatter/stale), which is a different fix.
run F5_ZA_GC DSV41_GATEUP_DUMP=/tmp/f5z DSV41_MOE_BS_SFDUMP=/tmp/sfdf5 DSV41_MOE_BS_ZERO_A=1
run F5_SF_GC DSV41_GATEUP_DUMP=/tmp/f5s DSV41_MOE_BS_SFDUMP=/tmp/sfdf5s DSV41_MOE_BS_ZERO_SF=1
# THE decisive probe: A and SF zeroed, expert ids INTACT (mode 5), so the tables stay valid and
# the scatter actually runs. A zero product must therefore give an exactly-zero output; anything
# non-zero means the MMA is not reading the operands we staged.
run F6_ASF   DSV41_GATEUP_DUMP=/tmp/f6  DSV41_MOE_BS_SFDUMP=/tmp/sfdf6 DSV41_MOE_BS_ZERO_ASF=1

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
echo
echo "=== the ZERO-probe contradiction, settled on the RAW tile (before the scatter) ==="
python3 - <<'PY'
import os, numpy as np
def rd(p):
    return np.fromfile(p, dtype='<f4') if os.path.exists(p) else None
for tag, d in (('ZERO_A ', '/tmp/f5z'), ('ZERO_SF', '/tmp/f5s')):
    gc = rd(f'{d}/eager/gc_all.f32') if os.path.exists(f'{d}/eager/gc_all.f32') else rd(f'{d}/eager/gc_seg0.f32')
    e = rd(f'{d}/eager/gateup.f32')
    gs = f'{d}/eager/gc_all.f32' if os.path.exists(f'{d}/eager/gc_all.f32') else f'{d}/eager/gc_seg0.f32'
    print(f'{tag}: gc={gs}')
    print(f'   gc  : ' + ('no dump' if gc is None else f'max|.|={np.abs(gc).max():.6g} nonzero={int((gc!=0).sum())}'))
    print(f'   out : ' + ('no dump' if e is None else f'max|.|={np.abs(e).max():.6g} nonzero={int((e!=0).sum())}'))
    if gc is not None and e is not None:
        if float(np.abs(gc).max()) > 0 and float(np.abs(e).max()) > 0:
            print('   => the MMA OUTPUT is non-zero with that factor zeroed: the probe did not reach the MMA.')
        elif float(np.abs(gc).max()) == 0 and float(np.abs(e).max()) > 0:
            print('   => the MMA output IS zero but the caller buffer is not: the value appears AFTER the')
            print('      MMA (scatter / stale caller buffer) — a completely different fix.')
        else:
            print('   => both zero: the probe works as advertised.')
PY
