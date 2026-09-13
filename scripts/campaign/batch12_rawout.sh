#!/bin/bash
# Batch 12 (raw-output probe): read the MMA's RAW output tile (`g_c`, before the scatter) from Rust
# and answer the one question the ZERO_A anomaly opened.
#
#   * with EVERY operand zeroed (ZERO_ALL: A + SF + eid), the tile MUST be exactly zero. If it is
#     not, the value never came from this call's computation.
#   * with the tile zeroed-check passing, a non-zero `ex_act_b` means the SCATTER (or a stale
#     upstream buffer) is the source — a completely different fix.
#
# Two arms only, and a repeat to see whether the raw tile is even deterministic.
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
  grep -a "bs-gc\|bs-sfdump\|ZERO-DIAG" "$HOME/armrun_${name}.log" | head -4
}

rm -rf /tmp/gc_all /tmp/gc_all2 /tmp/gc_norm
run GC_ALL  DSV41_MOE_BS_SFDUMP=/tmp/gc_all  DSV41_MOE_BS_ZERO_ALL=1 DSV41_GATEUP_DUMP=/tmp/gu_gc_all
run GC_ALL2 DSV41_MOE_BS_SFDUMP=/tmp/gc_all2 DSV41_MOE_BS_ZERO_ALL=1 DSV41_GATEUP_DUMP=/tmp/gu_gc_all2
run GC_NORM DSV41_MOE_BS_SFDUMP=/tmp/gc_norm DSV41_GATEUP_DUMP=/tmp/gu_gc_norm

echo "=== the verdict on the RAW MMA output ==="
python3 - <<'PY'
import os, numpy as np
def rd(p):
    return np.fromfile(p, dtype='<f4') if os.path.exists(p) else None
for tag, d in (('ZERO_ALL run 1', '/tmp/gc_all'), ('ZERO_ALL run 2', '/tmp/gc_all2'),
               ('normal (no zeroing)', '/tmp/gc_norm')):
    gc = rd(f'{d}/eager/gc_seg0.f32')
    if gc is None:
        print(f'{tag}: no gc dump ({d})'); continue
    print(f'{tag}: gc n={gc.size} max|.|={np.abs(gc).max():.6g} nonzero={int((gc!=0).sum())}')
    if 'ZERO_ALL' in tag:
        print('   -> ' + ('the MMA output IS zero with zeroed operands (the pipeline is faithful)'
                          if gc.size and float(np.abs(gc).max()) == 0.0 else
                          '**the MMA output is NOT zero with zeroed operands** — the value does not '
                          'come from this call\\'s operands'))
a, b = rd('/tmp/gc_all/eager/gc_seg0.f32'), rd('/tmp/gc_all2/eager/gc_seg0.f32')
if a is not None and b is not None:
    print('raw tile determinism (ZERO_ALL, two runs):',
          'IDENTICAL' if np.array_equal(a, b) else f'DIFFERS (max|d|={np.abs(a-b).max():.6g})')
print()
print('=== and the post-scatter buffer for the same runs (ex_act_b) ===')
for tag, p in (('ZERO_ALL r1', '/tmp/gu_gc_all/eager/gateup.f32'),
               ('ZERO_ALL r2', '/tmp/gu_gc_all2/eager/gateup.f32'),
               ('normal', '/tmp/gu_gc_norm/eager/gateup.f32')):
    e = rd(p)
    print(f'{tag}: ' + ('no dump' if e is None else
          f'n={e.size} max|.|={np.abs(e).max():.6g} nonzero={int((e!=0).sum())}'))
print()
print('READ: gc==0 && ex_act_b!=0  => the SCATTER/stale buffer is the source.')
print('      gc!=0 (with all operands zeroed) => the MMA/epilogue is the source.')
PY
