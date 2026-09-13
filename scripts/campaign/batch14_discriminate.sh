#!/bin/bash
# Batch 14 (discriminate the three survivors, per exact-io-audit):
#   (c) the arm did not actually run at that step (declined) => ex_act_b was written by the OLD
#       per-slot path, which never reads g_a, so "zero A, non-zero output" is trivially true.
#       CHEAPEST TEST: does the ZERO_A step's log show BOTH `[moe-bs] ARMED ..._dev` and
#       `[ZERO-DIAG] mode=2 (A)`? Either missing breaks the evidence chain.
#   (a) the epilogue's TMEM read / uncleared accumulator: with ALL operands zeroed the FIRST MMA
#       (enable_d=0 at k=0,ki=0) must leave an all-zero tile. `STAGE1=1` collapses the loop to that
#       very first stage, so ZERO_ALL+STAGE1 != 0 means the accumulator/TMEM is not cleared.
#   (b) `hw_wait_fail` early-return (its abort path does not write C => the scatter moves the
#       PREVIOUS call's g_c). Only reachable with BOUNDED_WAIT=1, so it is tested head-on.
# Plus the raw-tile probe (needs the .so that exports dsv41_moe_bs_gc_ptr/_bytes, committed since).
set -uo pipefail
cd "$HOME/ferrite"
echo "=== build (hash-gated) ==="
bash "$HOME/ensure_built.sh" || { echo "=== BUILD/ARTEFACT GATE FAILED — aborting before any measurement ==="; exit 1; }
git log --oneline -1
nm -D kernels/cuda/libferrite_kernels.so | grep -c "dsv41_moe_bs_gc_ptr" || true

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -2
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  echo "--- evidence chain: ARMED=$(grep -ac 'ARMED' "$HOME/armrun_${name}.log") ZERO_DIAG=$(grep -ac 'ZERO-DIAG' "$HOME/armrun_${name}.log") GATHER_DIAG=$(grep -ac 'GATHER-DIAG' "$HOME/armrun_${name}.log")"
}

rm -rf /tmp/b14_za /tmp/b14_zs1 /tmp/b14_zb /tmp/sfd_b14
run B14_ZA  DSV41_GATEUP_DUMP=/tmp/b14_za  DSV41_MOE_BS_ZERO_A=1 DSV41_MOE_BS_SFDUMP=/tmp/sfd_b14
run B14_ZAS1 DSV41_GATEUP_DUMP=/tmp/b14_zs1 DSV41_MOE_BS_ZERO_ALL=1 DSV41_MOE_BS_STAGE1=1
run B14_ZB  DSV41_GATEUP_DUMP=/tmp/b14_zb  DSV41_MOE_BS_ZERO_ALL=1 DSV41_MOE_BS_BOUNDED_WAIT=1 DSV41_MOE_BS_WAITDBG=1

echo "=== verdicts ==="
python3 - <<'PY'
import os, numpy as np
def rd(p):
    return np.fromfile(p, dtype='<f4') if os.path.exists(p) else None
for tag, d in (('ZERO_A      ', '/tmp/b14_za'), ('ZERO_ALL+STAGE1', '/tmp/b14_zs1'),
               ('ZERO_ALL+BOUNDED', '/tmp/b14_zb')):
    a = rd(f'{d}/eager/gateup.f32')
    print(f'{tag}: ' + ('no dump' if a is None else
          f'n={a.size} max|.|={np.abs(a).max():.6g} nonzero={int((a!=0).sum())}'))
g = rd('/tmp/sfd_b14/eager/gc_seg0.f32')
print('raw MMA tile (ZERO_A run): ' + ('no dump (accessor missing?)' if g is None else
      f'n={g.size} max|.|={np.abs(g).max():.6g} nonzero={int((g!=0).sum())}'))
print()
print('READ:')
print('  ZERO_A evidence chain OK  <=> its log had ARMED + ZERO-DIAG (see the counts above).')
print('  ZERO_ALL+STAGE1 == 0      => the first MMA clears the tile (accumulator/TMEM fine) => the')
print('                               problem is in the multi-stage accumulation.')
print('  ZERO_ALL+STAGE1 != 0      => even the FIRST MMA leaves a non-zero tile => uncleared')
print('                               accumulator / a TMEM read of a region this call never wrote.')
print('  ZERO_ALL+BOUNDED != 0     => the abort path moved a previous call\'s g_c into ex_act_b.')
PY
echo "=== the mbarrier health lines from the bounded run ==="
grep -a "TIMEOUT\|moe-bs-wait\|abort" "$HOME/armrun_B14_ZB.log" | head -8
