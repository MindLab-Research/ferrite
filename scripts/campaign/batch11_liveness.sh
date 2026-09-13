#!/bin/bash
# Batch 11 (output-liveness): if EVERY operand the MMA consumes is zeroed, the output MUST be
# exactly zero. Anything else means the value in `ex_act_b` was not produced by this call's
# computation at all — i.e. it came from a stale buffer somewhere along
# TMEM D -> kernel C write -> scatter -> ex_act_b.
#
# `DSV41_MOE_BS_ZERO_ALL=1` is mode 4 in the shim: it memsets the gathered A, the SF scratch AND
# the eid table before the MMA, so the MMA's operands are all zero.
set -uo pipefail
cd "$HOME/ferrite"

echo "=== build (hash-gated: skips when the .cu content is unchanged) ==="
bash "$HOME/ensure_built.sh" || { echo "=== BUILD/ARTEFACT GATE FAILED — aborting before any measurement ==="; exit 1; }
git log --oneline -1

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -2
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "ZERO-DIAG\|gateup-dump.*done\|GATHER-DIAG" "$HOME/armrun_${name}.log" | head -3
}

for n in ALL SF EID ZA2; do rm -rf /tmp/zl_$n; done
run ZL_ALL DSV41_GATEUP_DUMP=/tmp/zl_ALL DSV41_MOE_BS_ZERO_ALL=1
run ZL_SF  DSV41_GATEUP_DUMP=/tmp/zl_SF  DSV41_MOE_BS_ZERO_SF=1
run ZL_EID DSV41_GATEUP_DUMP=/tmp/zl_EID DSV41_MOE_BS_ZERO_EID=1
# A second ZERO_A sample: the killed batch6 already left one (/tmp/lv_ZA). If the two differ, the
# non-zero output under a zeroed operand is not merely "wrong" but STALE/RACY — which is the whole
# question. If they agree, it is deterministic but unrelated to the staged A.
run ZL_ZA2 DSV41_GATEUP_DUMP=/tmp/zl_ZA2 DSV41_MOE_BS_ZERO_A=1

echo "=== ZERO_A across two builds (same config) ==="
for p in /tmp/lv_ZA/eager/gateup.f32 /tmp/zl_ZA2/eager/gateup.f32; do
  [ -f "$p" ] && python3 -c "
import numpy as np,sys
a=np.fromfile('$p',dtype='<f4')
print('$p', 'n=%d max|.|=%.6g nonzero=%d'%(a.size,np.abs(a).max(),int((a!=0).sum())))
"
done
if [ -f /tmp/lv_ZA/eager/gateup.f32 ] && [ -f /tmp/zl_ZA2/eager/gateup.f32 ]; then
  cmp -s /tmp/lv_ZA/eager/gateup.f32 /tmp/zl_ZA2/eager/gateup.f32 \
    && echo "ZERO_A dumps IDENTICAL across builds (deterministic but unrelated to the staged A)" \
    || echo "ZERO_A dumps DIFFER across builds — the zeroed-operand output is STALE/RACY"
fi

echo "=== liveness verdicts ==="
python3 - <<'PY'
import os, numpy as np
ref = None
p = '/tmp/lv_REF/eager/gateup.f32'
if os.path.exists(p):
    ref = np.fromfile(p, dtype='<f4')
    print(f"reference (no zeroing): n={ref.size} max|.|={np.abs(ref).max():.6g}")
for tag, d in (('ZERO_ALL (A+SF+EID)', '/tmp/zl_ALL'), ('ZERO_SF', '/tmp/zl_SF'), ('ZERO_EID', '/tmp/zl_EID')):
    q = f'{d}/eager/gateup.f32'
    if not os.path.exists(q):
        print(f'{tag}: no dump'); continue
    a = np.fromfile(q, dtype='<f4')
    mx = float(np.abs(a).max()); nz = int((a != 0).sum())
    print(f'{tag}: n={a.size} max|.|={mx:.6g} nonzero={nz}')
    if tag.startswith('ZERO_ALL'):
        print('   VERDICT: ' + ('the output is produced by THIS call\\'s computation ✓'
                                if mx == 0.0 else
                                '**OUTPUT IS NOT FROM THIS CALL** — with every MMA operand zeroed the '
                                'result must be 0, so the value in ex_act_b is stale (somewhere along '
                                'TMEM D -> kernel C -> scatter)'))
PY
