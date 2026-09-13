#!/bin/bash
# Batch 6 (sharp liveness): three binary probes that pin down WHICH input the MMA actually consumes,
# plus the tagged eager/rows dumps and the CPU reference checks.
#
#   ZA : DSV41_MOE_BS_ZERO_A=1  — the gathered activations are zeroed after the gather. The MMA
#        output must then be EXACTLY zero; anything else means the MMA is reading something other
#        than the staged A operand.
#   ZS : DSV41_MOE_BS_ZERO_SF=1 — the SF scratch is zeroed. With every scale at the e8m0 zero byte
#        the output must be ~0 (2^-127); if it is NOT, the SF is not reaching the MMA at all.
#   ZE : DSV41_MOE_BS_ZERO_EID=1— every segment is told to use expert 0. The output must change
#        materially (the tables are live).
#
# Each arm also writes the tagged dumps: `<dir>/eager/` (the prefill-token-0 call, m=1) and
# `<dir>/rows/` (the first multi-row verify call, m=6 — the ONLY place the D row mapping is
# observable, since with one live row the live D row is row 0 by construction).
set -uo pipefail
cd "$HOME/ferrite"

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "gateup-dump" "$HOME/armrun_${name}.log" | head -4
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

rm -rf /tmp/lv_ZA /tmp/lv_ZS /tmp/lv_ZE /tmp/lv_REF
run LV_REF DSV41_GATEUP_DUMP=/tmp/lv_REF   DSV41_MOE_BS_SFDUMP=/tmp/sfd_LV_REF
run LV_ZA  DSV41_GATEUP_DUMP=/tmp/lv_ZA    DSV41_MOE_BS_ZERO_A=1
run LV_ZS  DSV41_GATEUP_DUMP=/tmp/lv_ZS    DSV41_MOE_BS_ZERO_SF=1
run LV_ZE  DSV41_GATEUP_DUMP=/tmp/lv_ZE    DSV41_MOE_BS_ZERO_EID=1

echo "=== what each dump contains ==="
for d in /tmp/lv_REF /tmp/lv_ZA /tmp/lv_ZS /tmp/lv_ZE; do
  printf '%-12s ' "$d"; ls "$d" 2>/dev/null | tr '\n' ' '; echo
done

echo "=== zero-probe verdicts (BS arm, m=1 eager dump) ==="
python3 - <<'PY'
import os, numpy as np
def load(p):
    return np.fromfile(p, dtype='<f4') if os.path.exists(p) else None
base = load('/tmp/lv_REF/eager/gateup.f32')
print(f'reference (no zero gate): n={0 if base is None else base.size} '
      f'max|.|={0 if base is None else float(np.abs(base).max()):.6g}')
for tag, d in (('ZERO_A', '/tmp/lv_ZA'), ('ZERO_SF', '/tmp/lv_ZS'), ('ZERO_EID', '/tmp/lv_ZE')):
    a = load(f'{d}/eager/gateup.f32')
    if a is None:
        print(f'{tag}: no eager dump'); continue
    print(f'{tag}: n={a.size} max|.|={float(np.abs(a).max()):.6g} nonzero={int((a!=0).sum())} '
          f'-> ' + ('CONSISTENT with the gate being applied' if (
              (tag == 'ZERO_A' and float(np.abs(a).max()) == 0.0) or
              (tag == 'ZERO_SF' and float(np.abs(a).max()) < 1e-6) or
              (tag == 'ZERO_EID' and base is not None and
               float(np.abs(a - base[:a.size]).max()) > 1e-3))
              else '**INCONSISTENT — the MMA is not consuming what we think**'))
PY

echo "=== our official-semantics CPU reference vs the PROVEN path (validates the reference) ==="
if [ -d /tmp/lv_REF/eager ]; then
  timeout 900 python3 "$HOME/gu_numpy_ref.py" --in-dir /tmp/lv_REF/eager --out /tmp/lv_numpy.f32 2>&1 | tail -8
  python3 "$HOME/gdu_cmp.py" /tmp/lv_numpy.f32 /tmp/lv_REF/eager/gateup.f32 6 640 10.0 2>&1 | head -8
else
  echo "(no eager dir — is the binary older than the tagging change?)"
fi

echo "=== staged-operand content check (BS) ==="
if [ -f "$HOME/sfdump_check.py" ]; then
  timeout 600 python3 "$HOME/sfdump_check.py" --in-dir /tmp/sfd_LV_REF 2>&1 | tail -18
else
  echo "(sfdump_check.py missing)"
fi
