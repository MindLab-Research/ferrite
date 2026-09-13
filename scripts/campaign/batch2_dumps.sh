#!/bin/bash
# Batch 2: rebuild both artefacts from the current tree, then produce the EXTENDED dumps
# (gate|up + the MoE input x + the quantised e4m3 activation + ids/weights) for the BS arm and
# for the proven per-slot path. Those dumps are the input side of the three-way comparison
# (official PyTorch / proven per-slot / block-scaled) that the main agent runs next.
set -uo pipefail
cd "$HOME/ferrite"

echo "=== .so + binary: same tree? ==="
md5sum kernels/cuda/libferrite_kernels.so target/release/ferrite-serve 2>/dev/null | head -2
stat -c "%y %n" kernels/cuda/libferrite_kernels.so target/release/ferrite-serve

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "gateup-dump" "$HOME/armrun_${name}.log" | head -8
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

rm -rf /tmp/gu_in_bs /tmp/gu_in_old
run GD2_BS  DSV41_GATEUP_DUMP=/tmp/gu_in_bs
run GD2_OLD DSV41_GATEUP_DUMP=/tmp/gu_in_old DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0

echo "=== dumps ==="
ls -la /tmp/gu_in_bs /tmp/gu_in_old 2>/dev/null
echo "=== A/B (proven per-slot = ground truth) ==="
python3 "$HOME/gdu_cmp.py" /tmp/gu_in_bs/gateup.f32 /tmp/gu_in_old/gateup.f32 6 640 || true
echo "=== x / ids / weights (must be IDENTICAL in both runs: same layer, same step) ==="
for f in x.f32 ids.i32 w.f32 xq4.u8; do
  a=/tmp/gu_in_bs/$f; b=/tmp/gu_in_old/$f
  if [ -f "$a" ] && [ -f "$b" ]; then
    printf '%-10s ' "$f"; cmp -s "$a" "$b" && echo "IDENTICAL ($(stat -c %s "$b") B)" || echo "DIFFER $(stat -c %s "$a") vs $(stat -c %s "$b") B"
  fi
done
echo "=== first values of each dump (for the official-torch side) ==="
python3 - <<'PY'
import numpy as np, os
for d in ('/tmp/gu_in_bs', '/tmp/gu_in_old'):
    print('##', d)
    for f, dt in (('x.f32', '<f4'), ('ids.i32', '<i4'), ('w.f32', '<f4'), ('xq4.u8', 'u1'), ('xsc4.f32', '<f4')):
        p = os.path.join(d, f)
        if os.path.exists(p):
            a = np.fromfile(p, dtype=dt)
            print(f'  {f}: n={a.size} first8={a[:8]}')
PY
