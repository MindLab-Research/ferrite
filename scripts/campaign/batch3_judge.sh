#!/bin/bash
# Batch 3 (judge): rebuild both artefacts, then produce a gate|up dump for FIVE configurations
# and judge EVERY one of them against the OFFICIAL PyTorch implementation (the only external
# truth) on byte-identical inputs (same prompt => the first `moe()` call is layer 0).
#
#   GD3_OLD      proven per-slot path            (the current working reference)
#   GD3_BS       block-scaled, default staging
#   GD3_SFREV    block-scaled + DSV41_MOE_BS_SFREV=1     (SF byte order within the word)
#   GD3_CPASYNC  block-scaled + DSV41_MOE_BS_CPASYNC=1   (double-buffered cp.async staging)
#   GD3_CANON    block-scaled + DSV41_MOE_BS_CANON=1     (repo canonical smem layout)
#
# The extended dump also carries x / ids / w / xq4 / xsc4, which (a) must be IDENTICAL across all
# arms and (b) is exactly the input the official golden script consumes.
set -uo pipefail
cd "$HOME/ferrite"

echo "=== rebuild kernels (.cu changed) ==="
(cd kernels/cuda && set -o pipefail; bash build.sh 103a 2>&1 | tail -3; echo KERNEL_RC=${PIPESTATUS[0]})
echo "=== rebuild binary ==="
source "$HOME/.cargo/env"
set -o pipefail; cargo build --release 2>&1 | tail -3; echo CARGO_RC=$?
md5sum kernels/cuda/libferrite_kernels.so
stat -c "%y %n" kernels/cuda/libferrite_kernels.so target/release/ferrite-serve

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "gateup-dump" "$HOME/armrun_${name}.log" | tail -2
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

for n in GD3_OLD GD3_BS GD3_SFREV GD3_CPASYNC; do rm -rf /tmp/gu_in_$n; done
run GD3_OLD     DSV41_GATEUP_DUMP=/tmp/gu_in_GD3_OLD DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0
run GD3_BS      DSV41_GATEUP_DUMP=/tmp/gu_in_GD3_BS
run GD3_SFREV   DSV41_GATEUP_DUMP=/tmp/gu_in_GD3_SFREV   DSV41_MOE_BS_SFREV=1
run GD3_CPASYNC DSV41_GATEUP_DUMP=/tmp/gu_in_GD3_CPASYNC DSV41_MOE_BS_CPASYNC=1

echo "=== inputs identical across arms? ==="
for f in x.f32 ids.i32 w.f32 xq4.u8 xsc4.f32; do
  printf '%-10s ' "$f"; cmp -s /tmp/gu_in_GD3_BS/$f /tmp/gu_in_GD3_OLD/$f && echo IDENTICAL || echo DIFFER
done
echo "=== dump sizes ==="; ls -la /tmp/gu_in_GD3_*/gateup.f32 2>/dev/null

echo "=== OFFICIAL PyTorch gate/up on the SAME dumped inputs (single GPU) ==="
CUDA_VISIBLE_DEVICES=0 timeout 900 python3 /tmp/gu_official.py --in-dir /tmp/gu_in_GD3_BS 2>&1 | tail -20

echo "=== VERDICT MATRIX: every arm judged against the official implementation ==="
python3 - <<'PY'
import os, numpy as np
act, topk, lim = 640, 6, 10.0
cands = []
for p in ('/tmp/gu_official.f32.bf16.f32', '/tmp/gu_official.f32'):
    if os.path.exists(p) and os.path.getsize(p) >= topk * act * 4:
        cands.append(('official', p, np.fromfile(p, dtype='<f4')[:topk * act].reshape(topk, act)))
        break
if not cands:
    print('no official dump — cannot judge'); raise SystemExit
ref_name, ref_path, ref = cands[0]
print(f'REFERENCE = {ref_name}  ({ref_path})')

def cl(x):
    y = x.reshape(topk, 2, act // 2).copy()
    y[:, 0] = np.minimum(y[:, 0], lim); y[:, 1] = np.clip(y[:, 1], -lim, lim)
    return y.reshape(topk, act)

refc = cl(ref)
rows = []
for arm in ('GD3_OLD', 'GD3_BS', 'GD3_SFREV', 'GD3_CPASYNC'):
    p = f'/tmp/gu_in_{arm}/gateup.f32'
    if not os.path.exists(p):
        rows.append((arm, None)); continue
    a = np.fromfile(p, dtype='<f4')
    if a.size < topk * act:
        rows.append((arm, None)); continue
    a = cl(a[:topk * act].reshape(topk, act))
    den = np.maximum(np.abs(refc), 1e-6); rel = np.abs(a - refc) / den
    rows.append((arm, (float(np.abs(a - refc).max()), float(np.median(rel)),
                       float((rel > 0.05).mean()), float(np.corrcoef(a.ravel(), refc.ravel())[0, 1]),
                       float(np.linalg.norm(a) / max(np.linalg.norm(refc), 1e-9)))))
print(f"{'arm':14} {'max|d|':>10} {'med rel':>10} {'frac>5%':>9} {'corr':>9} {'||arm||/||off||':>16}")
for arm, r in rows:
    if r is None:
        print(f'{arm:14} {"(no dump)":>10}'); continue
    print(f'{arm:14} {r[0]:>10.4g} {r[1]:>10.4g} {r[2]:>9.4f} {r[3]:>9.5f} {r[4]:>16.4f}')
print()
print('READ: the arm whose (med rel, frac>5%) is near zero and corr near 1 is the correct one;')
print('      the proven per-slot path is expected to sit there, the block-scaled arm is not.')
for i in range(topk):
    print(f'  slot {i}: ||official||={np.linalg.norm(refc[i]):.4g}')
PY
