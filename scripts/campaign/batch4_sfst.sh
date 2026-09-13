#!/bin/bash
# Batch 4: rebuild both artefacts, then run the SF-delivery experiment plus its two controls, all
# with the EXTENDED dump, and judge every arm against the OFFICIAL PyTorch implementation.
#
#   GD4_OLD   proven per-slot path            (control: must sit on top of the official)
#   GD4_BS    block-scaled, production staging (control: the arm under investigation)
#   GD4_SFST  block-scaled + DSV41_MOE_BS_SFST=1 (SF delivered with tcgen05.st, the layout the
#             isolated instrument PASSED with — the production smem->transpose->cp chain has
#             never been validated on its own)
set -uo pipefail
cd "$HOME/ferrite"

echo "=== rebuild kernels (.cu changed: SFST) ==="
(cd kernels/cuda && set -o pipefail; bash build.sh 103a 2>&1 | tail -3; echo KERNEL_RC=${PIPESTATUS[0]})
echo "=== rebuild binary ==="
source "$HOME/.cargo/env"
(set -o pipefail; cargo build --release 2>&1 | tail -3; echo CARGO_RC=${PIPESTATUS[0]})
md5sum kernels/cuda/libferrite_kernels.so
stat -c "%y %n" kernels/cuda/libferrite_kernels.so target/release/ferrite-serve

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "SF delivery via\|gateup-dump.*done" "$HOME/armrun_${name}.log" | head -2
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

for n in GD4_OLD GD4_BS GD4_SFST GD4_LDW GD4_BOTH; do rm -rf /tmp/gu_in_$n; done
rm -rf /tmp/sfd_GD4_OLD /tmp/sfd_GD4_BS
# The BS arm also collects the STAGED-operand dump in the same run (same one-shot latch), so one
# batch yields both the official verdict AND the material the content checker needs.
run GD4_OLD  DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_OLD  DSV41_MOE_BS_SFDUMP=/tmp/sfd_GD4_OLD \
             DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0
run GD4_BS   DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_BS   DSV41_MOE_BS_SFDUMP=/tmp/sfd_GD4_BS
run GD4_SFST DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_SFST DSV41_MOE_BS_SFST=1
run GD4_LDW  DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_LDW  DSV41_MOE_BS_LDW=1
run GD4_BOTH DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_BOTH DSV41_MOE_BS_SFST=1 DSV41_MOE_BS_LDW=1

echo "=== inputs identical across arms? (same first MoE call = prefill token 0, layer 0) ==="
for f in x.f32 ids.i32 w.f32 xq4.u8 xsc4.f32; do
  printf '%-10s ' "$f"; cmp -s /tmp/gu_in_GD4_BS/$f /tmp/gu_in_GD4_OLD/$f && echo IDENTICAL || echo DIFFER
done
echo "=== dumps ==="; ls -la /tmp/gu_in_GD4_*/gateup.f32 2>/dev/null

echo "=== OFFICIAL PyTorch gate/up on the SAME dumped inputs (single GPU) ==="
CUDA_VISIBLE_DEVICES=0 timeout 900 python3 /tmp/gu_official.py --in-dir /tmp/gu_in_GD4_BS 2>&1 | tail -18

echo "=== CPU formula-level reference (reads the official ckpt's true fp4) ==="
timeout 900 python3 "$HOME/gu_numpy_ref.py" --in-dir /tmp/gu_in_GD4_BS --out /tmp/gu_numpy.f32 2>&1 | tail -12

echo "=== VERDICT MATRIX vs the official implementation ==="
python3 - <<'PY'
import os, numpy as np
act, topk, lim = 640, 6, 10.0
ref = None
for p in ('/tmp/gu_official.f32.bf16.f32', '/tmp/gu_official.f32', '/tmp/gu_numpy.f32',
          '/tmp/gu_numpy.f32.bf16.f32'):
    if os.path.exists(p) and os.path.getsize(p) >= topk * act * 4:
        ref = (p, np.fromfile(p, dtype='<f4')[:topk * act].reshape(topk, act)); break
if ref is None:
    print('no reference dump produced'); raise SystemExit
print('REFERENCE =', ref[0])
def cl(x):
    y = x.reshape(topk, 2, act // 2).copy()
    y[:, 0] = np.minimum(y[:, 0], lim); y[:, 1] = np.clip(y[:, 1], -lim, lim)
    return y.reshape(topk, act)
rc = cl(ref[1])
print(f"{'arm':14} {'max|d|':>10} {'med rel':>10} {'frac>5%':>9} {'corr':>9} {'nrm ratio':>10}")
for arm in ('GD4_OLD', 'GD4_BS', 'GD4_SFST', 'GD4_LDW', 'GD4_BOTH'):
    p = f'/tmp/gu_in_{arm}/gateup.f32'
    if not os.path.exists(p):
        print(f'{arm:14} (no dump)'); continue
    a = cl(np.fromfile(p, dtype='<f4')[:topk * act].reshape(topk, act))
    den = np.maximum(np.abs(rc), 1e-6); rel = np.abs(a - rc) / den
    print(f'{arm:14} {np.abs(a-rc).max():>10.4g} {np.median(rel):>10.4g} '
          f'{float((rel>0.05).mean()):>9.4f} {np.corrcoef(a.ravel(),rc.ravel())[0,1]:>9.5f} '
          f'{np.linalg.norm(a)/max(np.linalg.norm(rc),1e-9):>10.4f}')
PY
