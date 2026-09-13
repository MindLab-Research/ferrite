#!/bin/bash
# Batch 4 (fix verdict): rebuild both artefacts from the current tree, then judge FOUR arms on
# byte-identical inputs against the OFFICIAL PyTorch implementation AND against our own CPU
# formula-level reference (which reads the official checkpoint's true fp4 weights):
#
#   GD4_OLD    proven per-slot path                              (reference)
#   GD4_BS     block-scaled, DEFAULT (per-warp TMEM lane read    <- the headline test: the
#              is now the default spelling, and the SF goes        lane-field fix is confirmed by
#              through the production smem->transpose->cp)         PTX/CUTLASS/Triton
#   GD4_LDOLD  block-scaled + DSV41_MOE_BS_LDW=0                (positive control: the old lane-0
#              spelling must reproduce the mismatch)
#   GD4_SFST   block-scaled + DSV41_MOE_BS_SFST=1               (the other unvalidated link: SF
#              delivered with the instrument's tcgen05.st path)
#
# Every arm also writes the STAGED-operand dump (DSV41_MOE_BS_SFDUMP) for sfdump_check.py, and
# the identity of x/ids across arms is CHECKED, not assumed.
set -uo pipefail
cd "$HOME/ferrite"

echo "=== rebuild kernels (.cu changed: LDW default) ==="
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
  grep -a "per-warp TMEM lane\|SF delivery via\|bs-sfdump\|gateup-dump.*done" "$HOME/armrun_${name}.log" | head -4
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

for n in GD4_OLD GD4_BS GD4_LDOLD GD4_SFST; do rm -rf /tmp/gu_in_$n /tmp/sfd_$n; done
run GD4_OLD   DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_OLD   DSV41_MOE_BS_SFDUMP=/tmp/sfd_GD4_OLD \
              DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0
run GD4_BS    DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_BS    DSV41_MOE_BS_SFDUMP=/tmp/sfd_GD4_BS
run GD4_LDOLD DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_LDOLD DSV41_MOE_BS_LDW=0
run GD4_SFST  DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_SFST  DSV41_MOE_BS_SFST=1

echo "=== inputs identical across arms? (same first MoE call = prefill token 0, layer 0) ==="
for f in x.f32 ids.i32 w.f32 xq4.u8 xsc4.f32; do
  printf '%-10s ' "$f"
  cmp -s "/tmp/gu_in_GD4_BS/$f" "/tmp/gu_in_GD4_OLD/$f" && echo IDENTICAL || echo "DIFFER/missing"
done
echo "=== dumps ==="; ls -la /tmp/gu_in_GD4_*/ 2>/dev/null | head -30
ls -la /tmp/sfd_GD4_*/ 2>/dev/null | head -20

echo "=== A/B: BS (with the lane fix) vs the proven path ==="
python3 "$HOME/gdu_cmp.py" /tmp/gu_in_GD4_BS/gateup.f32 /tmp/gu_in_GD4_OLD/gateup.f32 6 640 10.0 2>&1 | head -14
echo "=== A/B: BS with the OLD lane spelling (positive control) ==="
python3 "$HOME/gdu_cmp.py" /tmp/gu_in_GD4_LDOLD/gateup.f32 /tmp/gu_in_GD4_OLD/gateup.f32 6 640 10.0 2>&1 | head -6
echo "=== A/B: BS with the instrument SF path ==="
python3 "$HOME/gdu_cmp.py" /tmp/gu_in_GD4_SFST/gateup.f32 /tmp/gu_in_GD4_OLD/gateup.f32 6 640 10.0 2>&1 | head -6

echo "=== OFFICIAL PyTorch gate/up on the SAME dumped inputs ==="
CUDA_VISIBLE_DEVICES=0 timeout 900 python3 /tmp/gu_official.py --in-dir /tmp/gu_in_GD4_BS 2>&1 | tail -12

echo "=== our CPU formula-level reference on the SAME dump (reads the official ckpt's true fp4) ==="
timeout 900 python3 "$HOME/gu_numpy_ref.py" --in-dir /tmp/gu_in_GD4_BS --out /tmp/gu_numpy.f32 2>&1 | tail -10

echo "=== staged-operand content check ==="
if [ -f "$HOME/sfdump_check.py" ]; then
  timeout 600 python3 "$HOME/sfdump_check.py" --in-dir /tmp/sfd_GD4_BS 2>&1 | tail -22
else
  echo "(sfdump_check.py missing)"
fi

echo "=== VERDICT MATRIX (every arm vs every available reference) ==="
python3 - <<'PY'
import os, numpy as np
act, topk, lim = 640, 6, 10.0
def load(p):
    if os.path.exists(p) and os.path.getsize(p) >= topk*act*4:
        return np.fromfile(p, dtype='<f4')[:topk*act].reshape(topk, act)
    return None
refs = {}
for tag, p in (('official', '/tmp/gu_official.f32.bf16.f32'),
               ('official', '/tmp/gu_official.f32'),
               ('numpy', '/tmp/gu_numpy.f32'),
               ('proven(OLD)', '/tmp/gu_in_GD4_OLD/gateup.f32')):
    if tag in refs: continue
    a = load(p)
    if a is not None: refs[tag] = a
def cl(x):
    y = x.reshape(topk, 2, act//2).copy()
    y[:,0] = np.minimum(y[:,0], lim); y[:,1] = np.clip(y[:,1], -lim, lim)
    return y.reshape(topk, act)
print('references available:', list(refs))
for arm in ('GD4_OLD','GD4_BS','GD4_LDOLD','GD4_SFST'):
    a = load(f'/tmp/gu_in_{arm}/gateup.f32')
    if a is None: print(f'{arm}: no dump'); continue
    ac = cl(a)
    print(f'--- {arm} (norm {np.linalg.norm(ac):.4g})')
    for tag, r in refs.items():
        if arm == 'GD4_OLD' and tag == 'proven(OLD)': continue
        rc = cl(r); den = np.maximum(np.abs(rc),1e-6); rel = np.abs(ac-rc)/den
        print(f'    vs {tag:12} max|d|={np.abs(ac-rc).max():>9.4g} med rel={np.median(rel):>9.4g} '
              f'frac>5%={float((rel>0.05).mean()):>7.4f} corr={np.corrcoef(ac.ravel(),rc.ravel())[0,1]:>8.5f}')
PY
