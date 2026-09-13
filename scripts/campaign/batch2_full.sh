#!/bin/bash
# Batch 2 (full): rebuild BOTH artefacts from the current tree, produce the EXTENDED dumps
# (gate|up + MoE input x + the quantised e4m3 activation + ids/weights) for the block-scaled arm
# and for the proven per-slot path, then run the OFFICIAL PyTorch gate/up golden on our dumped
# inputs (single GPU; the official implementation is the only external truth) and print the
# three-way verdict: official vs proven per-slot vs block-scaled.
set -uo pipefail
cd "$HOME/ferrite"

echo "=== rebuild kernels (.cu changed: G3) ==="
(cd kernels/cuda && set -o pipefail; bash build.sh 103a 2>&1 | tail -3; echo KERNEL_RC=${PIPESTATUS[0]})
echo "=== rebuild binary ==="
source "$HOME/.cargo/env"
(set -o pipefail; cargo build --release 2>&1 | tail -3; echo CARGO_RC=${PIPESTATUS[0]}); rc=$?
md5sum kernels/cuda/libferrite_kernels.so
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
for f in x.f32 ids.i32 w.f32 xq4.u8 xsc4.f32; do
  a=/tmp/gu_in_bs/$f; b=/tmp/gu_in_old/$f
  if [ -f "$a" ] && [ -f "$b" ]; then
    printf '%-10s ' "$f"; cmp -s "$a" "$b" && echo "IDENTICAL ($(stat -c %s "$b") B)" || echo "DIFFER"
  fi
done

echo "=== A/B (proven per-slot = ground truth) ==="
python3 "$HOME/gdu_cmp.py" /tmp/gu_in_bs/gateup.f32 /tmp/gu_in_old/gateup.f32 6 640 10.0 2>&1 | head -22

echo "=== OFFICIAL PyTorch gate/up on the SAME dumped inputs (main agent, single GPU) ==="
CUDA_VISIBLE_DEVICES=0 timeout 900 python3 /tmp/gu_official.py --in-dir /tmp/gu_in_bs 2>&1 | tail -25
ls -la /tmp/gu_official.f32* 2>/dev/null

echo "=== THREE-WAY (official vs proven vs block-scaled), gate+up per slot ==="
python3 - <<'PY'
import os, numpy as np
act, topk = 640, 6
off_p = '/tmp/gu_official.f32'
off_bf = '/tmp/gu_official.f32.bf16.f32'
off = None
for p in (off_bf, off_p):
    if os.path.exists(p):
        a = np.fromfile(p, dtype='<f4')
        if a.size >= topk * act:
            off = a[:topk * act].reshape(topk, act); print('official source:', p); break
if off is None:
    print('no official dump produced — see the script output above'); raise SystemExit
bs = np.fromfile('/tmp/gu_in_bs/gateup.f32', dtype='<f4').reshape(topk, act)
old = np.fromfile('/tmp/gu_in_old/gateup.f32', dtype='<f4').reshape(topk, act)
lim = 10.0
def cl(x):
    y = x.reshape(topk, 2, act // 2).copy()
    y[:, 0] = np.minimum(y[:, 0], lim); y[:, 1] = np.clip(y[:, 1], -lim, lim)
    return y.reshape(topk, act)
offc, bsc, oldc = cl(off), cl(bs), cl(old)
for tag, a, b in (('official vs PROVEN(per-slot)', oldc, offc), ('official vs BLOCK-SCALED', bsc, offc)):
    den = np.maximum(np.abs(b), 1e-6); rel = np.abs(a - b) / den
    print(f'{tag}: max|d|={np.abs(a-b).max():.4g} median rel={np.median(rel):.4g} '
          f'frac rel>0.05={float((rel>0.05).mean()):.4f} corr={np.corrcoef(a.ravel(),b.ravel())[0,1]:.5f}')
for i in range(topk):
    print(f'slot {i}: |off|={np.linalg.norm(offc[i]):.4g} |old|={np.linalg.norm(oldc[i]):.4g} '
          f'|bs|={np.linalg.norm(bsc[i]):.4g}')
PY
