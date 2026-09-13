#!/bin/bash
# Batch 8 (verify path): the ONLY place the D row mapping is observable.
#
# With one live row (the eager/prefill path, m=1) the live D row is row 0 by construction, so a
# wrong per-warp TMEM lane base — and any row-mapping error — cannot change the result. In the
# multi-row verify path (DSpark draft+verify, m=6) the live rows drift across the tile, so this is
# where `DSV41_MOE_BS_LDW` (per-warp lane address, now the default) must make a difference.
#
# The multi-row dump is `[m][topk][act_slot]`; the oracle is fed with `--row` so both sides describe
# the SAME row (the tail one, whose D row index is 5).
set -uo pipefail
cd "$HOME/ferrite"

echo "=== rebuild BOTH artefacts + freshness gate ==="
(cd kernels/cuda && set -o pipefail; bash build.sh 103a 2>&1 | tail -2; echo KERNEL_RC=${PIPESTATUS[0]})
source "$HOME/.cargo/env"
(set -o pipefail; cargo build --release 2>&1 | tail -2; echo CARGO_RC=${PIPESTATUS[0]})
bash "$HOME/check_artifacts.sh" || { echo "ARTIFACTS_STALE — aborting"; exit 1; }
echo "=== tree ==="; git log --oneline -1

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run_fast.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "gateup-dump\|per-warp TMEM lane" "$HOME/armrun_${name}.log" | head -4
}

SPEC="DSV41_SPEC=1 DSV41_DSPARK=1"
rm -rf /tmp/vf_BS /tmp/vf_LOLD
run VF_BS   DSV41_GATEUP_DUMP=/tmp/vf_BS $SPEC
run VF_LOLD DSV41_GATEUP_DUMP=/tmp/vf_LOLD DSV41_MOE_BS_LDW=0 $SPEC

echo "=== which dumps landed? ==="
for d in /tmp/vf_BS /tmp/vf_LOLD; do for t in eager rows; do
  [ -d "$d/$t" ] && echo "$d/$t: $(ls $d/$t | tr '\n' ' ')"
done; done

echo "=== oracle on the verify row (--row 5) and the per-row comparison ==="
python3 - <<'PY'
import os, subprocess, numpy as np
HOME = os.path.expanduser("~")
for arm in ("VF_BS", "VF_LOLD"):
    d = f"/tmp/{arm}"
    if not os.path.isdir(f"{d}/rows"):
        print(f"{arm}: no rows/ dump"); continue
    gu = np.fromfile(f"{d}/rows/gateup.f32", dtype="<f4")
    m = gu.size // (6 * 640)
    print(f"--- {arm}: rows dump has m={m}")
    for row in range(m):
        out = f"/tmp/gu_{arm}_r{row}.f32"
        r = subprocess.run(["python3", f"{HOME}/gu_numpy_ref.py", "--in-dir", f"{d}/rows",
                            "--row", str(row), "--out", out], capture_output=True, text=True)
        if not os.path.exists(out):
            print(f"   row {row}: oracle failed: {r.stdout[-200:]} {r.stderr[-200:]}"); continue
        ref = np.fromfile(out, dtype="<f4").reshape(6, 640)
        got = gu.reshape(m, 6, 640)[row]
        den = np.maximum(np.abs(ref), 1e-6); rel = np.abs(got - ref) / den
        print(f"   row {row}: max|d|={np.abs(got-ref).max():.4g} median rel={np.median(rel):.4g} "
              f"frac>5%={float((rel>0.05).mean()):.4f} corr={np.corrcoef(got.ravel(), ref.ravel())[0,1]:+.5f}")
PY
