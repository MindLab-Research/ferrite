#!/bin/bash
# Batch 9 (per-stage advance check): the sharpest remaining test.
#
# `DSV41_MOE_BS_KEEP_STAGE=s` zeroes every A byte outside K in [128s, 128s+128), so the arm's
# output IS stage s's partial. The oracle then computes the partial for EVERY stage's K range, and
# matching the arm against all 40 of them answers the only question left in this class:
#
#   arm(s) == oracle(stage s)          => the four per-stage advances (A k*128, W k*64 bytes,
#                                         SFA k*4608, SFW k*320) are mutually consistent for s;
#   arm(s) == oracle(stage s') != s    => the A data advance and the SCALE advance disagree by
#                                         (s'-s) stages — i.e. exactly the "right magnitude,
#                                         element-wise uncorrelated, not a permutation" mechanism,
#                                         now localised to a concrete pair of index expressions.
set -uo pipefail
cd "$HOME/ferrite"

echo "=== rebuild BOTH artefacts + freshness gate ==="
(cd kernels/cuda && set -o pipefail; bash build.sh 103a 2>&1 | tail -2; echo KERNEL_RC=${PIPESTATUS[0]})
source "$HOME/.cargo/env"
(set -o pipefail; cargo build --release 2>&1 | tail -2; echo CARGO_RC=${PIPESTATUS[0]})
bash "$HOME/check_artifacts.sh" || { echo "ARTIFACTS_STALE — aborting"; exit 1; }
git log --oneline -1

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -2
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "keep-only K-stage" "$HOME/armrun_${name}.log" | head -1
}

# A few stages spread over the loop, plus one SFDUMP snapshot at a mid stage (so the content check
# can finally see the W/SF where the advances act).
for S in 0 1 17 39; do
  rm -rf /tmp/ks$S
  if [ "$S" = 17 ]; then
    run KS$S DSV41_GATEUP_DUMP=/tmp/ks$S DSV41_MOE_BS_KEEP_STAGE=$S \
            DSV41_MOE_BS_SFDUMP=/tmp/sfd_ks$S DSV41_MOE_BS_SFDUMP_K=17
  else
    run KS$S DSV41_GATEUP_DUMP=/tmp/ks$S DSV41_MOE_BS_KEEP_STAGE=$S
  fi
done

echo "=== which stage's partial does each arm actually equal? ==="
python3 - <<'PY'
import os, subprocess, numpy as np
HOME = os.path.expanduser("~")
for S in (0, 1, 17, 39):
    d = f"/tmp/ks{S}"
    sub = f"{d}/eager" if os.path.isdir(f"{d}/eager") else d
    p = f"{sub}/gateup.f32"
    if not os.path.exists(p):
        print(f"stage {S}: no dump"); continue
    got = np.fromfile(p, dtype="<f4").reshape(6, 640)
    # per-slot correlation against the oracle partial of EVERY stage (using the first slot's ids)
    best = None
    table = []
    for s2 in range(40):
        out = f"/tmp/gu_s{s2}.f32"
        if not os.path.exists(out):
            r = subprocess.run(["python3", f"{HOME}/gu_numpy_ref.py", "--in-dir", sub,
                                "--kmin", str(128 * s2), "--kmax", str(128 * (s2 + 1)),
                                "--out", out], capture_output=True, text=True)
            if not os.path.exists(out):
                print(f"  oracle stage {s2} failed: {r.stdout[-160:]} {r.stderr[-160:]}"); break
        ref = np.fromfile(out, dtype="<f4").reshape(6, 640)
        c = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
        table.append((s2, c, float(np.linalg.norm(got) / max(np.linalg.norm(ref), 1e-9))))
    table.sort(key=lambda t: -abs(t[1]))
    print(f"stage {S}: top-3 oracle-stage matches (stage, corr, norm_ratio): "
          + ", ".join(f"({s2},{c:+.4f},{nr:.3f})" for s2, c, nr in table[:3]))
PY
