#!/bin/bash
# repro_golden.sh — reproduce the 17:09 "golden" run verbatim and compare, answering definitively
# whether the earlier good numbers are reproducible today (i.e. whether any "regression" is real).
#
# The golden was produced by batch4_fix.sh's GD4_OLD arm with ONLY two overrides beyond arm_run.sh's
# COMMON: DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0. arm_run.sh is unversioned but its mtime
# (09-13 15:56) predates the golden (17:09), so the COMMON cannot have drifted.
#
# Two self-inflicted traps fixed here: the tree must be CLEAN first (a killed bisect leaves
# kernels/crates checked out at another commit), and all analysis lives in an uploaded script rather
# than an inline python whose nested quotes keep breaking.
set -uo pipefail
cd "$HOME/ferrite" || exit 1
echo "=== restore a CLEAN origin/main tree first ==="
git checkout -q origin/main -- kernels crates
git status --porcelain kernels crates | head -3
echo "HEAD: $(git log --oneline -1)"

pkill -9 -x ferrite-serve 2>/dev/null
for i in $(seq 1 20); do pgrep -x ferrite-serve >/dev/null || break; sleep 2; done

echo "=== rebuild so the artefacts match this tree ==="
bash "$HOME/ensure_built.sh" || { echo "BUILD GATE FAILED"; exit 1; }

echo "=== the golden run, reproduced verbatim ==="
rm -rf /tmp/gu_repro
bash "$HOME/num100.sh" GD4_REPRO DSV41_GATEUP_DUMP=/tmp/gu_repro \
     DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 >/dev/null 2>&1

python3 "$HOME/scripts/campaign/cmp_golden.py" 2>/dev/null || python3 - <<'PY'
import os, numpy as np
def rd(p, dt='<f4'):
    return np.fromfile(p, dtype=dt) if os.path.exists(p) else None
Gs = [("/tmp/gu_in_GD4_OLD", "GOLDEN 17:09"), ("/tmp/gu_repro/eager", "REPRO now"), ("/tmp/gu_repro", "REPRO now(flat)")]
data = {}
for d, tag in Gs:
    x, ids, gu, w = (rd(os.path.join(d, f)) for f in ("x.f32", "ids.i32", "gateup.f32", "w.f32"))
    print("%-16s x=%s ids=%s gateup=%s" % (tag, "ok" if x is not None else "-",
          ids.tolist() if ids is not None else "-", "ok" if gu is not None else "-"))
    if gu is not None:
        data[tag] = (x, ids, gu, w)
g = data.get("GOLDEN 17:09")
r = data.get("REPRO now") or data.get("REPRO now(flat)")
if not g or not r:
    print("VERDICT: incomplete dumps -- cannot compare")
    raise SystemExit
gx, gids, ggu, gw = g
rx, rids, rgu, rw = r
print()
print("same x     :", None if gx is None or rx is None else bool(np.array_equal(gx, rx)))
print("same ids   :", None if gids is None or rids is None else bool(np.array_equal(gids, rids)))
print("same w     :", None if gw is None or rw is None else bool(np.array_equal(gw, rw)))
den = np.maximum(np.abs(ggu), 1e-6); rel = np.abs(rgu - ggu) / den
print("REPRO vs GOLDEN: max|d|=%.6g med_rel=%.4g corr=%+.5f exact=%s"
      % (np.abs(rgu-ggu).max(), np.median(rel), np.corrcoef(np.nan_to_num(rgu), ggu)[0, 1],
         np.array_equal(rgu, ggu)))
print()
print("READ: exact + same inputs => the 17:09 numbers ARE reproducible today => no regression")
print("      divergence with the SAME inputs => a real code regression in (17:09, HEAD]")
PY
