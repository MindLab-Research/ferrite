#!/bin/bash
# golden_endpoint.sh — settle "illusion vs regression" with two clean, same-source reproductions.
#
# The cluster evidence (pairwise over every 3840-float gate|up artefact in /tmp):
#   cluster A (the oracle AND the 17:09 "golden"): gu_in_GD4_OLD, gu_bs_new, gu_numpy_old,
#                                                 gu_numpy_GD4OLD, gu_bs_row0, gu_km_range, oracle_chk/oracle.f32
#   cluster B: gu_old.f32 (16:45) and oracle_chk/eager/gateup.f32 (today's own dump)
# A != B, and the golden is the ONLY artefact from before 17:09 that sits in A. A monotone regression
# cannot explain an artefact that is "right" at 17:09 while both an older (16:45) and a newer (today)
# tree are "wrong" -- that pattern points at a transient tree at 17:09 (the campaign was actively
# editing files then), i.e. at the user's "was it an illusion?" hypothesis.
#
# Two verdicts, both with the golden's OWN config (its arm log records it verbatim:
#   DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 + arm_run.sh COMMON, which is unversioned and
#   unchanged since 15:56):
#   1. HEAD, clean + same-source rebuild
#   2. ac69b054, the newest commit before the golden's mtime (17:07:29 < 17:09:22)
# After any `git checkout <rev> -- kernels crates` the rebuild is FORCED (the stamp only tracks
# sources, and that exact lapse produced CARGO_RC=101 earlier).
set -uo pipefail
cd "$HOME/ferrite" || exit 1
CMP=/tmp/golden_cmp.py
cat > "$CMP" <<'PY'
import os, sys, numpy as np
def rd(p, dt='<f4'):
    return np.fromfile(p, dtype=dt) if os.path.exists(p) else None
gold = rd("/tmp/gu_in_GD4_OLD/gateup.f32")
gx   = rd("/tmp/gu_in_GD4_OLD/x.f32")
gids = rd("/tmp/gu_in_GD4_OLD/ids.i32")
new = rd("/tmp/goldchk/eager/gateup.f32") or rd("/tmp/goldchk/gateup.f32")
nx   = rd("/tmp/goldchk/eager/x.f32")   or rd("/tmp/goldchk/x.f32")
nids = rd("/tmp/goldchk/eager/ids.i32") or rd("/tmp/goldchk/ids.i32")
tag = sys.argv[1] if len(sys.argv) > 1 else "?"
if gold is None or new is None:
    print("  %s: dump missing (gold=%s new=%s)" % (tag, gold is not None, new is not None)); raise SystemExit
samex = (gx is not None and nx is not None and np.array_equal(gx, nx))
sameids = (gids is not None and nids is not None and np.array_equal(gids, nids))
den = np.maximum(np.abs(gold), 1e-6); rel = np.abs(new-gold)/den
exact = np.array_equal(new, gold)
print("  %s: inputs identical gold? x=%s ids=%s | vs GOLDEN max|d|=%.6g med_rel=%.4g corr=%+.5f exact=%s"
      % (tag, samex, sameids, np.abs(new-gold).max(), np.median(rel),
         np.corrcoef(np.nan_to_num(new), gold)[0,1], exact))
print("       cluster:", "A (matches the oracle/golden)" if exact else
      ("B (matches the 16:45 artefact)" if np.isfinite(np.corrcoef(np.nan_to_num(new), gold)[0,1]) and
       np.corrcoef(np.nan_to_num(new), gold)[0,1] < 0.5 else "?"))
PY

run_arm () {
  rm -rf /tmp/goldchk
  bash "$HOME/num100.sh" "$1" DSV41_GATEUP_DUMP=/tmp/goldchk \
       DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 >/dev/null 2>&1
  python3 "$CMP" "$1"
}

echo "################ 1. HEAD, clean + same-source ################"
git checkout -q origin/main -- kernels crates
git log --oneline -1
bash "$HOME/ensure_built.sh" || { echo "BUILD GATE FAILED (HEAD)"; exit 1; }
run_arm GEP_HEAD

echo "################ 2. ac69b054 (the golden's commit) ################"
git checkout -q ac69b054 -- kernels crates
echo "kernels/crates now at: $(git log -1 --format='%h %ad %s' --date=format:%H:%M ac69b054 | cut -c1-90)"
echo "--- FORCED rebuild (the stamp cannot be trusted after a partial checkout) ---"
(cd kernels/cuda && bash build.sh 103a 2>&1 | tail -1) || { echo "BUILD_SH FAILED at ac69b054"; }
(source "$HOME/.cargo/env" && cargo build --release 2>&1 | tail -1) || { echo "CARGO FAILED at ac69b054"; }
run_arm GEP_AC69

echo "################ restore origin/main + same-source rebuild ################"
git checkout -q origin/main -- kernels crates
bash "$HOME/ensure_built.sh" || echo "RESTORE BUILD FAILED"
echo "=== golden_endpoint DONE ==="
