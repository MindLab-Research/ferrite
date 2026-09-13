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

echo "=== build (hash-gated: skips when the .cu content is unchanged) ==="
bash "$HOME/ensure_built.sh"
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

# Two zero-change controls:
#  * CPASYNC=1 is a DIFFERENT staging implementation (cp.async + double buffering) of the same
#    operands; the file claims bit-exactness with the sequential default. If they differ and the
#    cp.async side is closer to the oracle, the defect is in the sequential staging.
#  * a repeat of the same arm: any difference between the two runs is a race.
rm -rf /tmp/ks_det_a /tmp/ks_det_b /tmp/ks_cpa
run KS_DET_A DSV41_GATEUP_DUMP=/tmp/ks_det_a DSV41_MOE_BS_KEEP_STAGE=17
run KS_DET_B DSV41_GATEUP_DUMP=/tmp/ks_det_b DSV41_MOE_BS_KEEP_STAGE=17
run KS_CPA   DSV41_GATEUP_DUMP=/tmp/ks_cpa   DSV41_MOE_BS_KEEP_STAGE=17 DSV41_MOE_BS_CPASYNC=1
echo "=== determinism: KS_DET_A vs KS_DET_B (must be bit-identical) ==="
for f in gateup.f32; do
  for d in /tmp/ks_det_a /tmp/ks_det_b; do [ -f "$d/eager/$f" ] && cp "$d/eager/$f" "$d/$f" 2>/dev/null; done
  cmp -s /tmp/ks_det_a/$f /tmp/ks_det_b/$f && echo "$f IDENTICAL (deterministic)" || echo "$f DIFFERS (race!)"
done
echo "=== CPASYNC vs sequential (the file claims bit-exact) ==="
for f in gateup.f32; do
  [ -f /tmp/ks_cpa/eager/$f ] && cmp -s /tmp/ks_cpa/eager/$f /tmp/ks_det_a/$f \
    && echo "$f IDENTICAL (claim holds)" || echo "$f DIFFERS — one of the two stagings is wrong"
done

# ---- THE MBARRIER HEALTH PROBE (existing kernel instrumentation, zero code change) ------------
# defect-synthesis's E6: armrun_GD4_BS.log / armrun_GD4_LDOLD.log were killed by the watchdog with
# the "§92 unbounded mbar spin" signature, i.e. the MMA-completion arrival is SOMETIMES MISSING.
# With the unbounded wait (default) that hangs; with the bounded wait the kernel proceeds on an
# INCOMPLETE MMA => the operand smem gets overwritten while the async MMA still reads it => the
# output is a mixture of stages = right magnitude, element-wise uncorrelated, not a permutation,
# and invisible to a one-stage instrument (which commits/wait exactly once at phase 0). The kernel
# already has the probe; this arm just turns it on and reports what it saw.
rm -rf /tmp/ks_wdbg
run KS_WDBG DSV41_GATEUP_DUMP=/tmp/ks_wdbg DSV41_MOE_BS_BOUNDED_WAIT=1 DSV41_MOE_BS_WAITDBG=1
rm -rf /tmp/ks_ring
run KS_RING DSV41_GATEUP_DUMP=/tmp/ks_ring DSV41_MOE_BS_MBAR_RING=1
echo "=== THE FIX TEST: ring arm vs the official-semantics oracle (full K) ==="
for d in /tmp/ks_ring; do
  e=$d; [ -d "$d/eager" ] && e=$d/eager
  if [ -f "$e/gateup.f32" ] && [ -f /tmp/gu_bs_new.f32 ]; then
    python3 "$HOME/gdu_cmp.py" "$e/gateup.f32" /tmp/gu_bs_new.f32 6 640 10.0 2>&1 | head -6
  else
    echo "(missing $e/gateup.f32 or the oracle dump)"
  fi
done

echo "=== mbarrier probe verdict ==="
grep -a "moe-bs-wait" "$HOME/armrun_KS_WDBG.log" | head -12
echo "-- counts --"
printf 'wait-dbg lines        : %s\n' "$(grep -ac 'moe-bs-wait-dbg' "$HOME/armrun_KS_WDBG.log" || true)"
printf 'TIMEOUT lines        : %s\n' "$(grep -ac 'TIMEOUT' "$HOME/armrun_KS_WDBG.log" || true)"
printf 'other_parity_ready=1 : %s\n' "$(grep -ac 'other_parity_ready=1' "$HOME/armrun_KS_WDBG.log" || true)"
printf 'arrived-never (both parities not ready) : %s\n' \
    "$(grep -ac 'other_parity_ready=0' "$HOME/armrun_KS_WDBG.log" || true)"

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
