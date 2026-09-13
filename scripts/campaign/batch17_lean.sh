#!/bin/bash
# Batch 17 — LEAN. Every arm must answer a question that no existing data answers; no "pass 2" of an
# arm whose verdict is already known (user rule, 2026-09-14: re-running a determined arm is waste).
#
# Already determined, so NOT re-run here:
#   * MBAR_PERSTAGE alone  => hangs (batch15: every wait spins to the cap, step 153 ms vs 13 ms, log
#     frozen, watchdog kill). Any arm containing it is skipped.
#   * ZERO_A / ZERO_SF / ZERO_ALL => their single-factor readings contradict the product model, so
#     they are discredited as probes (mode 5 replaces them).
#   * The baseline's wrongness (median rel 1.19, corr 0.04 vs the official oracle) => established.
#
# Open questions, one per arm:
#   F7_RING   : does an arrival land on a barrier that is NOT slot 0? (ring uses slots 0 and 1)
#               ring works  => multi-slot arrivals are fine => my per-stage init/loop logic is wrong
#               ring hangs  => anything past slot 0 fails => the commit's addressing/instruction form
#   F8_DRAIN  : same question at slot 40 (the far end), with the BOUNDED wait so a failure reports
#               instead of hanging: a timeout is itself the answer.
#   F6_ASF    : the clean zero-product test — A and SF zeroed, expert ids intact, so the tables stay
#               valid and the scatter runs. A zero product MUST give an exactly-zero output; anything
#               else means the MMA reads operands other than the ones we staged.
#   F6_ASFDC  : the same with an explicit D clear; if bare is non-zero and cleared is exactly zero, the
#               accumulator/TMEM path is the source.
#   F1_DC2    : the qNaN sentinel — nan==0 means no column went unwritten; nan>0 makes the "read a
#               column this launch never wrote" mechanism loud instead of silent.
set -uo pipefail
cd "$HOME/ferrite"
echo "=== build (hash-gated) ==="
bash "$HOME/ensure_built.sh" || { echo "=== BUILD/ARTEFACT GATE FAILED — aborting before any measurement ==="; exit 1; }
git log --oneline -1

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -2
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log") watchdog=$(grep -ac WATCHDOG "$HOME/armrun_${name}.log")"
  grep -a "step pos" "$HOME/armrun_${name}.log" | tail -1
}

rm -rf /tmp/f7 /tmp/f8 /tmp/f6 /tmp/f6dc /tmp/f1
run F7_RING   DSV41_GATEUP_DUMP=/tmp/f7 DSV41_MOE_BS_MBAR_RING=1
run F8_DRAIN  DSV41_GATEUP_DUMP=/tmp/f8 DSV41_MOE_BS_DRAIN=1 DSV41_MOE_BS_BOUNDED_WAIT=1
run F6_ASF    DSV41_GATEUP_DUMP=/tmp/f6 DSV41_MOE_BS_ZERO_ASF=1
run F6_ASFDC  DSV41_GATEUP_DUMP=/tmp/f6dc DSV41_MOE_BS_ZERO_ASF=1 DSV41_MOE_BS_DCLEAR=2
run F1_DC2    DSV41_GATEUP_DUMP=/tmp/f1 DSV41_MOE_BS_DCLEAR=2

cat > /tmp/f17_judge.py <<'PY'
import numpy as np, os
def rd(p):
    return np.fromfile(p, dtype='<f4') if (os.path.exists(p) and os.path.getsize(p) > 0) else None
oracle = rd('/tmp/gu_bs_new.f32')
print(f"oracle: {'missing' if oracle is None else f'n={oracle.size} max|.|={np.abs(oracle).max():.6g}'}")
for tag, d in (('F7_RING ', '/tmp/f7'), ('F8_DRAIN', '/tmp/f8'), ('F6_ASF  ', '/tmp/f6'),
               ('F6_ASFDC', '/tmp/f6dc'), ('F1_DC2  ', '/tmp/f1')):
    a = rd(f'{d}/eager/gateup.f32')
    g = rd(f'{d}/eager/gc_all.f32') or rd(f'{d}/eager/gc_seg0.f32')
    if a is None:
        print(f'{tag}: no dump'); continue
    line = (f'{tag}: out max|.|={np.abs(a).max():.6g} nonzero={int((a!=0).sum()):5d} '
            f'nan={int(np.isnan(a).sum()):4d}')
    if g is not None:
        line += f' | gc max|.|={np.abs(g).max():.6g} nan={int(np.isnan(g).sum())}'
    if oracle is not None and a.size == oracle.size:
        den = np.maximum(np.abs(oracle), 1e-6); rel = np.abs(a - oracle) / den
        line += f' | rel-med={np.median(rel):.4g} corr={np.corrcoef(np.nan_to_num(a), oracle)[0,1]:+.4f}'
    print(line)
PY
python3 /tmp/f17_judge.py
echo
echo "READ: F7_RING/F8_DRAIN step times decide the barrier question (13 ms = arrivals land, 150 ms = they do not)."
echo "      A timeout line in F8 is by itself the answer for slot 40."
