#!/bin/bash
# Batch 7 (stage-1 isolation): collapse production to the isolation instrument's regime — ONE 128-K
# stage — and judge the resulting K<128 partial against the CPU oracle.
#
#   S1 : DSV41_MOE_BS_STAGE1=1 — the hand-written kernel runs only K-stage 0.
#
# Reading:
#   oracle(K<128) == arm(S1)  => a single stage is RIGHT in production: the staging, the SF, the
#                                descriptors and the K indexing all agree with the official
#                                semantics, so the defect is in the STAGE-TO-STAGE progression
#                                (per-stage buffers/barriers/descriptor advance).
#   oracle(K<128) != arm(S1)  => even one stage is wrong in production, i.e. the difference from
#                                the PASSING instrument is on the ingest side (weight pool / gather /
#                                host-built arguments), not in the loop.
set -uo pipefail
cd "$HOME/ferrite"
echo "=== build (hash-gated: skips when the .cu content is unchanged) ==="
bash "$HOME/ensure_built.sh" || { echo "=== BUILD/ARTEFACT GATE FAILED — aborting before any measurement ==="; exit 1; }
echo "=== tree ==="; git log --oneline -1
stat -c "%y %n" kernels/cuda/libferrite_kernels.so target/release/ferrite-serve

run () {
  local name="$1"; shift
  echo "########## $name : $* ##########"
  bash "$HOME/arm_run.sh" "$name" "$@" 2>&1 | tee "$HOME/armrun_${name}.txt" \
     | grep -aE "OUT:|SERVE_FAILED|WATCHDOG" | head -3
  echo "--- steps=$(grep -ac 'step pos' "$HOME/armrun_${name}.log") ar5=$(grep -ac 'ar5-hang' "$HOME/armrun_${name}.log")"
  grep -a "single K-stage\|gateup-dump" "$HOME/armrun_${name}.log" | head -3
  grep -a "OUT:" "$HOME/armrun_${name}.txt" | head -1
}

# NOTE (user directive 2026-09-13): the proven per-slot path always is right — never spend an arm on
# it again. Its first-call inputs are deterministic (same prompt => same embedding => same layer-0
# MoE input), so the oracle can be fed from the arm's OWN dump.
rm -rf /tmp/st1_BS
run ST1_BS DSV41_GATEUP_DUMP=/tmp/st1_BS DSV41_MOE_BS_STAGE1=1 DSV41_MOE_BS_SFDUMP=/tmp/sfd_st1

echo "=== oracle K<128 (stage 0 only) on the SAME dumped input ==="
if [ -f /tmp/st1_BS/eager/x.f32 ]; then D=/tmp/st1_BS/eager; else D=/tmp/st1_BS; fi
timeout 900 python3 "$HOME/gu_numpy_ref.py" --in-dir "$D" --kmax 128 --out /tmp/gu_k128.f32 2>&1 | tail -8

echo "=== arm(stage1) vs oracle(K<128)  — the decisive split ==="
python3 "$HOME/gdu_cmp.py" "$D/gateup.f32" /tmp/gu_k128.f32 6 640 10.0 2>&1 | head -10
echo
echo "=== staged-operand content check + replay (same dump) ==="
if [ -f "$HOME/sfdump_check.py" ] && [ -d /tmp/sfd_st1 ]; then
  timeout 600 python3 "$HOME/sfdump_check.py" --in-dir /tmp/sfd_st1 2>&1 | tail -12
fi
if [ -f "$HOME/sfdump_replay.py" ] && [ -d /tmp/sfd_st1 ]; then
  timeout 900 python3 "$HOME/sfdump_replay.py" --sfdump /tmp/sfd_st1 --gu "$D" 2>&1 | tail -5
fi
