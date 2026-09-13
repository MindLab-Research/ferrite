#!/bin/bash
# Generic single-arm e2e runner:  bash arm_run.sh <name> [EXTRA_ENV ...]
# Starts one TP8 serve with the given extra env, asks the counting prompt, prints
# the output + error count + step + the NUMCHECK / GATHER-DIAG diagnostics, shuts down.
set -uo pipefail
cd ~/ferrite
NAME="${1:-arm}"; shift || true
PORT=8899
BASE="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda DSV41_KERNELS=$HOME/ferrite/kernels/cuda/libferrite_kernels.so"
# ALL FIVE graph gates are disabled so that (a) the BS shim is exercised on EVERY call (no
# capture-time decline / GEMV fallback) and (b) NUMCHECK's non-capture guard passes and the
# [NC] numeric probe actually prints. NOTE: with the graphs off the step time is ~10x slower,
# so never read this arm's STEP line as a performance number. See
# docs/agent/moe-bs-crash-investigation.md §16/§33/§35.
GRAPH_OFF="FERRITE_GRAPH=0 FERRITE_GRAPH_LAYER=0 FERRITE_GRAPH_MOE=0 FERRITE_GRAPH_MID=0 FERRITE_GRAPH_DSA=0 DSV41_GRAPH_STEP=0"
COMMON="DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 DSV41_EXPERT_ILV=0 DSV41_MOE_TILELANG_BS=1 DSV41_MOE_BS_HANDWRITTEN=1"
LOG=~/armrun_${NAME}.log
pkill -9 -x ferrite-serve 2>/dev/null; sleep 2
setsid nohup env $BASE $COMMON "$@" ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
  --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port $PORT > $LOG 2>&1 < /dev/null &
disown
ok=0
for i in $(seq 1 72); do curl -sf -m 3 http://localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
if [ $ok -ne 1 ]; then
  echo "[$NAME] SERVE_FAILED"
  echo "[$NAME] --- decisive lines ---"
  grep -nE "fault|mismatch|error|Error|refus|REFUS|abort|panic" $LOG 2>/dev/null | tail -8 | sed "s/^/[$NAME] /"
  echo "[$NAME] --- log tail ---"
  tail -12 $LOG 2>/dev/null | sed "s/^/[$NAME] /"
  exit 1
fi
echo "[$NAME] ENV: $*"
r=$(curl -s -m 180 http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"请从1数到10，每个数字单独一行"}],"max_tokens":40,"temperature":0}')
echo "[$NAME] OUT: $(echo "$r" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(repr(d.get("choices",[{}])[0].get("message",{}).get("content","")[:200]))
except Exception as e: print("PARSE_FAIL", e)' 2>/dev/null)"
echo "[$NAME] ERR_COUNT: $(grep -cE 'cuda error|err 716|err 700' $LOG)"
echo "[$NAME] SWITCHES: $(grep -E 'scale_vec::1X|canon layout|swapAB' $LOG | tr '\n' ' ')"
echo "[$NAME] STEP: $(grep -oE '\[dsv41\] step pos=[0-9]+: [0-9.]+ms \([0-9.]+ tok/s\)' $LOG | tail -1)"
grep -m1 "GATHER-DIAG" $LOG
grep "\[NC\]" $LOG | head -5
grep -m1 "\[NC\] WORST" $LOG
curl -sf -m 3 -X POST http://localhost:$PORT/shutdown >/dev/null 2>&1 || pkill -9 -x ferrite-serve 2>/dev/null
echo "[$NAME] DONE"
