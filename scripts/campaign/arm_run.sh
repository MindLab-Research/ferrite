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
GRAPH_OFF="FERRITE_GRAPH=0 FERRITE_GRAPH_LAYER=0 FERRITE_GRAPH_MOE=0 FERRITE_GRAPH_MID=0 FERRITE_GRAPH_DSA=0 "
         "DSV41_GRAPH_STEP=0 DSV41_VERIFY_GRAPH=0 DSV41_DRAFT_GRAPH=0"
# The last two are default-OFF today (chain_dev.rs:4233, dspark_dev.rs:186) but are now
# PINNED here: the project documents ar5-hang as specific to batched + CUDA graph
# (docs/agent/400-final-frontier-analysis.md:27-28) with nograph measured at 0 ar5-hang,
# so a diagnostic arm must not leave any graph gate to chance.
# AR-DEADLOCK AVOIDANCE (AGENTS.md measurement discipline, mandatory on this node):
# without DSV41_AR_V5=0 the arms hang in the all-reduce v5 path and print
#   [ar5-hang] rank=.. peer=.. need=.. cur=.. spins>5000000 TIMEOUT -> PARK
# forever, so the completion never returns and the round yields an EMPTY body with no text to
# judge. That silent hang is what made several "decisive" rounds inconclusive even though the
# binary was valid. NCCL_NVLS_ENABLE=0 is also required here (otherwise NCCL silently falls
# back to a ~2.4x slower host all-reduce), and FERRITE_P2P must be UNSET (the launcher passes
# `env -u FERRITE_P2P` below).
COMMON="DSV41_AR_V5=0 NCCL_NVLS_ENABLE=0 DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 DSV41_EXPERT_ILV=0 DSV41_MOE_TILELANG_BS=1 DSV41_MOE_BS_HANDWRITTEN=1 DSV41_MOE_BS_NUMCHECK=1 $GRAPH_OFF"
LOG=~/armrun_${NAME}.log
pkill -9 -x ferrite-serve 2>/dev/null; sleep 2
setsid nohup env -u FERRITE_P2P $BASE $COMMON "$@" ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
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
# TOOLING FIX: plain `curl -s` swallows failures, so a request made while the model is still
# warming up returns an EMPTY body and the round yields no text to judge (observed: OUT empty,
# wq_check NO-OUT-LINE). Capture the HTTP status and retry a bounded number of times; on final
# failure log the status and the body head so the log itself says why.
# WATCHDOG (2026-09-14): a hung serve used to be discovered only by a human noticing a frozen
# log mtime (AGENTS.md says exactly that), which burns a whole GPU window. Start a background
# monitor that kills the serve if the log stops growing, so the round fails fast instead.
# Excluded from the "no foreground sleep" rule: this loop runs INSIDE the script, in the
# background, and exits as soon as the run ends.
( prev=0; stall=0
  while pgrep -x ferrite-serve >/dev/null 2>&1; do
    sleep 10
    cur=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$cur" = "$prev" ]; then
      stall=$((stall+1))
      if [ "$stall" -ge 12 ]; then          # ~120s with no log growth => hung
        echo "[$NAME] WATCHDOG: log frozen at $cur bytes for ~120s -> killing serve (likely the §92 unbounded mbar spin)" >> "$LOG"
        pkill -x ferrite-serve 2>/dev/null
        break
      fi
    else
      stall=0
    fi
    prev=$cur
  done ) &
WD_PID=$!

BODY='{"messages":[{"role":"user","content":"请从1数到10，每个数字单独一行"}],"max_tokens":40,"temperature":0}'
r=""; CODE=""
for attempt in 1 2 3 4 5 6; do
  r=$(curl -s -m 180 -w "\n%{http_code}" http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" -d "$BODY")
  CODE=$(printf "%s" "$r" | tail -n1)
  r=$(printf "%s" "$r" | sed "$d")
  if [ "$CODE" = "200" ] && printf "%s" "$r" | grep -q "choices"; then break; fi
  echo "[$NAME] request attempt $attempt: http=$CODE body_head=$(printf "%s" "$r" | head -c 120)" >> $LOG
  r=""; sleep 5
done
if [ -z "$r" ]; then echo "[$NAME] WARN: completions never returned a usable body (last http=$CODE)" >> $LOG; fi
echo "[$NAME] OUT: $(echo "$r" | python3 -c {'
import json,sys
raw=sys.stdin.read()
try:
    d=json.loads(raw)
    c=d.get("choices",[{}])[0].get("message",{}).get("content","")
    print(repr(c[:400]))
except Exception:
    print(repr(raw[:400]))
'} 2>/dev/null)"