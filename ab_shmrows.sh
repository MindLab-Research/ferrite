#!/bin/bash
# A/B helper for the shared-expert multi-row gate (DSV41_SH_EXP_MROWS).
# usage:
#   ab_shmrows.sh launch <TAG> [EXTRA_ENV...]   -> detached serve + wait for ready
#   ab_shmrows.sh gen    <TAG>                  -> curl 出师表 max_tokens=300 + parse + shutdown
#   ab_shmrows.sh gens   <TAG> <N>              -> N repeated samples (no shutdown) + env check
#   ab_shmrows.sh envchk <TAG>                  -> dump the running serve's gate env
set -u
CMD="$1"; TAG="$2"; shift 2
LOG=/tmp/shmrows_${TAG}.log
OUT=/tmp/shmrows_${TAG}.json
PORT=8320
BASE_ENV="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1"

case "$CMD" in
launch)
  pkill -9 -x ferrite-serve 2>/dev/null; sleep 4
  cd "$HOME/ferrite" || exit 9
  setsid env $BASE_ENV "$@" LD_LIBRARY_PATH="$HOME/ferrite/kernels/cuda" \
    timeout 900 ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
    --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port $PORT \
    > "$LOG" 2>&1 < /dev/null &
  echo "launched tag=$TAG extra=[$*] log=$LOG"
  READY=0
  for i in $(seq 1 80); do
    if grep -q "chain ready, serving" "$LOG" 2>/dev/null; then READY=1; echo "READY after ~$((i*6))s"; break; fi
    if grep -qi "build-id mismatch" "$LOG" 2>/dev/null; then echo "!!!!! BUILD-ID MISMATCH !!!!!"; break; fi
    pgrep -x ferrite-serve >/dev/null || { echo "!!!!! serve died before ready"; break; }
    sleep 6
  done
  echo "=== gate/build-id 相关行 ==="
  grep -iE "build.id|mismatch|SH_EXP_MROWS|sh_exp" "$LOG" | head -8
  if [ "$READY" != 1 ]; then echo "=== NOT READY, tail ==="; tail -25 "$LOG"; exit 2; fi
  ;;
gen)
  sleep 4
  curl -s --noproxy "*" -m 300 http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"请完整背诵《出师表》全文。"}],"max_tokens":300,"stream":false}' > "$OUT"
  python3 -c "
import json
try:
    c=json.load(open('$OUT'))['choices'][0]['message']['content']
except Exception as e:
    print('PARSE-FAIL',repr(e)); print(open('$OUT').read()[:300]); raise SystemExit(0)
bad=[i for i in range(1,len(c)) if c[i]==c[i-1] and not c[i].isspace()]
print('LEN',len(c),'双字',len(bad))
print('HEAD:',''.join(c[:160].split()))
print('TAIL:',''.join(c[-100:].split()))
"
  echo "=== steps 行（原文） ==="; grep "dspark] steps" "$LOG" | tail -3
  echo "=== verify_graph 出现次数（判据要求=0） ==="; grep -c "verify_graph" "$LOG"
  echo "=== first_mismatch 计数 ==="; grep -c "first_mismatch=[0-9]" "$LOG"
  echo "=== decline/失败 相关 ==="; grep -inE "declin|fallback|reject|panic|abort" "$LOG" | head -6
  curl -s --noproxy "*" -m 10 -X POST http://localhost:$PORT/shutdown >/dev/null 2>&1
  sleep 3; pkill -9 -x ferrite-serve 2>/dev/null
  echo "=== done tag=$TAG ==="
  ;;
envchk)
  P=$(pgrep -x ferrite-serve | head -1)
  echo "serve pid=$P"
  [ -n "$P" ] && tr '\0' '\n' < /proc/$P/environ | grep -E "DSV41_" | sort
  ;;
gens)
  N="${3:-3}"
  sleep 4
  for i in $(seq 1 "$N"); do
    curl -s --noproxy "*" -m 300 http://localhost:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"请完整背诵《出师表》全文。"}],"max_tokens":300,"stream":false}' > /tmp/shmrows_${TAG}_s$i.json
    python3 -c "
import json
try:
    c=json.load(open('/tmp/shmrows_${TAG}_s$i.json'))['choices'][0]['message']['content']
except Exception as e:
    print('sample $i PARSE-FAIL',repr(e)); raise SystemExit(0)
import hashlib
bad=[j for j in range(1,len(c)) if c[j]==c[j-1] and not c[j].isspace()]
print('sample $i LEN',len(c),'双字',len(bad),'md5',hashlib.md5(c.encode()).hexdigest()[:12])
"
  done
  echo "=== 本臂 steps 行（全部样本） ==="; grep "dspark] steps" "$LOG"
  curl -s --noproxy "*" -m 10 -X POST http://localhost:$PORT/shutdown >/dev/null 2>&1
  sleep 3; pkill -9 -x ferrite-serve 2>/dev/null
  echo "=== done gens tag=$TAG ==="
  ;;
*)
  echo "usage: $0 {launch|gen|gens|envchk} <TAG> [EXTRA_ENV...]"; exit 1;;
esac
