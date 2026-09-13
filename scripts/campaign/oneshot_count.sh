#!/bin/bash
# ONE run, no baseline (we have it): start the standard serve, ask the counting prompt, look at the
# text, shut down. The counting prompt is the project's most sensitive probe for repetition and
# misalignment, and the first 61 lines must read 1..61.
set -uo pipefail
cd "$HOME/ferrite"
PORT=8899
LOG="$HOME/oneshot_num.log"
pkill -9 -x ferrite-serve 2>/dev/null
# readiness gate instead of a fixed sleep: wait for the port to be free, then for /health.
for i in $(seq 1 60); do pgrep -x ferrite-serve >/dev/null || break; done

BASE="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda DSV41_KERNELS=$HOME/ferrite/kernels/cuda/libferrite_kernels.so"
GRAPH_OFF="FERRITE_GRAPH=0 FERRITE_GRAPH_LAYER=0 FERRITE_GRAPH_MOE=0 FERRITE_GRAPH_MID=0 FERRITE_GRAPH_DSA=0 DSV41_GRAPH_STEP=0 DSV41_VERIFY_GRAPH=0 DSV41_DRAFT_GRAPH=0"
COMMON="NCCL_NVLS_ENABLE=0 DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 DSV41_EXPERT_ILV=0 DSV41_MOE_TILELANG_BS=1 DSV41_MOE_BS_HANDWRITTEN=1 DSV41_AR_V5=0 $GRAPH_OFF"

setsid nohup env -u FERRITE_P2P $BASE $COMMON ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
    --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port $PORT > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 120); do
    curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "[oneshot] serve is answering /health"; break; }
    pgrep -x ferrite-serve >/dev/null || { echo "[oneshot] serve DIED during load"; break; }
done

echo "=== the counting prompt, verbatim (first 64 lines) ==="
curl -s -m 300 "http://localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"请从1数到100，每个数字单独一行"}],"max_tokens":400,"temperature":0}' \
  | python3 -c 'import json,sys
try:
    t=json.load(sys.stdin)["choices"][0]["message"]["content"]
except Exception as e:
    print("PARSE_FAIL",e); raise SystemExit
print(t[:1500])'
echo "=== the standard judgement ==="
bash "$HOME/verify_correct.sh" $PORT oneshot 2>&1 | tail -25
echo "=== step timing (p50 of the last 200) ==="
grep -a "step pos" "$LOG" | tail -200 | python3 -c 'import sys,re
v=[float(re.search(r"([0-9.]+)ms",l).group(1)) for l in sys.stdin if re.search(r"([0-9.]+)ms",l)]
v.sort()
print(f"n={len(v)} p50={(v[len(v)//2] if v else 0):.2f}ms p10={(v[len(v)//10] if v else 0):.2f}ms")' 2>/dev/null
curl -s -m 20 -X POST "http://localhost:$PORT/shutdown" >/dev/null 2>&1
echo "[oneshot] done"
