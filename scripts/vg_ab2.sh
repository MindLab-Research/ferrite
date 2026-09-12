#!/usr/bin/env bash
# vg_ab2.sh — the DSpark verify-graph A/B, an EQUIVALENT of scripts/verify_graph_ab.sh
# with the cases chosen so each arm can actually reach the `dspark_steps % 50 == 0`
# gate that prints `[dspark] steps=… verify=Xms`.
#
# WHY the case set differs from verify_graph_ab.sh:
#   On the measured revision (cee7cffd3) the 出师表 prompt collapses to 63 tokens
#   (EOS) under the open verify-value corruption, i.e. ~49 dspark steps — ONE short
#   of the 50-step print, so that prompt can never produce `verify=Xms` here. The
#   prose prompt runs >150 steps, so it carries the measurement; 出师表 is kept as
#   the project yardstick for the ENGAGEMENT evidence and the arm-vs-arm TEXT check.
#
# 口径 (unchanged from verify_graph_ab.sh):
#   * 4 serves, strictly serial, one prompt per serve (the `[dspark]` accumulator is
#     a SERVE-PROCESS average, never reset per request);
#   * the =1 arm must print `[verify_graph] captured …`, the =0 arm must not;
#   * the two arms of the same prompt must emit the SAME text (chars + md5);
#   * verify_ms(=1) <= verify_ms(=0) - 2ms (the task's gate; the script's default
#     MIN_GAIN_MS=10 is the plan's expectation, printed too).
#
# USAGE: nohup bash vg_ab2.sh > /tmp/vgab2/run.log 2>&1 &
set -uo pipefail

REPO="${REPO:-$HOME/ferrite}"
LOGDIR="${LOGDIR:-/tmp/vgab2}"
MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
PORT_BASE="${PORT_BASE:-8195}"
CURL="curl -s --noproxy '*'"
mkdir -p "$LOGDIR"

# The long, measurable case: prose that runs past 150 dspark steps.
P_LONG="${P_LONG:-请写一篇关于春天的散文，内容充实，不少于800字。}"
MAXTOK_LONG="${MAXTOK_LONG:-800}"
# The project's long-prompt yardstick (engagement proof + text comparison).
P_SH="${P_SH:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"
MAXTOK_SH="${MAXTOK_SH:-300}"

echo "== vg_ab2 (DSV41_VERIFY_GRAPH=0 vs 1) =="
echo "-- rev $(git -C "$REPO" rev-parse --short HEAD)  ports $PORT_BASE..$((PORT_BASE + 3))  tp $TP"
echo "-- long case: max_tokens=$MAXTOK_LONG   yardstick: max_tokens=$MAXTOK_SH"

teardown() {
    local port="$1"
    $CURL -m 5 -X POST "http://localhost:$port/shutdown" >/dev/null 2>&1
    for _ in $(seq 1 10); do
        [ "$(pgrep -x ferrite-serve | wc -l)" = 0 ] && break
        sleep 3
    done
    pkill -9 -x ferrite-serve 2>/dev/null
    sleep 8
}

run_case() {  # arm prompt max_tok tag port
    local arm="$1" prompt="$2" max_tok="$3" tag="$4" port="$5"
    local log="$LOGDIR/$tag.log"
    echo
    echo "---- case $tag (port $port): DSV41_VERIFY_GRAPH=$arm  max_tokens=$max_tok"
    pkill -9 -x ferrite-serve 2>/dev/null
    sleep 6
    ( cd "$REPO" && nohup env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
        LD_LIBRARY_PATH="$HOME/ferrite/kernels/cuda" \
        DSV41_KERNELS="$HOME/ferrite/kernels/cuda/libferrite_kernels.so" \
        DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1 \
        DSV41_VERIFY_GRAPH="$arm" DSV41_AR_V5=1 \
        ./target/release/ferrite-serve --model dsv41 --serve --tp "$TP" \
        --model-dir "$MODEL_DIR" --port "$port" > "$log" 2>&1 & )

    local ok=0
    for _ in $(seq 1 60); do
        if $CURL -m 2 "http://localhost:$port/health" >/dev/null 2>&1; then ok=1; break; fi
        sleep 5
    done
    if [ "$ok" = 0 ]; then echo "   FATAL: $tag never healthy"; tail -8 "$log"; teardown "$port"; return 1; fi

    local body t0 t1
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
            "$MODEL_NAME" "$prompt" "$max_tok")"
    t0=$(date +%s)
    $CURL -m 900 "http://localhost:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" > "$LOGDIR/$tag.resp.json" 2>/dev/null
    t1=$(date +%s)
    grep 'dspark] steps' "$log" | tail -1 > "$LOGDIR/$tag.dspark" 2>/dev/null
    cp "$log" "$LOGDIR/$tag.log.kept" 2>/dev/null
    echo "   e2e $((t1 - t0))s   $(head -c 140 "$LOGDIR/$tag.dspark")"
    grep -m3 'verify_graph' "$log" | sed 's/^/   | /'
    teardown "$port"
}

run_case 0 "$P_LONG" "$MAXTOK_LONG" L0 "$PORT_BASE"
run_case 1 "$P_LONG" "$MAXTOK_LONG" L1 "$((PORT_BASE + 1))"
run_case 0 "$P_SH"   "$MAXTOK_SH"   S0 "$((PORT_BASE + 2))"
run_case 1 "$P_SH"   "$MAXTOK_SH"   S1 "$((PORT_BASE + 3))"

python3 "$LOGDIR/report.py" "$LOGDIR" | tee "$LOGDIR/table.txt"
rc=${PIPESTATUS[0]}
echo
echo "vg_ab2 exit=$rc  logs: $LOGDIR/<tag>.{log,log.kept,dspark,resp.json}  table: $LOGDIR/table.txt"
exit 0
