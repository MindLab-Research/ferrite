#!/usr/bin/env bash
# nsys per-kernel timing for the Wave 1 gate combination (DeepSeek-V4.1-Flash serve).
#
# WHY nsys: the serve-side wall clock (`[dsv41] step pos=N: X.XXms`) is a FAKE
# model figure — it carries five non-model overheads (curl/tail, HTTP/SSE, the
# admission ramp, request queuing, and the tail segment where the next request's
# prefill lands). The `[dspark] steps=` line is only HALF true: it is measured
# inside the model, but `verify=` still contains the host barrier and a D2H sync.
# The ONLY measurement that is pure model execution time is the per-kernel GPU
# time in an nsys profile. See docs/agent/dsv41-nsys-v14-plan.md §2 and the
# timing correction in docs/agent/dspark-correctness-chain.md ("用户的 step 计时
# 纠正（第二次强调）").
#
# ================================ AR MODE ================================
# The user asked for "nccl, not ar p2p (may deadlock)". The findings, verified
# against this tree (2026-09-12):
#
#   * DSV41's all-reduce is `ferrite_dsv41::tp::Collective::all_reduce_inplace`
#     (crates/ferrite-models/src/dsv41/tp.rs:597). It is an INDEPENDENT path:
#     peer copies + a local reduction, and the crate says so itself —
#     "Communication is peer copies plus a local reduction — no NCCL" (tp.rs:10).
#     DSV41 has NO NCCL all-reduce at all (NcclGroup is only used by the GLM
#     `ferrite-exec` path and by kernel tests).
#
#   * FERRITE_P2P is never read by DSV41. It belongs to the GLM bring-up
#     (crates/ferrite-exec/src/tp.rs:713 `std::env::var_os("FERRITE_P2P")`), so
#     the FERRITE_P2P p2p-vs-NCCL deadlock the user remembers DOES NOT APPLY
#     here — nsys is safe on that axis for the DSV41 serve.
#
#   * ⚠️ FERRITE_P2P is PRESENCE-checked (`var_os(...).is_some()`), so setting
#     `FERRITE_P2P=0` ENABLES p2p, the opposite of the intent. We `env -u
#     FERRITE_P2P` instead. (Harmless for DSV41 either way; done for clarity.)
#
#   * The REAL nsys hazard is the DEFAULT device-side AR
#     (`ferrite_p2p_ar_v5`): its publish kernel SPINS on peer stamps, and under
#     nsys's per-node tracing that spin is amplified ~300x (measured: 240s of
#     wall clock for 69 steps). We therefore pin the host-barrier AR —
#     `DSV41_AR_V5=0` + `DSV41_GRAPH_STEP=0` — which is the "not ar p2p" mode.
#     ar_v5() is `graph || env` (tp.rs:776), so BOTH pins are load-bearing:
#     with the whole-step graph left ON, DSV41_AR_V5=0 is short-circuited away.
#
#   * There is no NCCL AR in DSV41 to select, so "nsys 用 nccl" is satisfied in
#     spirit by the host-barrier AR above. NCCL_NVLS_ENABLE=0 is kept as a cheap
#     guard (it is the base env the Wave-1/full-stack runs already set).
#
# ============================ CAPTURE RANGE ============================
# Do NOT pass `--capture-range=cudaProfilerApi`. That works for the GLM serve
# (ferrite-serve's GpuEngine opens the window under FERRITE_NCU and ferrite-serve
# registers a profiler_stop shutdown hook), but the DSV41 serve registers NO
# profiler_stop hook — `ferrite_dsv41::serve::run_serve` builds
# `ServeOptions::new(addr, model_name)` with no `.on_shutdown(...)` — so nsys
# would wait for a cudaProfilerStop that never comes and NEVER flush the report
# (the documented "serve never exits" trap). A full-process profile stopped with
# SIGINT is the working shape.
#
# ================================ USAGE ================================
#   scripts/nsys_wave1.sh                 # 20-token request, /tmp/wave1_nsys
#   DUR=240 MAXTOK=20 scripts/nsys_wave1.sh
#   PROMPT='请背诵《出师表》开头' MAXTOK=20 scripts/nsys_wave1.sh
#   BIN=./target/release/dsv41-run scripts/nsys_wave1.sh   # the one-shot runner
#
# Products: /tmp/wave1_nsys.nsys-rep  +  .csv (cuda_gpu_kern_sum)  +  .log
# Per-kernel median table: see the summary the script prints at the end, or
# re-run by hand:
#   $NSYS stats --report cuda_gpu_kern_sum --format csv /tmp/wave1_nsys.nsys-rep
#
# HARD CAP: $DUR seconds (default 300 = the user's 5-minute limit). The watchdog
# below SIGINTs nsys at the cap, which is what finalizes the report.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-/tmp/wave1_nsys}"
DUR="${DUR:-300}"                       # hard cap, seconds (user: <= 5 min)
NSYS="${NSYS:-/usr/local/cuda-13.2/bin/nsys}"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
KERNELS="${DSV41_KERNELS:-$ROOT/kernels/cuda/libferrite_kernels.so}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
PORT="${PORT:-8097}"                    # off the A/B default (8090) so a stray serve is detected
BIN="${BIN:-$ROOT/target/release/ferrite-serve}"
PROMPT="${PROMPT:-请写一篇关于人工智能的短文}"
MAXTOK="${MAXTOK:-20}"                  # 20 tokens: a few spec steps is all a profile needs

# ---- Wave 1 gate combination (all DSV41_-prefixed) ------------------------
# HC_VERIFY_FUSE + HC_FRONT_ROWS + VERIFY_AR_FOLD  (A1/A2 + the AR fold)
# GATE_MROWS + INDEXER_MROWS + COMPRESSOR_MROWS    (the mrows family)
# LAZY_VERIFY + VERIFY_GRAPH                       (the verify arms)
# BF16_TRUNCATE                                    (the truncate arm)
# DSV41_SPEC/DSPARK arm the DSpark real-commit path the above gates live in;
# without them the verify gates are inert (spec_mode falls back to a plain step).
GATES=(
  DSV41_SPEC=1 DSV41_DSPARK=1
  DSV41_HC_VERIFY_FUSE=1
  DSV41_HC_FRONT_ROWS=1
  DSV41_VERIFY_AR_FOLD=1
  DSV41_GATE_MROWS=1
  DSV41_INDEXER_MROWS=1
  DSV41_COMPRESSOR_MROWS=1
  DSV41_LAZY_VERIFY=1
  DSV41_VERIFY_GRAPH=1
  DSV41_BF16_TRUNCATE=1
  DSV41_TIMING=1
)

# ---- nsys-safe execution mode --------------------------------------------
# AR: host barrier, NO device-side p2p spin (both pins required, see header).
AR_SAFE=(DSV41_AR_V5=0 DSV41_GRAPH_STEP=0)
# NCCL guard (harmless for DSV41 — no NCCL AR — keeps the GLM p2p path off).
NCCL_SAFE=(NCCL_NVLS_ENABLE=0)

# ---- preflight ------------------------------------------------------------
[ -e "$NSYS" ] || { echo "FATAL: nsys not found at $NSYS (override with NSYS=...)" >&2; exit 1; }
[ -e "$BIN" ]  || { echo "FATAL: missing $BIN - build it first (cargo build --release)" >&2; exit 1; }
[ -e "$KERNELS" ] || { echo "FATAL: missing $KERNELS - build the .so first" >&2; exit 1; }
if pgrep -x "$(basename "$BIN")" >/dev/null 2>&1; then
  echo "FATAL: a $(basename "$BIN") is already alive — the GPUs must be exclusive" >&2
  pgrep -ax "$(basename "$BIN")" >&2
  exit 1
fi

LOG="$OUT.log"
REP="$OUT.nsys-rep"
rm -f "$REP" "$OUT.csv" "$LOG"
mkdir -p "$(dirname "$OUT")"

echo "== nsys Wave-1 per-kernel profile =="
echo "   bin      : $BIN"
echo "   out      : $REP"
echo "   cap      : ${DUR}s"
echo "   request  : max_tokens=$MAXTOK  prompt='$PROMPT'"
echo "   gates    : ${GATES[*]}"
echo "   ar-safe  : ${AR_SAFE[*]}  ${NCCL_SAFE[*]}  (FERRITE_P2P unset)"

# ---- launch under nsys ----------------------------------------------------
# `--sample=none`: no CPU sampling (we only want the CUDA kernel table).
# `--trace=cuda,nvtx`: CUDA + NVTX ranges (the DSV41 path emits no NVTX ranges;
# this is harmless and keeps the door open for a future annotation).
# `env -u FERRITE_P2P`: see the header — "0" would ENABLE p2p, so UNSET it.
env -u FERRITE_P2P "${GATES[@]}" "${AR_SAFE[@]}" "${NCCL_SAFE[@]}" \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    DSV41_MODEL_DIR="$MODEL_DIR" \
    DSV41_KERNELS="$KERNELS" \
  "$NSYS" profile \
    --trace=cuda,nvtx --sample=none \
    --output="$OUT" --force-overwrite=true \
    "$BIN" --model dsv41 --serve --tp "$TP" --model-dir "$MODEL_DIR" --port "$PORT" \
    >"$LOG" 2>&1 &
NSYS_PID=$!

# 5-minute hard cap: SIGINT nsys (that finalizes the report).
( sleep "$DUR"; kill -INT "$NSYS_PID" 2>/dev/null ) &
WATCHDOG=$!

# ---- wait for the listener, then drive ONE short request ------------------
ready=0
for _ in $(seq 1 60); do
  sleep 2
  grep -q "serving" "$LOG" 2>/dev/null && { ready=1; break; }
  kill -0 "$NSYS_PID" 2>/dev/null || break
done
if [ "$ready" != 1 ]; then
  echo "FATAL: serve never printed 'serving'; see $LOG (tail):" >&2
  tail -20 "$LOG" >&2
  kill -INT "$NSYS_PID" 2>/dev/null; sleep 5; kill -9 "$NSYS_PID" 2>/dev/null
  kill "$WATCHDOG" 2>/dev/null
  exit 1
fi

# The ranks load BEHIND the first request, so this curl includes the weight load;
# generous client timeout, capped by the watchdog above.
echo "-- driving one request (max_tokens=$MAXTOK) --"
curl -s -m 260 -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAXTOK,\"temperature\":0}" \
  >"$OUT.reply.json" 2>"$OUT.curl.err" \
  && python3 -c 'import sys,json;d=json.load(open(sys.argv[1]));print("   reply:",repr(d["choices"][0]["message"]["content"][:60]))' "$OUT.reply.json" \
  || { echo "   curl failed (see $OUT.curl.err)"; tail -5 "$OUT.curl.err"; }

# ---- stop nsys cleanly (SIGINT finalizes the report) ----------------------
kill "$WATCHDOG" 2>/dev/null
kill -INT "$NSYS_PID" 2>/dev/null
for _ in $(seq 1 30); do
  kill -0 "$NSYS_PID" 2>/dev/null || break
  sleep 2
done
if kill -0 "$NSYS_PID" 2>/dev/null; then
  echo "WARN: nsys did not exit after SIGINT, killing (report may be truncated)" >&2
  kill -9 "$NSYS_PID" 2>/dev/null
fi
wait "$NSYS_PID" 2>/dev/null

# exact-PID cleanup of any surviving serve
for p in $(pgrep -x "$(basename "$BIN")"); do kill -9 "$p" 2>/dev/null; done

# ---- per-kernel table -----------------------------------------------------
if [ ! -s "$REP" ]; then
  echo "FATAL: no $REP — nsys did not capture. Last log lines:" >&2
  tail -20 "$LOG" >&2
  exit 1
fi
# CSV, never the default table: kernel names contain spaces/commas and would
# shift the columns (dsv41_profile.sh:31 documents the same trap).
"$NSYS" stats --report cuda_gpu_kern_sum --format csv "$REP" >"$OUT.csv" 2>"$OUT.stats.err"
rows=$(wc -l <"$OUT.csv")
if [ "$rows" -lt 5 ]; then
  echo "FATAL: $OUT.csv has $rows rows — the profile did not capture the binary." >&2
  echo "       stderr: $(cat "$OUT.stats.err")" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

# Steps: the number of decode steps the request actually ran (each printed as
# "[dsv41] step pos=N"). Used only to normalize ms/step; the per-kernel totals
# are the authoritative numbers.
STEPS=$(grep -cE "\[dsv41\] step pos=" "$LOG" 2>/dev/null || echo 0)
WORLD="$TP"

python3 - "$OUT.csv" "$STEPS" "$WORLD" <<'PY'
import csv, sys
csv_path, steps, world = sys.argv[1], max(int(sys.argv[2]), 1), int(sys.argv[3])
rows = []
for r in csv.reader(open(csv_path)):
    if len(r) < 6:
        continue
    try:
        total, inst, med = float(r[1]), int(r[2]), float(r[4])
    except ValueError:
        continue
    rows.append((total, inst, med, r[-1].strip()))   # Name is the LAST field
rows.sort(reverse=True)
tot = sum(r[0] for r in rows) or 1.0
print()
print(f"total GPU kernel time = {tot/1e6:.1f} ms over {steps} steps x {world} ranks")
print(f"{'share':>7} {'inst':>7} {'med us':>8} {'ms/stp/w':>9}  kernel")
for total, inst, med, name in rows[:20]:
    print(f"{total/tot*100:6.1f}% {inst:7d} {med/1000:8.2f} {total/1e6/steps/world:9.3f}  {name[:52]}")
print()
print(f"full CSV: {csv_path}")
PY
echo "raw reply: $OUT.reply.json    log: $LOG"
