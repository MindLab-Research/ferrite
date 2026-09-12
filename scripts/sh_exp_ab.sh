#!/usr/bin/env bash
# sh_exp_ab.sh — the ONE command for the SH_EXP_MROWS serve A/B.
#
# WHAT IT ANSWERS
#   "Is `DSV41_SH_EXP_MROWS=1` actually DISPATCHING the shared expert's multi-row
#    pass on the real serve, or is it a silent no-op again?"
#
# WHY A SCRIPT IS NEEDED AT ALL (the failure mode it exists to catch)
#   `DevChain::shared_expert_mrows` (crates/ferrite-models/src/dsv41/chain_dev.rs)
#   calls `gemm_fp8_mrows`. Until Direction B (commit 4c98b30) the C entry
#   declined every default-configured process:
#
#       if (g_gemv_a32) return 2;      // kernels/cuda/dsv41_kernels.cu:5084+
#
#   `DSV41_GEMV_A32` defaults ON, so 2 -> `Ok(false)` -> the caller quietly ran
#   its per-row loop and `DSV41_SH_EXP_MROWS=1` changed NOTHING. That is the trap:
#   correct-looking Rust, no kernel, no error, no log line. Direction B moved the
#   a32 choice INTO the kernel (it carries both operand forms and selects on the
#   same `g_gemv_a32` process gate), so the launcher no longer declines the
#   default. THIS SCRIPT IS THE PROOF THAT IT IS NOW LIVE. It does not touch the
#   source, the .so or the binary — it only drives two serves and reads numbers.
#
# HOW IT PROVES IT (a single-variable A/B, nothing else varies)
#   ARM base   DSV41_SH_EXP_MROWS=0   baseline: the per-row loop
#   ARM mrows  DSV41_SH_EXP_MROWS=1   the multi-row pass
#   That one variable is read by exactly ONE gate (`sh_exp_mrows()`) and consumed
#   by exactly ONE call site (`shared_expert_mrows`), so ceteris paribus every
#   millisecond of difference is attributable to the mrows dispatch — there is no
#   second reader to blame.
#
#   PRIMARY JUDGE — `verify_ms` from the `[dspark] steps=...` line (criterion c).
#     The shared expert is a SINGLE expert applied to every row, but `moe_rows`
#     re-reads its weights once per row: 4.42 MB x m per layer, ~25 launches x m
#     per layer. At the production shape (m=5, sh_il=288, dim=5120) that is
#     ~0.89 GB/step and ~960 launches/step of pure repeat traffic — bytes that
#     really do divide by m here (unlike the routed half, whose rows pick
#     DIFFERENT experts). The expected `verify_ms` drop is -2..-5 ms.
#     A drop >= SH_EXP_MIN_DELTA (default 2.0 ms) can ONLY come from the mrows
#     path running. A ~0 delta means the silent no-op is back; a negative delta
#     means it is a regression.
#
#   SECONDARY JUDGE — BIT-IDENTICAL TEXT (criterion d).
#     The multi-row pass is documented bit-identical to the per-row loop it
#     replaces, so both arms must return the SAME greedy answer to the SAME
#     prompt (temperature 0 -> deterministic). The script prints, per arm, the
#     answer length, the adjacent-repeat run count ("双字") and the md5. ANY drift
#     INVALIDATES the run: that is a correctness bug, and the default must NOT be
#     flipped on a drifted arm regardless of how good the timing looks.
#
# WHY TWO SERIAL SERVES AND NOT ONE
#   `sh_exp_mrows()` caches into a process-wide `OnceLock`: the FIRST call in a
#   serve freezes the gate for that process's lifetime, so flipping the env var
#   between two requests inside one serve changes nothing (the second request
#   would silently reuse the first arm's gate — a guaranteed false A/B). The two
#   arms are therefore TWO serial serves. NEVER concurrently: the serve takes all
#   8 GPUs, and a second one would either OOM or contend for SMs and poison the
#   timing. The script kills+waits between arms for exactly this reason.
#
#   (criterion b — launch counts — is deliberately NOT used: DSV41_TIMING does
#    not print a launch count, and adding one would mean changing code. The
#    verify_ms delta above is the reliable signal; the preflight symbol check
#    below covers the static half of the same question.)
#
# USAGE
#   bash scripts/sh_exp_ab.sh                    # run the A/B, print the verdict
#   bash scripts/sh_exp_ab.sh --judge-only       # re-judge existing /tmp logs
#   SH_EXP_MIN_DELTA=3 bash scripts/sh_exp_ab.sh # demand a 3 ms drop (stricter)
#   SH_EXP_REQS=3 bash scripts/sh_exp_ab.sh      # more requests -> more [dspark]
#                                                #   steps -> tighter mean
#   SH_EXP_PORT=8399 ...                         # avoid a busy port
#   SKIP_PREFLIGHT=1 ...                         # skip the .so/binary id gate
#
# EXIT CODES
#   0  mrows CONFIRMED live (verify_ms dropped >= threshold, text identical,
#      both arms fault-free)
#   1  NOT confirmed (drop < threshold -> silent no-op / regression suspected,
#      or text drift -> correctness bug) — DO NOT flip the default
#   2  harness error (missing binary/.so, build-id mismatch, the .so lacks the
#      mrows symbol, a serve never came up) — the numbers are NOT evidence
#
# The caller owns the tree: build BOTH products before running, .so FIRST
# (kernels/cuda/build.sh -> touch ferrite-kernel/build.rs -> cargo build
# --release; see scripts/dsv41_recovery_verify.sh phase 0 for why). This script
# never builds and never writes into the tree.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
K="$ROOT/kernels/cuda"
SO="$K/libferrite_kernels.so"
BIN="$ROOT/target/release/ferrite-serve"

PORT="${SH_EXP_PORT:-8320}"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
LOGDIR="${SH_EXP_LOGDIR:-/tmp/sh_exp_ab}"
MIN_DELTA="${SH_EXP_MIN_DELTA:-2.0}"
REQS="${SH_EXP_REQS:-2}"          # 出师表 requests per arm (steps accumulate per serve)
MAX_TOKENS="${SH_EXP_MAX_TOKENS:-384}"
TP="${SH_EXP_TP:-8}"
READY_TIMEOUT="${SH_EXP_READY_TIMEOUT:-900}"   # seconds to reach "chain ready, serving"
SERVE_TIMEOUT="${SH_EXP_SERVE_TIMEOUT:-1800}"  # hard kill for a wedged rank pool

A_TAG="${SH_EXP_A_TAG:-base}"
B_TAG="${SH_EXP_B_TAG:-mrows}"

# temperature 0 -> greedy, so the two arms are comparable BYTE FOR BYTE.
PROMPT="请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。"

# The three env pins every DSV41 serve A/B in this repo carries. DSV41_SPEC
# implies DSV41_DSPARK (main.rs arms it), but pinning both documents the arm;
# DSV41_TIMING is REQUIRED for the `[dspark] steps=` line to exist at all.
BASE_ENV=(
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    DSV41_MODEL_DIR="$MODEL_DIR"
    DSV41_KERNELS="$SO"
    DSV41_SPEC=1
    DSV41_DSPARK=1
    DSV41_SIDS_WRITEBACK=1
    DSV41_TIMING=1
)

log() { printf '%s\n' "$*" | tee -a "$MAIN_LOG"; }

have() { command -v "$1" >/dev/null 2>&1; }

# Exact-PID cleanup only (pkill -f would match our own command line).
kill_serves() {
    local p
    for p in $(pgrep -x ferrite-serve 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    sleep 8
}

# ---------------------------------------------------------------- preflight --
# Two STATIC facts have to hold before an A/B can mean anything, and neither is
# observable from the timing numbers:
#   1. the binary and the .so are the SAME build (else the serve aborts at dlopen
#      with "kernel build id mismatch" and the whole run is moot);
#   2. the .so actually EXPORTS `dsv41_gemm_fp8_mrows` (else
#      `supports_gemm_fp8_mrows()` is false, `shared_expert_mrows` returns early
#      and BOTH arms run the per-row loop — the A/B would report "no-op" for a
#      reason that has nothing to do with the a32 gate, which is exactly the
#      misdiagnosis this script exists to prevent).
preflight() {
    [ -x "$BIN" ] || { log "FATAL: missing $BIN — build: cargo build --release"; exit 2; }
    [ -f "$SO" ]  || { log "FATAL: missing $SO — build: (cd kernels/cuda && bash build.sh ${ARCH:-103a})"; exit 2; }

    if [ -f "$K/.build_id" ] && have strings; then
        local stamp
        stamp="$(cat "$K/.build_id")"
        # grep -c (not -q): grep -q exits at the first match, SIGPIPEs `strings`,
        # and a pipefail shell then reports the PIPELINE as failed even though the
        # id WAS found — the gate would misfire "stale" on a good pair.
        if ! strings "$BIN" | grep -cF -- "$stamp" >/dev/null; then
            log "FATAL: $BIN does not embed the .so's build_id '$stamp' (stale/mismatched pair)."
            log "  rebuild, .so FIRST:"
            log "    (cd kernels/cuda && bash build.sh ${ARCH:-103a}) && touch crates/ferrite-kernel/build.rs && cargo build --release"
            exit 2
        fi
        log "preflight: build_id $stamp present in both .so and binary"
    else
        log "preflight: no .build_id / no strings — skipping the same-source check"
    fi

    if have nm; then
        if nm -D "$SO" 2>/dev/null | grep -q "dsv41_gemm_fp8_mrows"; then
            log "preflight: .so exports dsv41_gemm_fp8_mrows (multi-row entry present)"
        else
            log "FATAL: .so exports no dsv41_gemm_fp8_mrows — the multi-row path cannot"
            log "  run at all. A no-delta A/B here would be a stale .so, NOT the a32 gate."
            log "  rebuild the kernels: (cd kernels/cuda && bash build.sh ${ARCH:-103a})"
            exit 2
        fi
    else
        log "preflight: nm not found — skipping the symbol check"
    fi
}

# ------------------------------------------------------------------ one arm --
# Launches a fresh serve with BASE_ENV + this arm's one variable, waits for the
# rank pool, dumps the running process's DSV41_* env (proof the pin ARRIVED, not
# just that it was typed), drives REQS identical greedy requests, then tears the
# serve down. Artifacts:
#   $LOGDIR/$tag.log              full serve log (the [dspark] lines live here)
#   $LOGDIR/${tag}_r$i.json       raw response of request i
run_arm() {
    local tag="$1"; shift
    local extra=("$@")
    local logf="$LOGDIR/$tag.log"
    local pid="" n="" i=""

    # `${extra[0]:-}` (not plain `${extra[0]}`): under `set -u` an empty array
    # would read as an unbound variable and kill the shell before the arm runs.
    log "== arm $tag: DSV41_SH_EXP_MROWS=${extra[0]:-<none>} (extra env: ${extra[*]:-<none>}) =="
    kill_serves
    : > "$logf"

    # setsid + </dev/null detaches it from our tty so a Ctrl-C on the script does
    # not also SIGINT the serve mid-measurement; timeout is the backstop for a
    # wedged rank pool.
    setsid env "${BASE_ENV[@]}" ${extra[@]+"${extra[@]}"} \
        timeout "$SERVE_TIMEOUT" \
        "$BIN" --model dsv41 --serve --tp "$TP" \
        --model-dir "$MODEL_DIR" --port "$PORT" \
        >"$logf" 2>&1 </dev/null &

    log "  waiting for 'chain ready, serving' (up to ${READY_TIMEOUT}s) ..."
    local ready=0 waited=0
    while [ "$waited" -lt "$READY_TIMEOUT" ]; do
        if grep -q "chain ready, serving" "$logf" 2>/dev/null; then ready=1; break; fi
        # Fail fast rather than burning 15 min: a dead serve or an id mismatch
        # will never become ready.
        if grep -qi "build-id mismatch\|build id mismatch" "$logf" 2>/dev/null; then
            log "  !!!!! BUILD-ID MISMATCH — the pair is not same-source; aborting"
            break
        fi
        pgrep -x ferrite-serve >/dev/null 2>&1 || { log "  !!!!! serve died before ready"; break; }
        sleep 5; waited=$((waited + 5))
    done
    if [ "$ready" != 1 ]; then
        log "  FATAL: arm $tag never reached ready. Last 25 lines of $logf:"
        tail -25 "$logf" | sed 's/^/    /' | tee -a "$MAIN_LOG"
        kill_serves
        return 2
    fi
    log "  ready after ~${waited}s"

    # The gate input proof: what the RUNNING process actually has, not what we
    # typed. If DSV41_SH_EXP_MROWS is absent here, the arm is invalid.
    pid="$(pgrep -x ferrite-serve | head -1)"
    if [ -n "$pid" ] && [ -r "/proc/$pid/environ" ]; then
        log "  running serve pid=$pid env pins:"
        tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^DSV41_' | sort | sed 's/^/    /' | tee -a "$MAIN_LOG"
    fi

    # Give the HTTP listener a beat past the "serving" banner before the first
    # request (the banner prints as the pool comes up; the bind is right behind).
    sleep 4

    for i in $(seq 1 "$REQS"); do
        local out="$LOGDIR/${tag}_r$i.json"
        curl -s --noproxy "*" -m 300 "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"deepseek-v4.1-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0,\"stream\":false}" \
            > "$out"
        python3 - "$out" "$i" <<'PY' | tee -a "$MAIN_LOG"
import json, sys, hashlib
path, i = sys.argv[1], sys.argv[2]
try:
    c = json.load(open(path))["choices"][0]["message"]["content"]
except Exception as e:
    print(f"  req {i}: PARSE-FAIL {e!r}: {open(path).read()[:200]}")
    raise SystemExit(0)
# "双字" = adjacent identical NON-space characters, the same definition every
# other arm-printer in this repo uses (a repetition/loop canary).
bad = sum(1 for j in range(1, len(c)) if c[j] == c[j - 1] and not c[j].isspace())
print(f"  req {i}: LEN {len(c)} 双字 {bad} md5 {hashlib.md5(c.encode()).hexdigest()[:12]}")
PY
    done

    log "  [dspark] steps lines (cumulative mean; the LAST one is the arm's number):"
    grep "dspark] steps" "$logf" | sed 's/^/    /' | tee -a "$MAIN_LOG"
    log "  fault-looking lines: $(grep -cE 'illegal|fault' "$logf" 2>/dev/null || true)"

    curl -s --noproxy "*" -m 10 -X POST "http://localhost:$PORT/shutdown" >/dev/null 2>&1
    sleep 3
    kill_serves
    return 0
}

# ------------------------------------------------------------------- judge --
# Reads the two arms' logs + first responses and prints the verdict. Kept in
# python so the parsing matches dspark_verify.rs's (LAST `[dspark] steps=` line
# wins — the line is printed every 50 steps with CUMULATIVE averages, so the
# final one is the whole run).
judge() {
    python3 - "$LOGDIR" "$A_TAG" "$B_TAG" "$MIN_DELTA" <<'PY'
import hashlib, json, re, sys

logdir, a_tag, b_tag, min_delta = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])

def steps_of(tag):
    try:
        log = open(f"{logdir}/{tag}.log", errors="replace").read()
    except OSError:
        return None
    line = None
    for l in log.splitlines():
        if "[dspark] steps=" in l and "mean-k=" in l:
            line = l          # last one wins (cumulative)
    if line is None:
        return None
    def f(key):
        k = line.find(key)
        if k < 0:
            return None
        s = line[k + len(key):]
        m = re.match(r"[-+]?[0-9]*\.?[0-9]+", s)
        return float(m.group()) if m else None
    return dict(steps=f("steps="), verify=f("verify="), draft=f("draft="),
                commit=f("commit="), tok_step=f("tok/step="), mean_k=f("mean-k="))

def text_of(tag):
    try:
        c = json.load(open(f"{logdir}/{tag}_r1.json"))["choices"][0]["message"]["content"]
    except Exception:
        return None
    bad = sum(1 for j in range(1, len(c)) if c[j] == c[j - 1] and not c[j].isspace())
    return dict(len=len(c), bad=bad, md5=hashlib.md5(c.encode()).hexdigest())

def faults(tag):
    try:
        return sum(1 for l in open(f"{logdir}/{tag}.log", errors="replace")
                   if "illegal" in l or "fault" in l)
    except OSError:
        return None

A, B = steps_of(a_tag), steps_of(b_tag)
TA, TB = text_of(a_tag), text_of(b_tag)
FA, FB = faults(a_tag), faults(b_tag)

print()
print("================ SH_EXP_MROWS A/B verdict ================")
for tag, s in ((a_tag, A), (b_tag, B)):
    if s is None:
        print(f"  {tag:6s}: NO `[dspark] steps=` line — DSV41_TIMING missing, or <50 spec steps")
    else:
        print(f"  {tag:6s}: steps={s['steps']:.0f} verify={s['verify']:.2f}ms "
              f"draft={s['draft']:.2f}ms commit={s['commit']:.2f}ms tok/step={s['tok_step']:.3f}")
print(f"  faults: {a_tag}={FA}  {b_tag}={FB}")

rc = 0
harness = False
if A is None or B is None:
    print("  VERDICT: harness error — an arm produced no [dspark] line, nothing to compare.")
    harness = True
else:
    dv = A["verify"] - B["verify"]          # >0 => mrows is faster
    print(f"  verify_ms delta = {A['verify']:.2f} - {B['verify']:.2f} = {dv:+.2f}ms "
          f"(threshold {min_delta:+.2f})")
    if TA and TB:
        same = (TA["md5"] == TB["md5"])
        print(f"  text: {a_tag} LEN {TA['len']} 双字 {TA['bad']} md5 {TA['md5'][:12]} | "
              f"{b_tag} LEN {TB['len']} 双字 {TB['bad']} md5 {TB['md5'][:12]}")
        if not same:
            print("  VERDICT: TEXT DRIFT — the arms are not bit-identical. This is a")
            print("           CORRECTNESS bug; do NOT flip DSV41_SH_EXP_MROWS on. Re-run")
            print("           `bash scripts/verify_mrows.sh` first.")
            rc = 1
    else:
        print("  text: one arm has no parseable response — text parity NOT checked")
        harness = True

    if rc == 0 and not harness:
        if dv >= min_delta:
            print(f"  VERDICT: ✅ mrows IS DISPATCHING — verify_ms dropped {dv:.2f}ms,")
            print("           only the multi-row pass can explain it. Text identical.")
            rc = 0
        elif dv > 0:
            print(f"  VERDICT: ⚠️  tiny drop ({dv:.2f}ms < {min_delta:.2f}ms). Likely the")
            print("           SILENT NO-OP is back (or mrows runs but saves less than")
            print("           expected). Check dsv41_kernels.cu:5084+ for a residual")
            print("           `if (g_gemv_a32) return 2;` before trusting the code.")
            rc = 1
        else:
            print(f"  VERDICT: ❌ mrows is SLOWER ({dv:.2f}ms). Not a no-op — a regression.")
            print("           Do not flip the default; bisect the multi-row path.")
            rc = 1

    if rc == 0 and (FA or FB):
        print(f"  VERDICT overridden: fault lines present ({a_tag}={FA} {b_tag}={FB}) —")
        print("           an arm with faults is not a valid measurement.")
        rc = 1

print("=========================================================")
sys.exit(2 if harness else rc)
PY
}

# --------------------------------------------------------------------- main --
mkdir -p "$LOGDIR"
MAIN_LOG="$LOGDIR/summary.log"
echo "== sh_exp_ab.sh $(date -u +%FT%TZ) root=$ROOT port=$PORT reqs=$REQS min_delta=$MIN_DELTA ==" >> "$MAIN_LOG"

if [ "${1:-}" = "--judge-only" ]; then
    judge
    exit $?
fi

[ "${SKIP_PREFLIGHT:-0}" = "1" ] || preflight

run_arm "$A_TAG" DSV41_SH_EXP_MROWS=0 || { log "FATAL: arm $A_TAG failed (baseline); nothing to compare."; exit 2; }
run_arm "$B_TAG" DSV41_SH_EXP_MROWS=1 || { log "FATAL: arm $B_TAG failed (mrows); nothing to compare."; exit 2; }

judge
rc=$?
log "artifacts: $LOGDIR/{$A_TAG,$B_TAG}.log + ${A_TAG}_r*.json + ${B_TAG}_r*.json"
exit "$rc"
