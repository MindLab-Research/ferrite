#!/usr/bin/env bash
# One-shot post-reboot verification for b300-4, in the order the reboot-recovery
# plan fixes (STATUS.md "恢复计划"):
#
#   0. build BOTH products (the .so FIRST - its build id is baked into the binary)
#   1. sentinel    : one-shot --tp 8 --prompt "1+1=" --max-tokens 3, exit 0 = alive
#   2. base serve  : dsv41_serve_ab.sh base  (~8.2ms = no regression vs round-42;
#                    ~7ms = the accumulated landing all works)
#   3. a32 A/B     : the a32/occupancy bet, isolated serve A/B (p50)
#                      default  = DSV41_GEMV_FP8_MODE=4 DSV41_GEMV_A32=1
#                      noa32    = DSV41_GEMV_A32=0
#                    plus the staging-only arm (MODE=3, a32 kept) for contrast.
#   4. (optional) isolated kernel bench: scripts/dsv41_a32_bench.sh
#
# WHY the A/B is DSV41_GEMV_A32 and NOT DSV41_GEMV_FP8_MODE: mode 3 and mode 4
# BOTH build the a32 table (`s_af`); they differ only in whether the fp8
# ACTIVATION is staged in smem (mode 4) or re-read from global (mode 3). The
# 20 KB a32 table has its own gate, added for this experiment. See the header of
# scripts/dsv41_a32_bench.sh for the full reading of the modes.
#
# Usage:  scripts/dsv41_recovery_verify.sh            # phases 0-3
#         PHASES="1 2 3" scripts/dsv41_recovery_verify.sh
#         PHASES="3" SKIP_BUILD=1 scripts/dsv41_recovery_verify.sh   # just the A/B
#         RUN_KERNEL_BENCH=1 scripts/dsv41_recovery_verify.sh         # + phase 4
#
# Every phase is a separate serve (the gates are read once per process, so an arm
# can only change by restarting). Fail-fast: a phase that dies stops the run.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
K="$ROOT/kernels/cuda"
PHASES="${PHASES:-0 1 2 3}"
ARCH="${ARCH:-103a}"   # B300 = sm_103a. 100a is the stale README value (AGENTS.md)
PORT="${DSV41_PORT:-8090}"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
BIN="$ROOT/target/release/dsv41-run"
LOG="/tmp/recovery_verify.log"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

have() { command -v "$1" >/dev/null 2>&1; }

kill_serves() {
    for p in $(pgrep -x dsv41-run); do kill -9 "$p"; done
    sleep 8
}

# Post-arm gate: an arm is only usable if (a) every prompt returned a body and
# (b) the serve logged no fault lines. serve_ab.sh always exits 0, so the exit
# status alone cannot tell a healthy arm from a wedged one.
ab_gate() {
    local tag="$1" logf="/tmp/ab_${tag}.log" outf="/tmp/ab_${tag}_out.txt"
    local f
    [ -s "$outf" ] || { log "  ab_gate[$tag]: no output file $outf"; return 1; }
    if grep -q "(failed)" "$outf" 2>/dev/null; then
        log "  ab_gate[$tag]: at least one prompt FAILED (empty body / timeout)"; return 1
    fi
    f="$(grep -cE 'illegal|fault' "$logf" 2>/dev/null || echo 0)"
    if [ "${f:-0}" != "0" ]; then
        log "  ab_gate[$tag]: WARNING $f fault-looking lines in $logf - inspect before trusting the numbers"
        return 1
    fi
    return 0
}

for phase in $PHASES; do
    case "$phase" in
    0)
        log "== phase 0: build both products =="
        if [ "${SKIP_BUILD:-0}" != "1" ]; then
            have nvcc || { log "FATAL: nvcc not found (CUDA toolkit required, no GPU needed)"; exit 1; }
            ( cd "$K" && bash build.sh "$ARCH" ) || { log "FATAL: kernel build failed"; exit 1; }
            log "kernel .so: $(cat "$K/.build_id" 2>/dev/null) $(ls -l "$K/libferrite_kernels.so" | awk '{print $5}') bytes"
            # the binary embeds .build_id through ferrite-kernel/build.rs, so it
            # MUST be built after the .so - otherwise the id check refuses to start.
            # shellcheck disable=SC1090
            source "$HOME/.cargo/env" 2>/dev/null || true
            ( cd "$ROOT" && cargo build --release ) || { log "FATAL: cargo build failed"; exit 1; }
            log "binary: $BIN"
        else
            log "SKIP_BUILD=1: reusing $K/libferrite_kernels.so + $BIN"
        fi
        [ -f "$K/libferrite_kernels.so" ] || { log "FATAL: missing libferrite_kernels.so"; exit 1; }
        [ -x "$BIN" ] || { log "FATAL: missing $BIN"; exit 1; }
        have nvidia-smi || { log "FATAL: nvidia-smi not found - this is not the GPU node"; exit 1; }
        log "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
        ;;

    1)
        log "== phase 1: sentinel (one-shot, exit 0 = environment alive) =="
        kill_serves
        set +e
        timeout "${SENTINEL_TIMEOUT:-300}" \
            env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            DSV41_MODEL_DIR="$MODEL_DIR" \
            DSV41_KERNELS="$K/libferrite_kernels.so" \
            DSV41_TIMING=1 \
            "$BIN" --tp 8 --prompt "1+1=" --max-tokens 3 >/tmp/recovery_sentinel.log 2>&1
        rc=$?
        set -e
        log "sentinel exit=$rc (tail)"
        tail -5 /tmp/recovery_sentinel.log | sed 's/^/    /' | tee -a "$LOG"
        if [ "$rc" != "0" ]; then
            log "SENTINEL FAILED (exit=$rc) - the node is not recovered; do NOT continue to serve A/B."
            log "  rc=124 => hung, not crashed (GPU wedge / driver stall). Other rc => see the log."
            log "  see /tmp/recovery_sentinel.log and dmesg | tail for Xid / refcnt refs"
            exit 1
        fi
        log "sentinel OK"
        ;;

    2)
        log "== phase 2: base serve (round-42 regression gate) =="
        log "   expected p50: ~8.2ms = no regression vs round-42; ~7ms = all landings work"
        bash "$ROOT/scripts/dsv41_serve_ab.sh" base 2>&1 | tee -a "$LOG"
        ab_gate base || { log "FATAL: base arm failed. Next: run the round-41-equivalent gate set"
            log "  (POST_RECOVERY_COMMANDS.md step 2) to confirm the tree, then bisect by risk group."
            exit 1; }
        ;;

    3)
        log "== phase 3: a32/occupancy serve A/B (p50) =="
        log "   arm m4_a32 (MODE=4 A32=1, explicit) vs arm m4_noa32 (MODE=4 A32=0)"
        DSV41_PORT="$PORT" bash "$ROOT/scripts/dsv41_serve_ab.sh" m4_a32 DSV41_GEMV_FP8_MODE=4 DSV41_GEMV_A32=1 2>&1 | tee -a "$LOG"
        ab_gate m4_a32 || { log "FATAL: default arm failed - stop, the tree is not healthy"; exit 1; }
        DSV41_PORT="$PORT" bash "$ROOT/scripts/dsv41_serve_ab.sh" m4_noa32 DSV41_GEMV_FP8_MODE=4 DSV41_GEMV_A32=0 2>&1 | tee -a "$LOG"
        ab_gate m4_noa32 || { log "FATAL: noa32 arm failed - not a valid A/B, do NOT change the default"; exit 1; }
        log "   contrast arm: mode 3 (activation staging off, a32 kept)"
        DSV41_PORT="$PORT" bash "$ROOT/scripts/dsv41_serve_ab.sh" m3_a32 DSV41_GEMV_FP8_MODE=3 2>&1 | tee -a "$LOG"
        ab_gate m3_a32 || log "WARN: contrast arm m3_a32 failed (non-fatal, A/B verdict still valid)"
        log "   => compare the p50 lines; the four-prompt texts must be identical"
        log "      (a32=0 is bit-identical to a32=1 by construction). Any text drift"
        log "      INVALIDATES the arm - do not flip DSV41_GEMV_A32 on a drifted run."
        ;;

    *)
        log "unknown phase '$phase' (0|1|2|3)"; exit 2
        ;;
    esac
done

if [ "${RUN_KERNEL_BENCH:-0}" = "1" ]; then
    log "== phase 4: isolated kernel bench =="
    ( cd "$ROOT" && bash scripts/dsv41_a32_bench.sh ) 2>&1 | tee -a "$LOG"
fi

log "done. full log: $LOG"
