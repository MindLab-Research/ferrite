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
#   3b. PDL A/B    : DSV41_PDL=1 (DEFAULT ON) vs DSV41_PDL=0, one variable only,
#                    both arms pinned to the a32 config phase 3 picked as the
#                    winner (see PDL_BASE_MODE / PDL_BASE_A32 below).
#                    WHY this is a recovery gate: DSV41_PDL defaults ON, so
#                    EVERY serve after this rebuild runs the PDL path --
#                    cudaLaunchKernelEx + cudaLaunchAttributeProgrammatic-
#                    StreamSerialization + a consumer-side
#                    cudaGridDependencySynchronize() gating the first read of
#                    the producer's output. Capture COMPATIBILITY was already
#                    shown by the in-repo microbench ferrite_pdl_exp(mode=3)
#                    (A->B chains, captured, 200 replays, equal checksums --
#                    crates/ferrite-kernel/tests/gpu_smoke.rs:1273), but the
#                    PRODUCTION graph has never run on this node. If the
#                    captured/replayed launch behaves differently from the
#                    plain one, the consumer reads the producer too early and
#                    the output is silently WRONG -- a fault counter will not
#                    always catch that, so the verdict is TEXT IDENTITY first,
#                    p50 second.
#                    COVERED (11 launch points, two TUs, one gate): attention
#                    projection chain 8 points (dsv41_kernels.cu:2665
#                    dsv41_pdl_or_plain) + expert chain 3 points
#                    (dsv41_experts_mxf4.cu:873 dsv41_experts_pdl_or_plain,
#                    called from the batched gate-up / down / down-reduce
#                    launchers -- 5 call sites, 3 launch points).
#                    NOT covered: M>1 gemm_fp8_kernel tiles, the split/merge/
#                    warp attention variants, the *_indirect per-slot expert
#                    entries -- plain launches in both arms, so they cannot
#                    invalidate this A/B.
#   4. (optional) isolated kernel bench: scripts/dsv41_a32_bench.sh
#
# WHY the A/B is DSV41_GEMV_A32 and NOT DSV41_GEMV_FP8_MODE: mode 3 and mode 4
# BOTH build the a32 table (`s_af`); they differ only in whether the fp8
# ACTIVATION is staged in smem (mode 4) or re-read from global (mode 3). The
# 20 KB a32 table has its own gate, added for this experiment. See the header of
# scripts/dsv41_a32_bench.sh for the full reading of the modes.
#
# Usage:  scripts/dsv41_recovery_verify.sh            # phases 0-1-2-3-3b
#         PHASES="1 2 3 3b" scripts/dsv41_recovery_verify.sh
#         PHASES="3" SKIP_BUILD=1 scripts/dsv41_recovery_verify.sh   # just the a32 A/B
#         PHASES="3b" SKIP_BUILD=1 scripts/dsv41_recovery_verify.sh  # just the PDL A/B
#         RUN_KERNEL_BENCH=1 scripts/dsv41_recovery_verify.sh         # + phase 4
#
# Phase 3b knobs:
#   PDL_BASE_MODE / PDL_BASE_A32   the a32 config BOTH 3b arms are pinned to
#                                  (default 4 / 1 = phase 3's default arm and
#                                  today's shipped default). If phase 3 says
#                                  noa32 wins, re-run 3b with PDL_BASE_A32=0 --
#                                  do NOT let 3b vary two things at once.
#   PDL_REUSE_CONTROL=1            reuse phase-3's m4_a32 serve as the PDL=1
#                                  control instead of paying for a third serve
#                                  with the identical env. Default off: 3b is
#                                  self-contained so PHASES="3b" alone works.
#
# Every phase is a separate serve (the gates are read once per process, so an arm
# can only change by restarting). Fail-fast: a phase that dies stops the run.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
K="$ROOT/kernels/cuda"
PHASES="${PHASES:-0 1 2 3 3b}"
ARCH="${ARCH:-103a}"   # B300 = sm_103a. 100a is the stale README value (AGENTS.md)
PORT="${DSV41_PORT:-8090}"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
BIN="$ROOT/target/release/dsv41-run"
LOG="/tmp/recovery_verify.log"
AB_PROMPTS="${AB_PROMPTS:-4}"   # must match the prompt list in dsv41_serve_ab.sh

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

have() { command -v "$1" >/dev/null 2>&1; }

kill_serves() {
    for p in $(pgrep -x dsv41-run); do kill -9 "$p"; done
    sleep 8
}

# Post-arm gate: an arm is only usable if (a) every prompt returned a body and
# (b) the serve logged no fault lines. serve_ab.sh always exits 0, so the exit
# status alone cannot tell a healthy arm from a wedged one.
#
# AB_PROMPTS must match the prompt list length in dsv41_serve_ab.sh.
ab_gate() {
    # NOTE: one local per assignment. `local tag="$1" logf="...${tag}..."` looks
    # fine but bash expands EVERY word of the command before the builtin runs,
    # so under `set -u` the second assignment reads an unset $tag and the shell
    # dies with "unbound variable" -- no gate, no message, the phase just stops.
    local tag="$1"
    local logf="/tmp/ab_${tag}.log"
    local outf="/tmp/ab_${tag}_out.txt"
    local f n
    [ -s "$outf" ] || { log "  ab_gate[$tag]: no output file $outf"; return 1; }
    [ -s "$logf" ] || { log "  ab_gate[$tag]: no serve log $logf"; return 1; }
    # Count DECODED answer bodies. The old `grep "(failed)"` could never fire:
    # serve_ab.sh echoes "(failed)" to stdout, which the phase-level tee sends to
    # $LOG -- it is never written to $outf. A short arm must fail here, because
    # an empty/truncated body is indistinguishable from a fast arm downstream.
    n="$(ab_texts "$tag" | grep -c -v '^<')"
    if [ "${n:-0}" != "$AB_PROMPTS" ]; then
        log "  ab_gate[$tag]: only ${n:-0}/$AB_PROMPTS prompts returned a usable body (see $outf)"
        return 1
    fi
    f="$(grep -cE 'illegal|fault' "$logf" 2>/dev/null)"
    # ⚠️ NEVER write `grep -c ... || echo 0` here: grep -c PRINTS 0 and exits 1
    # when nothing matches, so the `||` appends a SECOND "0" and f becomes the
    # two-line string "0\n0" -- which != "0", i.e. a perfectly clean arm fails
    # the gate. ($logf existence is checked above, so f="" cannot mean a missing
    # file.) The git version of this line had exactly that bug.
    f="${f:-0}"
    if [ "${f:-0}" != "0" ]; then
        log "  ab_gate[$tag]: WARNING $f fault-looking lines in $logf - inspect before trusting the numbers"
        return 1
    fi
    return 0
}

# p50 (ms) of one arm's per-step latencies. Same basis as dsv41_serve_ab.sh's
# printer -- the "[dsv41] step pos=N: Xms" lines, never a segment average --
# so a number produced here is directly comparable with the one that driver
# already printed for the arm.
ab_p50() {
    python3 - "/tmp/ab_$1.log" <<'PY'
import re, sys
xs = sorted(float(m.group(1)) for line in open(sys.argv[1])
            for m in [re.search(r"\[dsv41\] step pos=\d+: ([\d.]+)ms", line)] if m)
print(f"{xs[len(xs) // 2]:.2f}" if xs else "nan")
PY
}

# The four answer texts of one arm, one per line, in prompt order.
# dsv41_serve_ab.sh tees the raw bodies to /tmp/ab_<tag>_out.txt and curl writes
# no trailing newline, so the file is usually ONE line holding four JSON objects
# -- hence raw_decode in a loop instead of one json.loads per line. Only the
# content field is printed: the response envelope carries timestamps/id, which
# would differ between two runs of the SAME config and turn a text diff into
# noise.
ab_texts() {
    python3 - "/tmp/ab_$1_out.txt" <<'PY'
import json, sys
s = open(sys.argv[1]).read()
dec, i, out = json.JSONDecoder(), 0, []
while i < len(s):
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    if i >= len(s):
        break
    try:
        obj, i = dec.raw_decode(s, i)
    except ValueError:
        out.append("<unparsable: %s>" % s[i:i + 40])
        break
    try:
        out.append(obj["choices"][0]["message"]["content"].replace("\n", " "))
    except (KeyError, IndexError, TypeError):
        out.append("<no-content>")
print("\n".join(out))
PY
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

    3b)
        log "== phase 3b: PDL serve A/B (DSV41_PDL=1 default ON vs =0) =="
        # ONE variable. Both arms carry the SAME a32 config: the phase-3 winner,
        # because PDL and a32 are independent knobs and varying both at once
        # would make a p50 delta unassignable. Default 4/1 = phase 3's default
        # arm AND today's shipped default, so arm pdl_on is env-identical to
        # base / m4_a32 by construction; if phase 3 lands on noa32 instead,
        # re-run 3b with PDL_BASE_A32=0.
        pdl_mode="${PDL_BASE_MODE:-4}"
        pdl_a32="${PDL_BASE_A32:-1}"
        log "   a32 config pinned for BOTH arms: DSV41_GEMV_FP8_MODE=$pdl_mode DSV41_GEMV_A32=$pdl_a32"
        log "   (if phase 3 picked noa32, re-run: PDL_BASE_MODE=4 PDL_BASE_A32=0 PHASES=3b ...)"

        on_tag=pdl_on
        if [ "${PDL_REUSE_CONTROL:-0}" = "1" ] && [ -s /tmp/ab_m4_a32.log ]; then
            # Cheap path: phase 3 already paid for a serve with exactly this
            # env. Only the same-run artifacts under /tmp are reusable -- if the
            # log is from an older binary the control is WRONG, so this is
            # opt-in and never the default.
            on_tag=m4_a32
            log "   PDL_REUSE_CONTROL=1: reusing phase-3 arm m4_a32 as the PDL=1 control"
            log "     (valid only if that serve ran in THIS session, same build)"
        else
            DSV41_PORT="$PORT" bash "$ROOT/scripts/dsv41_serve_ab.sh" "$on_tag" \
                DSV41_GEMV_FP8_MODE="$pdl_mode" DSV41_GEMV_A32="$pdl_a32" DSV41_PDL=1 2>&1 | tee -a "$LOG"
        fi
        ab_gate "$on_tag" || { log "FATAL: the PDL=1 arm (DEFAULT ON) is not healthy."
            log "  This is the exact risk the recovery gate exists for: the PDL path is"
            log "  what every serve runs by default. Next: rebuild, then re-run"
            log "  PHASES=3b SKIP_BUILD=1 ... and read /tmp/ab_${on_tag}.log around the"
            log "  first fault/illegal line. If it reproduces, do NOT ship the default"
            log "  as-is - set the default to OFF (or fix the missing"
            log "  cudaGridDependencySynchronize at the covered launch points)."
            exit 1; }

        DSV41_PORT="$PORT" bash "$ROOT/scripts/dsv41_serve_ab.sh" pdl_off \
            DSV41_GEMV_FP8_MODE="$pdl_mode" DSV41_GEMV_A32="$pdl_a32" DSV41_PDL=0 2>&1 | tee -a "$LOG"
        ab_gate pdl_off || { log "FATAL: the DSV41_PDL=0 fallback arm failed."
            log "  Not a valid A/B. PDL=0 is the plain-launch rollback, so a broken"
            log "  rollback is a tree/environment problem, not a PDL verdict."
            exit 1; }

        # Verdict 1 (primary): text identity. PDL moves launch timing only, so
        # the two arms must produce byte-identical answers. A drift means the
        # consumer read the producer's output before cudaGridDependencySynchronize
        # -- real corruption, and the reason this phase exists.
        ab_texts "$on_tag" >/tmp/ab_pdl_on_text.txt
        ab_texts pdl_off  >/tmp/ab_pdl_off_text.txt
        n_on="$(grep -c . /tmp/ab_pdl_on_text.txt || true)"
        n_off="$(grep -c . /tmp/ab_pdl_off_text.txt || true)"
        if [ "${n_on:-0}" != "4" ] || [ "${n_off:-0}" != "4" ]; then
            log "   FATAL: expected 4 answer texts per arm, got ${n_on:-0}/${n_off:-0}"
            log "     (the prompt list in dsv41_serve_ab.sh is 4 entries; a short arm"
            log "      means a body was dropped/timeout'd and the diff below is moot)"
            exit 1
        fi
        if diff -q /tmp/ab_pdl_on_text.txt /tmp/ab_pdl_off_text.txt >/dev/null; then
            log "   texts IDENTICAL across the two arms (expected: PDL never changes values)"
        else
            log "   FATAL: TEXT DRIFT between $on_tag and pdl_off:"
            diff -u /tmp/ab_pdl_on_text.txt /tmp/ab_pdl_off_text.txt | sed 's/^/    /' | tee -a "$LOG"
            log "     The PDL launch changed a VALUE => a consumer read the producer"
            log "     early in the production graph. DO NOT keep DSV41_PDL default ON."
            exit 1
        fi

        # Verdict 2 (secondary): p50. PDL only wins back node-transition cost,
        # so a flat p50 is a perfectly good outcome (correctness was the point);
        # act only on a consistent >2% delta.
        pdl_p_on="$(ab_p50 "$on_tag")"
        pdl_p_off="$(ab_p50 pdl_off)"
        log "   p50: ${on_tag}=${pdl_p_on}ms  pdl_off=${pdl_p_off}ms"
        python3 - "$pdl_p_on" "$pdl_p_off" "$on_tag" <<'PY' | tee -a "$LOG"
import sys
on, off, tag = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
if on != on or off != off:          # nan on either side -> no step lines
    print("   p50 delta: not computable (an arm logged no step lines)")
else:
    d = (off - on) / on * 100.0
    print(f"   p50 delta (pdl_off - {tag}) = {d:+.1f}%")
    if abs(d) <= 2.0:
        print("   => within noise: PDL is neutral here. Correctness is the verdict;")
        print("      keeping the default ON is fine (or OFF - cost the same).")
    elif d < 0:
        print("   => PDL OFF is faster by >2% on this node: the default ON is a cost.")
        print("      Re-check with a second run before flipping the default (=0 rolls back).")
    else:
        print("   => PDL ON is faster by >2%: the node-transition win is real.")
PY
        log "   NOTE: one node, four prompts. This A/B decides CORRECTNESS, not a"
        log "      publishable speedup - re-run before any default change."
        ;;

    *)
        log "unknown phase '$phase' (0|1|2|3|3b)"; exit 2
        ;;
    esac
done

if [ "${RUN_KERNEL_BENCH:-0}" = "1" ]; then
    log "== phase 4: isolated kernel bench =="
    ( cd "$ROOT" && bash scripts/dsv41_a32_bench.sh ) 2>&1 | tee -a "$LOG"
fi

log "done. full log: $LOG"
