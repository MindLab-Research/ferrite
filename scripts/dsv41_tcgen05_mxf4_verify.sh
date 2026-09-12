#!/usr/bin/env bash
# dsv41_tcgen05_mxf4_verify.sh — the four-step verification sequence for the
# Phase-1 mxf4 tcgen05 swapAB gate/up kernel.
#
# WHAT IS BEING VERIFIED
#   kernels/cuda/dsv41_experts_mxf4.cu:3512-4028 — the mxf4 arm, a COMPILE-TIME
#   #ifdef block (`DSV41_TCGEN05_GATEUP_MXF4_SKELETON`) whose extern "C" entry
#   `dsv41_expert_tcgen05_gate_up_mxf4` (:4007) is additionally behind a
#   PROCESS-STATIC runtime gate (`DSV41_EXPERT_TCGEN05_MXF4`, read once).
#   => A .so built without the macro has NO such symbol (the Rust side probes it:
#   device.rs:3438 `supports_expert_tcgen05_mxf4`), and a .so built with it is
#   inert until the env var is set. Both halves must be handled, in this order.
#
# THE FOUR STEPS (each one gates the next; criterion in brackets)
#   1. compile verify            no GPU  [0 errors, 0 spills, mxf4 MMA spelling]
#   2. .so + binary build        no GPU  [symbol present, pair same-source]
#   3. GPU numerical parity      1 GPU   [see the bars in the harness header]
#   4. serve A/B                 8 GPUs  [four texts + faults=0 + p50]
#
# USAGE
#   scripts/dsv41_tcgen05_mxf4_verify.sh                 # steps 1-3 (safe: no serve)
#   scripts/dsv41_tcgen05_mxf4_verify.sh --all           # 1-4 (needs the idle model box)
#   scripts/dsv41_tcgen05_mxf4_verify.sh --step 3
#   NODE=ubuntu@host scripts/dsv41_tcgen05_mxf4_verify.sh --step 1   # run steps 1/3
#                                          # on a box that has nvcc but not the tree
#
# WHY NODE EXISTS: this source box may have cargo but NO nvcc and NO GPU (step 1
# and step 3 only need nvcc + one GPU, so they are shipped to NODE as TWO files
# and run out of /tmp). Steps 2 and 4 need the whole tree (cargo, the .so, the
# binary, the model dir) and are therefore LOCAL-ONLY — with NODE set they are
# skipped with an explicit message rather than silently mis-run.
#
# ⚠️ THE THREE PITFALLS THIS SCRIPT ENCODES (all three cost a full cycle if a
#    human re-derives them):
#   (a) compile-time gate => the flag must be IN the .so build. build.sh turns it
#       on via DSV41_BUILD_TCGEN05_MXF4 and folds the flag into BUILD_ID, so the
#       same-source gate cannot call a symbol-less .so "matching". EXPORT the
#       variable before step 4: dsv41_serve_ab.sh self-heals a stale pair by
#       re-running build.sh, and an unexported flag would rebuild the .so
#       WITHOUT the skeleton.
#   (b) process-static runtime gate => each A/B ARM MUST BE A SEPARATE PROCESS.
#       dsv41_serve_ab.sh already launches one serve per arm, so this is
#       satisfied — but never try to flip the gate inside one process.
#   (c) THE A/B IS A MEASUREMENT BIAS TRAP UNTIL THE DISPATCH IS WIRED.
#       chain_dev.rs:4249 keeps the routed MoE on the proven GEMV and prints
#       "DSV41_EXPERT_TCGEN05_MXF4 is set, but the routed MoE still dispatches
#       the gate/up to the proven GEMV". While that line appears, BOTH ARMS RUN
#       THE SAME KERNEL: a neutral A/B is EXPECTED and is NOT evidence about the
#       kernel. Step 4 greps that line and says so instead of reporting a
#       meaningless "no win".
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
K="$ROOT/kernels/cuda"
ARCH="${ARCH:-103a}"                 # B300 = sm_103a
NODE="${NODE:-}"                     # e.g. ubuntu@43.202.208.136
CU="${CU:-dsv41_experts_mxf4.cu}"
HARNESS="${HARNESS:-tests_tcgen05_mxf4_gateup.cu}"
STEP=""; SERVE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --all) SERVE=1 ;;
        --step) STEP="${2:?--step needs a number}"; shift ;;
        --step=*) STEP="${1#*=}" ;;
        -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
        # kept for symmetry with the other dsv41_*.sh drivers
        --no-serve) SERVE=0 ;;
        *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

pass=0; fail=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
warn() { echo "  WARN  $*"; }
head1(){ echo; echo "=============================================================="; echo "== $*"; echo "=============================================================="; }

# ---------------------------------------------------------------- target glue
# Local mode: run in $K on this box. NODE mode: the two sources are pushed to
# /tmp/mxf4_verify and every command runs there.
if [ -n "$NODE" ]; then
    TDIR="/tmp/mxf4_verify"
    have_local_nvcc=0
    run()  { ssh -o BatchMode=yes "$NODE" "cd $TDIR && $1"; }
    runq() { ssh -o BatchMode=yes "$NODE" "$1"; }
    sync_src() {
        tar -C "$K" -cf - "$CU" "$HARNESS" | \
            ssh -o BatchMode=yes "$NODE" "mkdir -p $TDIR && tar -C $TDIR -xf -" || return 1
    }
    TARGET="NODE=$NODE ($TDIR)"
else
    run()  { ( cd "$K" && bash -c "$1" ); }
    runq() { bash -c "$1"; }
    sync_src() { return 0; }
    TARGET="local ($K)"
fi

if command -v nvcc >/dev/null 2>&1; then have_local_nvcc=1; else have_local_nvcc=0; fi

echo "== dsv41 tcgen05 mxf4 gate/up — verification sequence =="
echo "   tree   : $ROOT"
echo "   target : $TARGET"
echo "   arch   : sm_$ARCH    steps: ${STEP:-1-4}${SERVE:+ (serve enabled)}"

if [ -z "$NODE" ] && [ "$have_local_nvcc" = "0" ]; then
    echo
    echo "FATAL: no nvcc locally and NODE is unset — steps 1 and 3 need nvcc."
    echo "       Re-run with a GPU/compile box, e.g.:"
    echo "         NODE=ubuntu@43.202.208.136 $0 ${STEP:+--step $STEP}"
    echo "       (steps 2 and 4 must then be run on the box that owns the tree)"
    exit 1
fi
sync_src || { echo "FATAL: could not push the sources to $NODE"; exit 1; }

want() { [ -z "$STEP" ] || [ "$STEP" = "$1" ]; }

# =============================================================================
# STEP 1 — COMPILE VERIFICATION (no GPU, no .so; the state the skeleton is in today)
# =============================================================================
if want 1; then
head1 "STEP 1 — compile verify (nvcc, no GPU)"
CC_CMD="nvcc -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 --use_fast_math -std=c++17 \
-Xcompiler -fPIC -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 -Xptxas -v -c $CU -o /tmp/mxf4.o"
echo "  \$ $CC_CMD"
out="$(run "$CC_CMD" 2>&1)"; rc=$?
errs="$(printf '%s\n' "$out" | grep -c 'error')"
printf '%s\n' "$out" | grep -E 'error|Used [0-9]+ registers|spill' | sed 's/^/    /' | head -20
[ "$rc" = 0 ] && [ "$errs" = 0 ] && ok "skeleton TU compiles clean (0 errors)" \
                                || bad "skeleton TU compile failed (rc=$rc, error lines=$errs)"

# ptxas -v reports one block per __global__; the mxf4 kernel must be 0-spill and
# inside the register ceiling the launcher's occupancy note assumes.
kstats="$(printf '%s\n' "$out" | awk '/Function properties for|Used [0-9]+ registers|spill|stack frame/')"
regs="$(printf '%s\n' "$kstats" | grep -A2 'gateup_mxf4_kernel' | grep -oE 'Used [0-9]+ registers' | grep -oE '[0-9]+' | head -1)"
spills="$(printf '%s\n' "$kstats" | grep -A2 'gateup_mxf4_kernel' | grep -oE '[0-9]+ bytes spill' | grep -oE '^[0-9]+' | head -1)"
[ -n "${spills:-}" ] && [ "${spills:-0}" != 0 ] && bad "mxf4 kernel spills $spills bytes" \
                     || ok "mxf4 kernel 0 spills (registers: ${regs:-see ptxas output})"

# The whole point of the arm is the MMA spelling: if ptxas accepted a different
# kind/scale_vec, the numbers would still be checked by step 3 but the DESIGN
# (2X SF pairing, 128-K pairing contract) would not be the one documented.
sass_cnt="$(run "cuobjdump -sass /tmp/mxf4.o 2>/dev/null | grep -c 'tcgen05.mma'" 2>/dev/null)"
mxf4_cnt="$(run "cuobjdump -sass /tmp/mxf4.o 2>/dev/null | grep -c 'tcgen05.mma.*block_scale'" 2>/dev/null)"
echo "    sass: tcgen05.mma x$sass_cnt (block_scale x$mxf4_cnt)"
[ "${sass_cnt:-0}" -gt 0 ] && ok "tcgen05.mma is present in SASS" \
                           || warn "no tcgen05.mma in SASS — check cuobjdump/arch mismatch"

# Regression guard: the SAME TU without the macro must still compile (the stock
# .so is the fallback every A/B arm compares against).
out2="$(run "nvcc -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 --use_fast_math -std=c++17 \
-Xcompiler -fPIC -c $CU -o /tmp/mxf4_stock.o" 2>&1)"; rc2=$?
[ "$rc2" = 0 ] && ok "stock TU (macro OFF) still compiles" \
               || { bad "stock TU broke (rc=$rc2)"; printf '%s\n' "$out2" | grep -m5 error | sed 's/^/    /'; }

# Step 3's artifact is compiled here (still no GPU needed): a compile failure of
# the harness must be visible in step 1, not 20 minutes into step 3.
out3="$(run "nvcc -gencode arch=compute_${ARCH},code=sm_${ARCH} -O2 -std=c++17 \
-DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 -o /tmp/t_mxf4_gateup $HARNESS" 2>&1)"; rc3=$?
[ "$rc3" = 0 ] && ok "GPU parity harness compiles (/tmp/t_mxf4_gateup)" \
               || { bad "harness does not compile (rc=$rc3)"; printf '%s\n' "$out3" | grep -m8 'error' | sed 's/^/    /'; }
fi

# =============================================================================
# STEP 2 — .so + BINARY (the skeleton must be COMPILED IN, and the pair kept same-source)
# =============================================================================
if want 2; then
head1 "STEP 2 — build the .so WITH the skeleton + re-stamp the binary"
if [ -n "$NODE" ]; then
    warn "skipped: this needs the crate tree + cargo, which NODE mode does not ship"
else
    export DSV41_BUILD_TCGEN05_MXF4=1   # read by build.sh; also inherited by serve_ab's self-heal
    ( cd "$K" && bash build.sh "$ARCH" ) || bad "build.sh $ARCH failed"
    [ -f "$K/libferrite_kernels.so" ] && ok ".so built ($(du -h "$K/libferrite_kernels.so" | cut -f1), build_id $(cat "$K/.build_id" 2>/dev/null))" \
                                      || bad "no .so produced"

    # Symbol probe — this is exactly what the Rust side does (device.rs:3438).
    if nm -D --defined-only "$K/libferrite_kernels.so" 2>/dev/null \
         | grep -q 'dsv41_expert_tcgen05_gate_up_mxf4'; then
        ok "symbol dsv41_expert_tcgen05_gate_up_mxf4 is exported (supports_expert_tcgen05_mxf4() = true)"
    else
        bad "symbol MISSING — the .so was built without DSV41_BUILD_TCGEN05_MXF4=1"
    fi
    # Regression guard: the proven entries must survive the extra block.
    for s in dsv41_expert_gate_up_fp4_batched dsv41_quant_fp4 dsv41_mxf4_test_gemm; do
        nm -D --defined-only "$K/libferrite_kernels.so" 2>/dev/null | grep -q "$s" \
            && ok "stock symbol kept: $s" || bad "stock symbol LOST: $s"
    done

    # The binary must embed the id the .so currently carries (the same-source
    # rule; WHY `grep -cF` and not `-qF`: see dsv41_serve_ab.sh's pre-flight).
    touch "$ROOT/crates/ferrite-kernel/build.rs"   # force build.rs to re-run
    ( cd "$ROOT" && cargo build --release ) || bad "cargo build --release failed"
    BIN="$ROOT/target/release/dsv41-run"
    if [ -f "$BIN" ] && [ -f "$K/.build_id" ] && \
       strings "$BIN" | grep -cF -- "$(cat "$K/.build_id")" >/dev/null; then
        ok "dsv41-run embeds .build_id $(cat "$K/.build_id")"
    else
        bad "dsv41-run does NOT embed the .so's .build_id — rebuild both, in this order"
    fi
fi
fi

# =============================================================================
# STEP 3 — GPU NUMERICAL PARITY (one free GPU; ~12 MB, sub-second)
# =============================================================================
if want 3; then
head1 "STEP 3 — GPU numerical parity (1 GPU)"
# Pick a GPU by IDLE MEMORY, not by assumption: a serve elsewhere on the box may
# own device 0. Never touch a busy device.
GPU="${GPU:-}"
if [ -z "$GPU" ]; then
    GPU="$( ${NODE:+ssh -o BatchMode=yes $NODE }nvidia-smi --query-gpu=index,memory.used \
            --format=csv,noheader,nounits 2>/dev/null \
            | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ' )"
fi
if [ -z "$GPU" ]; then
    bad "no GPU visible on the target"
else
    used="$( ${NODE:+ssh -o BatchMode=yes $NODE }nvidia-smi --query-gpu=memory.used \
              --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ' )"
    echo "  GPU ${GPU} (memory.used=${used:-?} MiB)"
    [ "${used:-0}" -gt 2048 ] && warn "GPU $GPU is not idle (${used} MiB used) — another tenant is on it"

    out="$(run "CUDA_VISIBLE_DEVICES=$GPU /tmp/t_mxf4_gateup" 2>&1)"; rc=$?
    printf '%s\n' "$out" | sed 's/^/    /'
    printf '%s\n' "$out" | grep -q 'RESULT: all cases PASS' \
        && ok "parity suite: all 5 cases PASS (bars: GOLDEN_Q 1e-4 / GEMV 1e-3 / GOLDEN_F32 5e-2, relL2)" \
        || bad "parity suite FAILED (rc=$rc) — see the per-case lines above"

    out="$(run "CUDA_VISIBLE_DEVICES=$GPU /tmp/t_mxf4_gateup --gate-off" 2>&1)"
    printf '%s\n' "$out" | sed 's/^/    /'
    printf '%s\n' "$out" | grep -q 'PASS (real no-op)' \
        && ok "gate OFF is a real no-op (outputs untouched, rc=0)" \
        || bad "the OFF arm is not inert — an A/B would be measuring nothing"

    out="$(run "CUDA_VISIBLE_DEVICES=$GPU /tmp/t_mxf4_gateup --graph" 2>&1)"
    printf '%s\n' "$out" | grep -E 'graph:' | sed 's/^/    /'
    printf '%s\n' "$out" | grep -q 'REPLAY-EXACT' \
        && ok "CUDA-graph capture + replay is bit-identical (serve captures the step)" \
        || warn "graph replay not bit-identical / not capturable — check before any serve A/B"
fi
fi

# =============================================================================
# STEP 4 — SERVE A/B (needs the model box: 8 idle GPUs, the tree, the model dir)
# =============================================================================
if want 4; then
head1 "STEP 4 — serve A/B: DSV41_EXPERT_TCGEN05_MXF4=0 vs =1"
if [ -n "$NODE" ]; then
    warn "skipped: the serve needs the tree, the model and 8 idle GPUs on the SAME host"
elif [ "$SERVE" = 0 ]; then
    warn "skipped: pass --all to run it (it takes ~2 x 1.5 min and owns all 8 GPUs)"
else
    export DSV41_BUILD_TCGEN05_MXF4=1   # MUST be exported: serve_ab self-heals by re-running build.sh
    DRIVER="$HERE/dsv41_serve_ab.sh"
    # Pre-flight: with the symbol missing the ON arm is inert and the A/B is a
    # pure self-consistency check (worst case: reported as a "neutral result").
    if ! nm -D --defined-only "$K/libferrite_kernels.so" 2>/dev/null \
           | grep -q 'dsv41_expert_tcgen05_gate_up_mxf4'; then
        bad "the .so has no tcgen05 mxf4 symbol — run step 2 first (the ON arm would be inert)"
    else
        ok "symbol present in the .so"
        LOG_OFF="/tmp/ab_mxf4_off.log"; LOG_ON="/tmp/ab_mxf4_on.log"
        echo "  --- arm OFF ---"; bash "$DRIVER" mxf4_off DSV41_EXPERT_TCGEN05_MXF4=0
        echo "  --- arm ON  ---"; bash "$DRIVER" mxf4_on  DSV41_EXPERT_TCGEN05_MXF4=1

        # (c) the measurement-bias detector: is the gate actually WIRED to the MoE?
        wired=1
        grep -q "still dispatches" "$LOG_ON" 2>/dev/null && wired=0
        if [ "$wired" = 0 ]; then
            warn "DSV41_EXPERT_TCGEN05_MXF4 is NOT wired into moe() yet (chain_dev.rs:4249): the ON arm"
            warn "dispatched the proven GEMV, so BOTH arms are the same kernel. A neutral result here is"
            warn "EXPECTED and says NOTHING about the tcgen05 kernel. The A/B becomes meaningful only"
            warn "with the Phase-2 indirect launcher (contiguous gate|up pool + e8m0 activation scales"
            warn "+ device-side ids). Treat step 4 today as a build/launch smoke test."
        else
            ok "the ON arm reached the tcgen05 dispatch (no 'still dispatches' warning)"
        fi

        # Text + fault + p50 comparison.
        for f in "$LOG_OFF" "$LOG_ON"; do
            n="$(grep -cE 'illegal|fault' "$f" 2>/dev/null)"
            [ "${n:-1}" = 0 ] && ok "faults=0 in $f" || bad "faults=$n in $f"
        done
        p50() { python3 - "$1" <<'PY'
import re, sys
xs = sorted(float(m.group(1)) for line in open(sys.argv[1])
            for m in [re.search(r"\[dsv41\] step pos=\d+: ([\d.]+)ms", line)] if m)
print(f"{xs[len(xs)//2]:.2f}" if xs else "none")
PY
        }
        P_OFF="$(p50 "$LOG_OFF")"; P_ON="$(p50 "$LOG_ON")"
        echo "  p50: OFF=${P_OFF}ms  ON=${P_ON}ms"
        T_OFF="/tmp/ab_mxf4_off_out.txt"; T_ON="/tmp/ab_mxf4_on_out.txt"
        if [ -f "$T_OFF" ] && [ -f "$T_ON" ]; then
            if diff -q "$T_OFF" "$T_ON" >/dev/null; then
                ok "four prompts character-for-character identical across arms"
            elif [ "$wired" = 0 ]; then
                bad "text differs between two arms that should be the SAME kernel — investigate"
            else
                echo "  prompts differ (expected for a numerics-changing arm); per-prompt diff:"
                diff "$T_OFF" "$T_ON" | head -20 | sed 's/^/      /'
                warn "a differing BODY (not just the first token) is a hard fail per dsv41-methodology"
            fi
        fi
        if [ "$wired" = 1 ] && [ "$P_OFF" != none ] && [ "$P_ON" != none ] && \
           python3 -c "import sys; sys.exit(0 if float('$P_ON') < float('$P_OFF') else 1)"; then
            ok "p50 improved: $P_OFF -> $P_ON ms"
        elif [ "$wired" = 1 ]; then
            warn "p50 did not improve ($P_OFF -> $P_ON ms) — the Phase-1 gate decision (plan §6: \
neutral => flip the default OFF) applies"
        fi
    fi
fi
fi

head1 "SUMMARY"
echo "  $pass pass, $fail fail"
[ "$fail" = 0 ] || exit 1
echo "  next: with steps 1-3 green the kernel is numerically verified; step 4 stays a smoke"
echo "        test until the Phase-2 indirect launcher lands (chain_dev.rs:4249)."
