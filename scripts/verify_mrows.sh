#!/usr/bin/env bash
# verify_mrows.sh — the ONE command for the multi-row (grid.z) kernel regression.
#
# WHAT IT RUNS (bit-exact acceptance suites; each prints "RESULT: ..." and exits
# nonzero on failure — that exit code is the gate this script propagates)
#   kernels/cuda/tests_dsv41_gemm_mrows.cu      multi-row GEMM: an m-row call is
#                                               BIT-IDENTICAL to m single-row calls
#   kernels/cuda/tests_dsv41_experts_mrows.cu   multi-row batched expert chain
#                                               (gate/up -> swiglu -> down+reduce)
#   kernels/cuda/tests_dsv41_head_mrows.cu      the verify head's multi-row GEMV:
#                                               an m-row call is BIT-IDENTICAL to
#                                               m single-row ferrite_gemv_bf16_v2
#                                               (nrows=1) launches and to one
#                                               ferrite_gemv_bf16_nt (nrows=m)
#
# WHY THE SOURCES ARE SHIPPED TO A REMOTE NODE
#   Compiling these suites needs nvcc; running them needs ONE free GPU. This
#   source box may have neither (the project's .so is normally built on the model
#   box). So the harness copies kernels/cuda/ to NODE:$WORK and compiles+runs
#   THERE — it never touches the live tree ($HOME/ferrite) and never needs the
#   .so or the binary. NODE="-" runs everything locally (for a box with nvcc).
#
# FLAGS ARE COPIED FROM kernels/cuda/build.sh (ARCH=103a) ON PURPOSE
#   The kernels under test are compiled into the production .so with a specific
#   flag set; a regression harness that used a different one would accept code the
#   .so cannot run. Copied verbatim: -O3, --use_fast_math (OFF only when
#   FERRITE_NO_FAST_MATH is set, exactly as build.sh), -std=c++17 and
#   -gencode arch=compute_103a,code=sm_103a. DROPPED: -shared/-Xcompiler -fPIC
#   (they describe a shared object; these targets are executables with a main())
#   and -DFERRITE_KERNEL_BUILD_ID (ferrite_kernels.cu defaults it to "unstamped"
#   and no test asserts on it).
#
# USAGE
#   bash scripts/verify_mrows.sh                  # both suites, full, remote
#   bash scripts/verify_mrows.sh --quick          # fused arms only (forwarded)
#   bash scripts/verify_mrows.sh --test head     # one suite only (gemm|experts|head)
#   MROWS_GPU=5 bash scripts/verify_mrows.sh      # pin the GPU (default: the
#                                                 #   most idle one, auto-picked)
#   MROWS_ENV="DSV41_GATEUP_FUSE=0" bash scripts/verify_mrows.sh
#                                                 # extra env for the RUN step
#   NODE=ubuntu@host bash scripts/verify_mrows.sh # another node
#   NODE=- bash scripts/verify_mrows.sh           # local (needs nvcc + GPU)
#   MROWS_COMPILE_ONLY=1 bash scripts/verify_mrows.sh
#                                                 # compile both suites, run
#                                                 #   nothing (nvcc needs NO GPU —
#                                                 #   safe while the box serves)
#
# OUTPUT
#   one row per suite: compile status, exit code, parsed RESULT, FAIL/SKIP counts
#   and a PASS/FAIL verdict, then the log paths (local /tmp/verify_mrows/).
#   Exit code: 0 = every suite PASSed, 1 = a suite FAILED, 2 = harness error
#   (missing source / sync or compile failure => the number above is NOT evidence
#   about the kernels, which is why it is distinguished from 1).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
K="$ROOT/kernels/cuda"

NODE="${NODE:-ubuntu@43.202.208.136}"
ARCH="${ARCH:-103a}"          # build.sh default 100a is stale; B300 = sm_103a
WORK="${WORK:-/tmp/ferrite_mrows}"   # remote scratch (we own it; --delete is safe)
LOGDIR="${MROWS_LOGDIR:-/tmp/verify_mrows}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)

# The suites, and the extra translation units each one needs. The expert suite's
# header documents its own set ("the expert TU is INCLUDED; dsv41_glue.cu is a
# second input file, and dsv41_kernels.cu a third because dsv41_glue.cu's entry
# points reference the fp8 quantise kernel"). The gemm suite's dependency set is
# not fixed here: the list is tried IN ORDER and the first one that compiles wins,
# so a suite that includes its TU keeps working as-is and one that links it is
# covered too. A candidate that duplicates extern "C" symbols simply fails to
# compile and the next one is tried.
TESTS=(gemm experts head rope gate)
extra_sources() {
    case "$1" in
        gemm)
            # Order = fewest inputs first (the suite may #include its TU).
            printf '%s\n' '' 'dsv41_kernels.cu' 'dsv41_kernels.cu dsv41_glue.cu'
            ;;
        experts)
            printf '%s\n' 'dsv41_glue.cu dsv41_kernels.cu'
            ;;
        head)
            # The suite #includes dsv41_glue.cu (the kernel under test) and links
            # ferrite_kernels.cu as the SECOND TU — the reference program
            # (ferrite_gemv_bf16_v2 / _nt) lives there, and this is exactly the
            # two-TU split the production .so is linked with.
            printf '%s\n' 'ferrite_kernels.cu'
            ;;
        rope)
            # The row-fold rope suite #includes dsv41_kernels.cu (the kernel under
            # test, exactly how tests_dsv41_attn.cu builds) -> no extra TU.
            printf '%s\n' ''
            ;;
        gate)
            # The MoE-gate row fold links ferrite_kernels.cu as the second TU for
            # the reference program (ferrite_gemv_bf16_v2 / _nt) — the same split
            # the head suite uses, and the same split the .so is linked with.
            printf '%s\n' 'ferrite_kernels.cu'
            ;;
    esac
}
test_file() {
    case "$1" in
        gemm)    echo "tests_dsv41_gemm_mrows.cu" ;;
        experts) echo "tests_dsv41_experts_mrows.cu" ;;
        head)    echo "tests_dsv41_head_mrows.cu" ;;
        rope)    echo "tests_rope_fold.cu" ;;
        gate)    echo "tests_gate_mrows.cu" ;;
    esac
}

FORWARD=()          # extra argv for the suite (--quick)
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) FORWARD+=(--quick) ;;
        --test)  ONLY="${2:?--test needs gemm|experts}"; shift ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)"; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Flags, copied from build.sh (see the header for what was dropped and why).
# ---------------------------------------------------------------------------
FAST_MATH_FLAG="--use_fast_math"
if [ -n "${FERRITE_NO_FAST_MATH:-}" ]; then FAST_MATH_FLAG=""; fi
NVCC_FLAGS=(-O3 $FAST_MATH_FLAG -std=c++17 -gencode "arch=compute_${ARCH},code=sm_${ARCH}")

# Every remote command goes through here, so BatchMode is impossible to forget
# and a dead node fails fast instead of halfway through the suite.
rcmd() { if [ "$NODE" = "-" ]; then bash -c "$1"; else ssh "${SSH_OPTS[@]}" "$NODE" "$1"; fi; }
rsync_to() {
    local src="$1" dst="$2"
    if [ "$NODE" = "-" ]; then mkdir -p "$dst" && cp -a "$src/." "$dst/"; else
        rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "$src/" "$NODE:$dst/"
    fi
}

mkdir -p "$LOGDIR"

# ---------------------------------------------------------------------------
# 1. Pre-flight: the sources must exist. A missing suite is a HARNESS error, not
#    a kernel failure — say so instead of reporting a green run from a suite that
#    never compiled.
# ---------------------------------------------------------------------------
echo "== ferrite multi-row regression (node=$NODE arch=$ARCH) =="
missing=()
for t in "${TESTS[@]}"; do
    [ "$ONLY" ] && [ "$ONLY" != "$t" ] && continue
    f="$(test_file "$t")"
    [ -f "$K/$f" ] || missing+=("$f")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "FATAL: missing suite source(s) under kernels/cuda/: ${missing[*]}"
    echo "       (expected the multi-row harnesses; nothing was compiled or run)"
    exit 2
fi

# ---------------------------------------------------------------------------
# 2. Ship the kernel sources to the node. The whole kernels/cuda dir goes over
#    (the suites #include their TUs and reference the fp8/glue launchers), so a
#    suite can never silently compile against a stale sibling.
# ---------------------------------------------------------------------------
if [ -n "${SKIP_SYNC:-}" ]; then
    echo "-- sync skipped (SKIP_SYNC=1)"
else
    echo "-- syncing kernels/cuda -> $NODE:$WORK"
    rcmd "mkdir -p $WORK" || { echo "FATAL: cannot reach $NODE"; exit 2; }
    rsync_to "$K" "$WORK" || { echo "FATAL: rsync to $NODE:$WORK failed"; exit 2; }
fi

# One free GPU for the run steps; pick the least-used one so a co-tenant
# (e.g. a serve on the other 8-slice) is not disturbed. MROWS_GPU overrides.
if [ -z "${MROWS_GPU:-}" ] && [ "$NODE" != "-" ]; then
    MROWS_GPU="$(rcmd "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -n | head -1 | cut -d, -f1" 2>/dev/null)"
    MROWS_GPU="${MROWS_GPU:-0}"
fi
MROWS_GPU="${MROWS_GPU:-0}"
echo "-- GPU: $MROWS_GPU   run env: ${MROWS_ENV:-<none>}   argv: ${FORWARD[*]:-<none>}"
echo

# ---------------------------------------------------------------------------
# 3. Compile + run + parse, one suite at a time.
# ---------------------------------------------------------------------------
declare -a R_NAME R_COMPILE R_RC R_RESULT R_FAILS R_SKIPS R_VERDICT
overall=0

for t in "${TESTS[@]}"; do
    [ "$ONLY" ] && [ "$ONLY" != "$t" ] && continue
    f="$(test_file "$t")"
    bin="t_${t}_mrows"
    clog="$LOGDIR/${t}.compile.log"
    rlog="$LOGDIR/${t}.run.log"

    echo "----------------------------------------------------------------"
    echo "### $f"
    compiled=0
    while IFS= read -r extra; do
        # shellcheck disable=SC2086 # $extra is a deliberate space-separated list
        cmd="cd $WORK && nvcc ${NVCC_FLAGS[*]} -o $WORK/$bin $f $extra"
        if rcmd "$cmd" >"$clog" 2>&1; then
            echo "  compile OK   (extra TU: ${extra:-<none>})"
            compiled=1
            break
        fi
        echo "  compile retry without/with another TU set (extra: ${extra:-<none>}) — see $clog"
    done < <(extra_sources "$t")
    if [ "$compiled" != 1 ]; then
        echo "  COMPILE FAILED — $f did not build with any candidate TU set; log: $clog"
        tail -5 "$clog" | sed 's/^/    | /'
        R_NAME+=("$f"); R_COMPILE+=("FAIL"); R_RC+=("-"); R_RESULT+=("(not built)")
        R_FAILS+=("-"); R_SKIPS+=("-"); R_VERDICT+=("HARNESS")
        overall=2
        echo
        continue
    fi

    if [ -n "${MROWS_COMPILE_ONLY:-}" ]; then
        # nvcc needs no GPU, so a compile-only pass is safe while the box serves.
        R_NAME+=("$f"); R_COMPILE+=("OK"); R_RC+=("-"); R_RESULT+=("(compile only)")
        R_FAILS+=("-"); R_SKIPS+=("-"); R_VERDICT+=("COMPILED")
        echo "  run skipped (MROWS_COMPILE_ONLY=1)"
        echo
        continue
    fi

    # The run env: the suites read their knobs once per process, so MROWS_ENV is
    # applied to the RUN (not to the compile) — a process-static gate must be a
    # process decision. `timeout` bounds a wedged CUDA launch.
    rcmd "cd $WORK && CUDA_VISIBLE_DEVICES=$MROWS_GPU ${MROWS_ENV:-} timeout 900 ./$bin ${FORWARD[*]:-} " \
        >"$rlog" 2>&1
    rc=$?

    # Parse. The contract is a final "RESULT: all checks passed" / "RESULT: N
    # check(s) FAILED"; the exit code is authoritative and the text is cross-
    # checked against it (a crash produces neither, and must not read as PASS).
    result="$(grep -m1 '^RESULT:' "$rlog" || true)"
    fails="$(grep -c 'FAIL' "$rlog" || true)"
    skips="$(grep -c 'SKIP' "$rlog" || true)"
    verdict="PASS"
    if [ "$rc" -ne 0 ] || [ -z "$result" ] || echo "$result" | grep -q 'FAILED'; then
        verdict="FAIL"
        overall=1
    fi

    echo "  run rc=$rc   $result"
    echo "  FAIL lines: $fails   SKIP lines: $skips"
    [ "$verdict" = FAIL ] && tail -8 "$rlog" | sed 's/^/    | /'

    R_NAME+=("$f"); R_COMPILE+=("OK"); R_RC+=("$rc")
    R_RESULT+=("${result:-<no RESULT line>}"); R_FAILS+=("$fails"); R_SKIPS+=("$skips")
    R_VERDICT+=("$verdict")
    echo
done

# ---------------------------------------------------------------------------
# 4. The table. FAIL/HARNESS both exit nonzero; only FAIL is evidence about the
#    kernels.
# ---------------------------------------------------------------------------
printf '%-38s %-8s %-4s %-26s %-6s %-6s %s\n' \
    SUITE COMPILE RC RESULT FAILS SKIPS VERDICT
for i in "${!R_NAME[@]}"; do
    printf '%-38s %-8s %-4s %-26s %-6s %-6s %s\n' \
        "${R_NAME[$i]}" "${R_COMPILE[$i]}" "${R_RC[$i]}" \
        "${R_RESULT[$i]}" "${R_FAILS[$i]}" "${R_SKIPS[$i]}" "${R_VERDICT[$i]}"
done
echo
echo "logs: $LOGDIR/{gemm,experts}.{compile,run}.log"
if [ "$overall" = 0 ]; then
    echo "verify_mrows: ALL SUITES PASSED"
else
    echo "verify_mrows: FAILED (exit $overall: 1 = a suite failed, 2 = harness)"
fi
exit "$overall"
