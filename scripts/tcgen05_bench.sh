#!/usr/bin/env bash
# =============================================================================
# tcgen05_bench.sh — T1 ISOLATED MICROBENCH DRIVER (go/no-go) for the tcgen05
#                    mxf4 routed-expert gate/up arm.
# =============================================================================
#
# WHAT IT DECIDES
#   docs/agent/final-400-config.md §P0-3 lists "tcgen05 mxf4 routed 落地"
#   (−6.8 ms) and gates it on "先过单层微基准门（22.2 / 17.2 µs）"; §5-T1 calls
#   that gate the go/no-go. The numbers come from
#   docs/agent/routed-expert-residual.md §3(b) and STATUS.md:
#     * the SIMT gate/up (`expert_gemv_fp4_batched_kernel`) is ncu-measured at
#       22.2 us/call (STATUS:4936 / :5026), so the tcgen05 arm must come in
#       UNDER 22.2 us at the same shape to be worth integrating;
#     * the fused down (`expert_gemv_fp4_down_reduce`) is 17.2 us (STATUS:5575),
#       the reference for the down half. ⚠️ NO tcgen05 down kernel exists yet
#       (residual §3(b)-①: "down 的 swapAB + fused asc-slot reduce 未写"), so
#       this script MEASURES that reference and reports the tcgen05 side as
#       NOT-IMPLEMENTED — it does not invent an ABI.
#
#   PASS (gateup < 22.2 us)  => GO     → proceed to the T4 serve A/B
#   FAIL                     => NO-GO  → tcgen05 stays default OFF, fall back to
#                                        the down 4-value path (§止损门)
#
# SHAPE (production, corrected 2026-09-12): dim=5120 / inter_local=320 / topk=6 /
#   n_routed=384 / world=8, rows(m)=5. The harness prints the real grid for both
#   arms: the tcgen05 gate/up is (2*inter/128, slots) = (5, 6) = 30 CTAs, i.e.
#   30 of 148 SMs — the documented TMEM occupancy limit, and the reason a
#   slots=1 number is a LATENCY measurement, not a bandwidth one.
#
# WHAT IT RUNS (all on ONE free GPU; nothing here touches a serve)
#   STEP 1  .so symbol gate — `nm -D` for the tcgen05 entry; rebuilds the .so
#           with kernels/cuda/build.sh if the symbol is missing (build.sh has
#           -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 ON by default since
#           2026-09-12, so an ordinary rebuild carries it).
#   STEP 2  compile the microbench (scripts/tcgen05_mxf4_bench.cu) with the same
#           flags as build.sh + sm_103a, linked against the PRODUCTION .so.
#   STEP 3  two passes — gate OFF then gate ON:
#             OFF  proves the gate is a REAL gate (the entry is a silent no-op:
#                  rc==0 and nothing written — the project's #1 measurement-bias
#                  trap) and puts the baselines on record;
#             ON   the T1 measurement (tcgen05 gate/up + the SIMT baselines
#                  re-taken in the same process, back to back).
#   STEP 4  verdict: median vs 22.2 us (gateup) / 17.2 us (down reference).
#
# USAGE (run on the box that owns the tree, the .so AND one free GPU)
#   bash scripts/tcgen05_bench.sh
#   ITERS=100 WARMUP=20 ROWS=5 bash scripts/tcgen05_bench.sh
#   GPU=3 bash scripts/tcgen05_bench.sh
#   (NODE/ssh remote driving is deliberately NOT implemented here: the harness
#    must link the .so that this tree built, so the run belongs on that box.)
#
# EXIT CODES  0 = GO · 1 = NO-GO · 2 = setup/build failure
#
# LOGS  /tmp/tcgen05_bench_off.log (gate OFF) · /tmp/tcgen05_bench_on.log (ON)
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
K="$ROOT/kernels/cuda"
HARNESS="$HERE/tcgen05_mxf4_bench.cu"
BIN="${BIN:-/tmp/tcgen05_mxf4_bench}"
LOG_OFF="${LOG_OFF:-/tmp/tcgen05_bench_off.log}"
LOG_ON="${LOG_ON:-/tmp/tcgen05_bench_on.log}"

ARCH="${ARCH:-103a}"                 # B300 = sm_103a (AGENTS.md: 必带)
ITERS="${ITERS:-100}"                # T1: median over 100 launches
WARMUP="${WARMUP:-20}"
ROWS="${ROWS:-5}"                    # the T1 "m" (activation rows)
SLOTS="${SLOTS:-6}"                  # topk / n_activated_experts
NEXP="${NEXP:-384}"                  # n_routed_experts (full production pool)
DIM="${DIM:-5120}"
IL="${IL:-320}"                      # inter_local
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

GATEUP_GATE_US="${GATEUP_GATE_US:-22.2}"   # STATUS:4936 / :5026 (ncu, SIMT gateup)
DOWN_GATE_US="${DOWN_GATE_US:-17.2}"       # STATUS:5575 (fused down+reduce)
NSYS_MED_US="83.1"                         # STATUS:3585 (nsys, m=5, per-launch median)

pass=0; fail=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
warn() { echo "  WARN  $*"; }
head1(){ echo; echo "=============================================================="; echo "== $*"; echo "=============================================================="; }

echo "== tcgen05 mxf4 — T1 isolated microbench (go/no-go) =="
echo "   tree    : $ROOT"
echo "   .so     : $K/libferrite_kernels.so"
echo "   GPU     : $GPU   arch: sm_$ARCH"
echo "   shape   : dim=$DIM inter=$IL slots=$SLOTS experts=$NEXP rows(m)=$ROWS"
echo "   timing  : warmup=$WARMUP iters=$ITERS (median)"
echo "   gates   : gateup < ${GATEUP_GATE_US}us   down < ${DOWN_GATE_US}us (reference; no tcgen05 down kernel)"
echo "   context : nsys expert_gemv_fp4_batched median ${NSYS_MED_US}us/launch at m=5 (STATUS:3585)"

# ---------------------------------------------------------------- step 0 ---
head1 "STEP 0 — preflight (nvcc, free GPU, .so)"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "FATAL: no nvcc on PATH — this script must run on the GPU/compile box." >&2
    echo "       (the repo convention is: ssh <gpu-node>, cd <repo>, then this script)" >&2
    exit 2
fi
ok "nvcc: $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"

for p in dsv41-run dsv41-serve ferrite-serve ferrite-dsv41; do
    if pgrep -x "$p" >/dev/null 2>&1; then
        echo "FATAL: '$p' is alive — the GPU must be free (a serve would pollute every number)." >&2
        exit 2
    fi
done
if command -v nvidia-smi >/dev/null 2>&1; then
    apps="$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
    if [ -n "$apps" ]; then
        echo "FATAL: GPU $GPU already has compute apps:" >&2
        printf '%s\n' "$apps" | sed 's/^/       /' >&2
        exit 2
    fi
    ok "GPU $GPU idle (nvidia-smi: no compute apps)"
else
    warn "nvidia-smi not found — the busy check is pgrep-only"
fi

if [ ! -f "$K/libferrite_kernels.so" ]; then
    warn "no .so at $K/libferrite_kernels.so — STEP 1 will build it"
fi
[ -f "$HARNESS" ] || { echo "FATAL: harness not found: $HARNESS" >&2; exit 2; }

# ---------------------------------------------------------------- step 1 ---
head1 "STEP 1 — .so symbol gate (compile-time half of the two-part gate)"

SO="$K/libferrite_kernels.so"
check_sym() { # $1 = symbol
    [ -f "$SO" ] && nm -D --defined-only "$SO" 2>/dev/null | grep -qw "$1"
}
TC_SYM="dsv41_expert_tcgen05_gate_up_mxf4"
SIMT_SYMS=(dsv41_expert_gate_up_fp4_batched dsv41_expert_down_reduce_fp4_batched)

if check_sym "$TC_SYM"; then
    ok "$TC_SYM present"
else
    warn "$TC_SYM ABSENT — rebuilding the .so (build.sh defaults the mxf4 block ON)"
    ( cd "$K" && DSV41_BUILD_TCGEN05_MXF4=1 bash build.sh "$ARCH" ) || {
        echo "FATAL: build.sh failed" >&2; exit 2; }
    if check_sym "$TC_SYM"; then
        ok "$TC_SYM present after rebuild"
        warn "the .so was rebuilt alone — the Rust binary pair is now STALE. Before any serve A/B:"
        warn "  touch crates/ferrite-kernel/build.rs && cargo build --release   (AGENTS.md 双产物纪律)"
    else
        bad "$TC_SYM still absent after a rebuild with DSV41_BUILD_TCGEN05_MXF4=1"
        echo "       the tcgen05 arm cannot be measured; fix the build before re-running" >&2
        echo "RESULT: NO-GO (setup: the tcgen05 symbol is not in the .so)" >&2
        exit 2
    fi
fi
for sy in "${SIMT_SYMS[@]}"; do
    if check_sym "$sy"; then ok "$sy present"; else bad "$sy MISSING (the baseline arms cannot link)"; fi
done

for probe in dsv41_expert_tcgen05_down_mxf4 dsv41_expert_tcgen05_down_reduce_mxf4; do
    if check_sym "$probe"; then
        warn "$probe is present: a tcgen05 DOWN kernel now exists — extend the harness to measure it"
    fi
done

if [ -f "$K/.build_id" ]; then
    echo "   build_id: $(cat "$K/.build_id")"
else
    warn "no $K/.build_id — cannot state which source the .so came from"
fi
if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
    warn "working tree is DIRTY — the .so is only valid for the current (uncommitted) source"
fi

# ---------------------------------------------------------------- step 2 ---
head1 "STEP 2 — build the microbench (build.sh flags + sm_$ARCH, linked against the .so)"

# Same flag set as kernels/cuda/build.sh (FAST_MATH ON by default there) plus the
# arch, minus -shared: this is an executable that links the .so.
CC_CMD=(nvcc -O3 --use_fast_math -std=c++17
        -gencode "arch=compute_${ARCH},code=sm_${ARCH}"
        -o "$BIN" "$HARNESS" -L"$K" -l:libferrite_kernels.so)
echo "  \$ ${CC_CMD[*]}"
cc_out="$("${CC_CMD[@]}" 2>&1)"; cc_rc=$?
if [ "$cc_rc" != 0 ]; then
    printf '%s\n' "$cc_out" | grep -m10 -i 'error' | sed 's/^/    /'
    bad "harness compile failed (rc=$cc_rc)"
    echo "RESULT: NO-GO (setup: the harness did not build)" >&2
    exit 2
fi
ok "harness built: $BIN"

# ---------------------------------------------------------------- step 3 ---
head1 "STEP 3 — run (gate OFF, then gate ON)"

# Production env (scripts/expert_ncu_bench.cu documents this set; DSV41_PDL=0 is
# the isolation convention — a PDL launch may start during the previous kernel's
# tail and would UNDERSTATE a per-launch interval).
# ⚠️ DSV41_EXPERT_ILV=0 is the documented companion of the tcgen05 gate. It is a
# Rust-side (loader) switch, NOT read by the .so: the harness builds BOTH pool
# layouts explicitly (ILV for the production SIMT baseline, direct for tcgen05)
# and passes the layout as a launcher argument. Exporting it keeps the run's env
# identical to the serve A/B this milestone feeds.
BASE_ENV=(
    DSV41_EXPERT_ILV=0
    DSV41_PDL=0
    DSV41_EXPERT_FP4_MODE=2
    DSV41_GATEUP_KSPLIT=2
    DSV41_GATEUP_ROWS=8
    DSV41_GATEUP_FUSE=1
    DSV41_DOWN_FUSE=1
    DSV41_DOWN_VEC4=1
)
COMMON_ARGS=(--iters "$ITERS" --warmup "$WARMUP" --rows "$ROWS"
             --slots "$SLOTS" --nexp "$NEXP" --dim "$DIM" --inter "$IL")

run_pass() { # $1 = gate value (0|1), $2 = log
    local gate="$1" log="$2"
    echo "  \$ DSV41_EXPERT_TCGEN05=$gate ${BASE_ENV[*]} $BIN ${COMMON_ARGS[*]}"
    env -u DSV41_EXPERT_TCGEN05_MXF4 -u DSV41_W2_PREWARM \
        DSV41_EXPERT_TCGEN05="$gate" "${BASE_ENV[@]}" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        LD_LIBRARY_PATH="$K${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$BIN" "${COMMON_ARGS[@]}" >"$log" 2>&1
    local rc=$?
    sed -n '1,12p' "$log" | sed 's/^/    /'
    echo "    ... (full log: $log)"
    return $rc
}

run_pass 0 "$LOG_OFF" || { bad "gate-OFF pass exited non-zero (see $LOG_OFF)"; sed -n '1,40p' "$LOG_OFF" >&2; }
run_pass 1 "$LOG_ON"  || { bad "gate-ON pass exited non-zero (see $LOG_ON)";  sed -n '1,40p' "$LOG_ON"  >&2; }

ran_off="$(awk '$1=="TCGEN05_RAN"{print $2}' "$LOG_OFF" 2>/dev/null | tail -1)"
ran_on="$(awk  '$1=="TCGEN05_RAN"{print $2}' "$LOG_ON"  2>/dev/null | tail -1)"

head1 "STEP 3b — the gate is a REAL gate (the #1 measurement-bias trap)"
if [ "$ran_off" = "no" ]; then
    ok "gate OFF: the tcgen05 entry returned 0 and wrote NOTHING (real no-op)"
else
    bad "gate OFF: the tcgen05 entry WROTE its output (ran_off='${ran_off:-<none>}') — the runtime gate is broken"
fi
if [ "$ran_on" = "yes" ]; then
    ok "gate ON: the tcgen05 kernel actually ran and wrote its output"
else
    bad "gate ON: no output written — DSV41_EXPERT_TCGEN05 is not reaching the .so (ran_on='${ran_on:-<none>}')"
fi

# ---------------------------------------------------------------- step 4 ---
head1 "STEP 4 — verdict"

med() { awk -v t="$1" '$1=="MEDIAN_US" && $2==t {v=$3} END{if (v!="") print v}' "$2"; }
num() { [ -n "$1" ] && [ "$1" != "-" ]; }

tc_med="$(med tcgen05_gateup "$LOG_ON")"
s_ilv1="$(med gateup_simt_ilv1 "$LOG_ON")"
s_ilv1_r1="$(med gateup_simt_ilv1_r1 "$LOG_ON")"
s_ilv0="$(med gateup_simt_ilv0 "$LOG_ON")"
s_ilv0_r1="$(med gateup_simt_ilv0_r1 "$LOG_ON")"
dn="$(med down_simt "$LOG_ON")"
dn_r1="$(med down_simt_r1 "$LOG_ON")"

echo "  ARM                        median us"
printf "  %-26s %s\n" "tcgen05_gateup"        "${tc_med:-<none>}"
printf "  %-26s %s\n" "gateup_simt_ilv1 (prod,m=$ROWS)" "${s_ilv1:-<none>}"
printf "  %-26s %s\n" "gateup_simt_ilv1 (m=1)" "${s_ilv1_r1:-<none>}"
printf "  %-26s %s\n" "gateup_simt_ilv0 (m=$ROWS)" "${s_ilv0:-<none>}"
printf "  %-26s %s\n" "gateup_simt_ilv0 (m=1)" "${s_ilv0_r1:-<none>}"
printf "  %-26s %s\n" "down_simt (m=$ROWS)"    "${dn:-<none>}"
printf "  %-26s %s\n" "down_simt (m=1)"       "${dn_r1:-<none>}"
echo

verdict=0
if ! num "$tc_med"; then
    bad "GATEUP: no tcgen05 number (see STEP 3b) — the gate cannot be cleared"
    verdict=1
elif awk -v m="$tc_med" -v g="$GATEUP_GATE_US" 'BEGIN{exit !(m < g)}'; then
    ok "GATEUP: tcgen05 ${tc_med}us < ${GATEUP_GATE_US}us  => PASS (go)"
else
    bad "GATEUP: tcgen05 ${tc_med}us >= ${GATEUP_GATE_US}us => FAIL (no-go)"
    verdict=1
fi
if num "$tc_med" && num "$s_ilv1_r1"; then
    awk -v a="$s_ilv1_r1" -v b="$tc_med" \
        'BEGIN{ if (b>0) printf "  INFO  same-work speedup vs SIMT m=1 (ilv1): %.2fx  (%.3f -> %.3f us)\n", a/b, a, b }'
fi
if num "$tc_med" && num "$s_ilv1"; then
    awk -v a="$s_ilv1" -v b="$tc_med" -v r="$ROWS" \
        'BEGIN{ if (b>0) printf "  INFO  speedup vs SIMT m=%s (ilv1): %.2fx  (m=%s folds %s rows/launch; the tcgen05 grid has NO rows dim, so this one is informational only)\n", r, a/b, r, r }'
fi

# The down half: measured reference only — no tcgen05 down kernel exists.
if num "$dn"; then
    if awk -v m="$dn" -v g="$DOWN_GATE_US" 'BEGIN{exit !(m < g)}'; then
        echo "  INFO  DOWN reference: SIMT down ${dn}us < ${DOWN_GATE_US}us (already at the gate; nothing to win here)"
    else
        warn "DOWN reference: SIMT down ${dn}us >= ${DOWN_GATE_US}us"
    fi
fi
warn "DOWN: NOT-IMPLEMENTED — no tcgen05 down kernel in the tree (residual §3(b)-①: swapAB +"
warn "      fused asc-slot reduce 未写). The ${DOWN_GATE_US}us down gate is UNMEASURABLE today; the"
warn "      down half of the −6.8ms cannot be collected until that kernel exists."

echo
if [ "$verdict" = 0 ]; then
    echo "RESULT: GO — the tcgen05 mxf4 gate/up clears the T1 microbench gate."
    echo "        next: T4 serve A/B (EXPERT_TCGEN05=0/1) with the four texts + faults=0."
    [ "$ran_on" != "yes" ] && echo "        ⚠️ but the gate-ON sentinel failed — re-read STEP 3b first."
    echo "        caveat: a GO here is an ISOLATION result. This kernel has gone isolated-"
    echo "               positive → serve-neutral before (expert-tcgen05-plan.md §5-7); the"
    echo "               stop-loss is a serve A/B Δ<0.05ms ⇒ default stays OFF."
else
    echo "RESULT: NO-GO — the tcgen05 gate/up does NOT clear the gate."
    echo "        per final-400-config.md §止损门: keep tcgen05 default OFF, do not invest in"
    echo "        the variant matrix; fall back to the down 4-value (40-reg) path."
fi
echo
echo "logs: $LOG_OFF (gate OFF) · $LOG_ON (gate ON)"
echo "repro: ITERS=$ITERS ROWS=$ROWS GPU=$GPU bash scripts/tcgen05_bench.sh"
echo "summary: gateup ${tc_med:-n/a}us vs ${GATEUP_GATE_US}us gate · down ${dn:-n/a}us (reference ${DOWN_GATE_US}us) · nsys ctx ${NSYS_MED_US}us"

[ "$verdict" = 0 ] || exit 1
exit 0
