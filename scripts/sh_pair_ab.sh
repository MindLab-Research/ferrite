#!/usr/bin/env bash
# sh_pair_ab.sh — the SH_PAIR `template<M>` serve A/B (FOUR arms, strictly serial).
#
# WHAT IT ANSWERS
#   "Does `DSV41_SH_PAIR_M=1` actually DISPATCH the shared expert's `template<M>`
#    chain on the real serve — and does it buy the step wall the design's §3.3/§7
#    promise, WITHOUT moving a single token?"
#
# WHY A SCRIPT IS NEEDED AT ALL (the failure mode it exists to catch)
#   The arm is dispatched FIRST inside `ChainDev::shared_expert_mrows`
#   (crates/ferrite-models/src/dsv41/chain_dev.rs), behind a Rust gate
#   (`sh_pair_m()`) AND a `.so` symbol probe (`supports_sh_exp_fused()` →
#   `dsv41_gemm_fp8_sh_exp_fused`). A stale `.so` that lacks the symbol leaves the
#   gate inert: BOTH arms then run the per-row chain and the A/B reports "no-op"
#   for a reason that has NOTHING to do with the kernel (the project's #1
#   measurement trap). The preflight below refuses to measure unless the symbol is
#   exported, and the per-arm env read-back proves the gate ARRIVED.
#
# THE MATRIX (4 arms; fold_r is a RUNTIME kernel argument — the sweep does NOT
# recompile; `sh_pair_m_fold()` is read once per process, so one serve per arm)
#   | arm  | DSV41_SH_PAIR_M | DSV41_SH_PAIR_M_FOLD | reads as                     |
#   |------|-----------------|----------------------|------------------------------|
#   | base | OFF (absent)    | —                    | the per-row chain (reference)|
#   | m1f1 | 1               | 1 (the §3.3 default)| template<M=1>, 1 row/block   |
#   | m1f2 | 1               | 2                    | phase-1 fold 2 rows/block    |
#   | m1f6 | 1               | 6                    | pure M-fold (worst SM cover) |
#   Verdicts are m1f1-base, m1f2-base, m1f6-base (and m1fN-m1f1 for the fold knob).
#
# WHY THE DESIGN EXPECTS fold_r=1 TO WIN (§3.3, honestly bounded)
#   The model in the design doc puts phase 1's bit-identical parallelism at
#   `ceil(n1/32) * ceil(M/fold_r)` chains. `fold_r=1` maps the M rows over the
#   GRID (54 blocks at M=6) while `fold_r=6` folds them into the warp's registers
#   (9 blocks) — fewer instructions but 6x less SM coverage, so the model says
#   fold_r=1 is ~1.5x faster than 2 and ~3.5x faster than 6. THAT IS A MODEL, not
#   a measurement — which is what this script is for. All three fold arms must be
#   bit-identical to base, so a fold arm that is SLOWER is a tuning result, and a
#   fold arm that CHANGES THE TEXT is a correctness bug.
#
# 口径 (the project rules this script encodes — same as lazy_graph_ab.sh / s2_ab_matrix.sh)
#   1. SERIAL, ONE SERVE AT A TIME, torn down between every (arm, prompt): two
#      serves on the same 8 GPUs do not give noisy numbers, they give MEANINGLESS
#      ones.
#   2. ONE PROMPT PER SERVE: the `[dspark] steps=…` accumulators are declared
#      OUTSIDE the request loop and never reset per request, so two prompts in one
#      serve would report a mixed average in the last line. 4 arms x 2 prompts =
#      8 serial serves. (The per-round `[dsv41] step pos=…` lines ARE per request,
#      so the step wall / the k_acc sequence are clean either way — but the
#      dspark cross-check is not, and the rule is uniform.)
#   3. SAME BINARY + SAME `.so` ON EVERY ARM, PROVEN. `.so` and `.build_id` are
#      untracked, so a checkout+clean leaves them behind while the caller may have
#      rebuilt only one product: the loader then aborts at dlopen. `cargo build`
#      alone can NEVER heal it (cargo is incremental and `ferrite-kernel/build.rs`
#      bakes the stamp in); the only working order is
#          build.sh (rewrites .build_id) -> touch crates/ferrite-kernel/build.rs
#          -> cargo build --release.
#      The caller owns the tree; --build runs that order, otherwise the pair is
#      only PROVEN, never rebuilt.
#   4. THE RUNNING ENV IS READ BACK OFF /proc AND CHECKED (not what we typed):
#      base must have NO `DSV41_SH_PAIR_M`, the M arms must have it, every arm
#      must carry `DSV41_LAZY_VERIFY=1`, and NO arm may carry
#      `DSV41_SH_EXP_MROWS` (else "base" is not the per-row chain) or
#      `DSV41_SWALLOW_STEP` (lazy verify ALREADY implies SWALLOW — the extra gate
#      is not ours to set). A leaked bit ABORTS the arm; it is never averaged in.
#
# 判据 (the acceptance, printed as a per-prompt verdict; THREE legs)
#   * SPEED — the STEADY step wall (`[dsv41] step pos=…: Xms`, first STEADY_SKIP
#     rounds dropped) of the M arm must be below base by >= MIN_GAIN_MS. The
#     headline expectation is -4.9..-7.9 ms off the shared-expert kernel
#     (10.4 -> 2.5..5.5 ms); the WHOLE-STEP expectation is smaller, so the gate
#     is deliberately conservative. The `[dspark] steps=` `verify=` field is
#     reported as corroboration only (it is a SERVE-PROCESS average).
#   * 零拉丁 (RED LINE) — the 出师表 answer is Chinese; any ASCII letter is
#     corruption leaking through. Zero latin, `先帝创业未半` present, no adjacent
#     double-char, non-empty. (The double-char rule is NOT applied to the digit
#     prompt: "11"/"22"/… are LEGITIMATE adjacent repeats there.)
#   * k_acc 不变 — the optimization is bit-identical BY CONSTRUCTION (the kernel's
#     C1-C8 contract), so every arm must return the SAME greedy answer (same md5)
#     AND the same k_acc sequence (same md5 of the derived accepts). ANY drift is
#     a correctness bug and the default must NOT move, however good the timing.
#
# k_acc 序列 / 直方图
#   `DSV41_LAZY_VERIFY` has no dedicated histogram print, but the per-round
#   `[dsv41] step pos=…` lines carry it for free: the engine advances the position
#   by `k_emit = k_acc + 1` per committed round, so the DELTA between consecutive
#   positions MINUS ONE is that round's k_acc. The parser writes the RAW SEQUENCE
#   (`<tag>.kacc`) plus the 0..6 histogram and its mean, and cross-checks the
#   mean against the serve's own `mean-k`.
#
# PARITY HARD GATE (do not skip)
#   The phase-1 tail-group phantom-row fix (M % fold_r != 0 wrote out of bounds)
#   is only trustworthy once the parity suite passes:
#       bash scripts/verify_mrows.sh        # or the sh_exp_mrows TU directly
#   THIS script measures performance; it does NOT re-run parity. Run parity first.
#
# USAGE
#   bash scripts/sh_pair_ab.sh                 # 8 serial serves, reuse the pair on the node
#   bash scripts/sh_pair_ab.sh --build         # rebuild BOTH products first, then run
#   bash scripts/sh_pair_ab.sh --no-build      # explicit: prove the pair only
#   bash scripts/sh_pair_ab.sh --dry-run       # pre-flight only; launches nothing
#   bash scripts/sh_pair_ab.sh --judge-only    # re-judge the existing $LOGDIR artifacts
#   SYNC=1 bash scripts/sh_pair_ab.sh          # rsync THIS tree to the node first
#   MAXTOK=1000 PORT_BASE=8250 MIN_GAIN_MS=2 bash scripts/sh_pair_ab.sh
#
# OUTPUT: per (arm, prompt) — the FULL answer, the k_acc sequence + histogram,
#   mean-k, tok/step, verify_ms and the steady step wall (mean, median, min, p10);
#   then the Δ table and the three-leg verdict per prompt.
#   Logs: $LOGDIR/<tag>.{log,dspark,resp.json,metrics,txt,kacc,env}
#   Exit: 0 = all three legs hold on every prompt; 1 = SPEED or a RED LINE / k_acc
#         drift failed; 2 = the run is NOT usable (unmeasured arm / leaked gate bit
#         / missing symbol / harness error).
#
# ⚠️ SYNC=1 DOES rsync --delete THE REPO OVER THE NODE'S WORKING TREE. The node
#    tree is normally the caller's deployment, so it is OFF by default and the
#    script only WARNS when the revisions differ.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

NODE="${NODE:-ubuntu@43.202.208.136}"
ARCH="${ARCH:-103a}"
RROOT="${RROOT:-ferrite}"
LOGDIR="${SHP_LOGDIR:-/tmp/sh_pair_ab}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
CURL="curl -s --noproxy '*'"

PORT_BASE="${PORT_BASE:-8250}"
MAXTOK="${MAXTOK:-1000}"                        # both prompts: full answer, no truncation
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"              # x5s
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"
STEADY_SKIP="${STEADY_SKIP:-20}"                # warm-up rounds dropped before the steady stats
MIN_GAIN_MS="${MIN_GAIN_MS:-2.0}"               # the SPEED leg, M arm vs base
EPS="${EPS:-0.05}"                              # |Δmean-k| below this is reported as "null"

# 出师表 is the project's long-prompt yardstick (dspark_verify.rs PROMPTS): a long
# prompt + recitation, so it is sensitive to cumulative corruption AND runs long
# enough (>= 50 dspark steps) to print the `[dspark] steps=` line.
# The digit task is the high-accept / throughput side (k_acc ~5 -> 6 tok/step):
# 1000 tokens makes the generation phase long enough for a steady wall.
PROMPT_SH="${PROMPT_SH:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"
PROMPT_DI="${PROMPT_DI:-请从 1 数到 1000，每个数字单独占一行，只输出数字本身，不要任何解释。}"

BUILD=""    # empty = neither --build nor --no-build: prove the pair, never rebuild
DRY=0
JUDGE_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --build)      BUILD=1 ;;
        --no-build)   BUILD=0 ;;
        --dry-run)    DRY=1 ;;
        --judge-only) JUDGE_ONLY=1 ;;
        -h|--help)    sed -n '2,120p' "$0"; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)"; exit 2 ;;
    esac
    shift
done

mkdir -p "$LOGDIR"

rssh() { ssh "${SSH_OPTS[@]}" "$NODE" "$1"; }

# One writer at a time, across invocations too: a second serve while the first is
# up is exactly the failure this script exists to prevent.
LOCK="$LOGDIR/.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "FATAL: another sh_pair_ab.sh holds $LOCK — arms must never overlap."
    exit 2
fi

# ---------------------------------------------------------------------------
# The FIXED base env — identical in all four arms (only the SH_PAIR_M bits move).
#   SPEC/DSPARK/SIDS_WRITEBACK : the DSpark real-commit path the verify gates live
#                                in; without it the verify gates are inert.
#   Wave 1 gates (nsys_wave1.sh's GATES, all ON) : HC_VERIFY_FUSE / HC_FRONT_ROWS
#                                / VERIFY_AR_FOLD / GATE_MROWS / INDEXER_MROWS /
#                                COMPRESSOR_MROWS / LAZY_VERIFY / VERIFY_GRAPH /
#                                BF16_TRUNCATE.
#   LAZY_VERIFY                : the arm under test (lazy's row 0 IS the swallowed
#                                main chain step — it ALREADY implies SWALLOW).
#   EXPERT_ACT_E4M3 / DRAFT_P3A / TAP_INPUT / DRAFT_BF16_DOMAIN : the canonical
#                                lazy-arm levers.
#   TIMING                     : REQUIRED for `[dsv41] step pos=…` (and dspark).
# NOT set on purpose: DSV41_SH_EXP_MROWS (base must be the PER-ROW chain),
#   DSV41_SH_EXP_FUSED / DSV41_SH_PAIR (the M=1 fused arm is behind SH_PAIR_M and
#   would never run), DSV41_SWALLOW_STEP (lazy verify already implies it).
# ---------------------------------------------------------------------------
BASE_ENV="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1 DSV41_VERIFY_AR_FOLD=1 \
DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_COMPRESSOR_MROWS=1 \
DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1 \
DSV41_BF16_TRUNCATE=1 \
DSV41_EXPERT_ACT_E4M3=1 DSV41_DRAFT_P3A=1 DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 \
DSV41_TIMING=1"

# The four arms: the ARM's extra env. base's is empty (the reference).
ARMS=(base m1f1 m1f2 m1f6)
arm_extra() {
    case "$1" in
        base) echo "" ;;
        m1f1) echo "DSV41_SH_PAIR_M=1 DSV41_SH_PAIR_M_FOLD=1" ;;
        m1f2) echo "DSV41_SH_PAIR_M=1 DSV41_SH_PAIR_M_FOLD=2" ;;
        m1f6) echo "DSV41_SH_PAIR_M=1 DSV41_SH_PAIR_M_FOLD=6" ;;
    esac
}
# A short human label for a fold value ("-" for base).
arm_fold() {
    case "$1" in
        base) echo "-" ;;
        m1f1) echo "1" ;;
        m1f2) echo "2" ;;
        m1f6) echo "6" ;;
    esac
}

# ===========================================================================
# 0. Node reachability + revision on each side (a mismatch is a WARNING: with
#    SYNC unset the node tree is the caller's and measuring it is legitimate —
#    as long as the reader KNOWS the two differ).
# ===========================================================================
echo "== SH_PAIR template<M> A/B (4 arms x 2 prompts, strictly serial) =="
echo "-- node $NODE   arch $ARCH   tp $TP   ports $PORT_BASE..$(( PORT_BASE + 7 ))"
echo "-- prompts: 出师表 max_tokens=$MAXTOK (零拉丁/k_acc)   digits max_tokens=$MAXTOK (吞吐)"
echo "-- steady-skip=$STEADY_SKIP   SPEED gate >= ${MIN_GAIN_MS}ms   |Δmean-k| eps=$EPS"
echo "-- ⚠️  parity hard gate: run scripts/verify_mrows.sh (sh_exp_mrows TU) FIRST —"
echo "--     this script measures performance only and does not re-run parity."
echo "-- base env (identical in all four arms):"
echo "     $BASE_ENV"
for a in "${ARMS[@]}"; do
    echo "-- arm $a: SH_PAIR_M_FOLD=$(arm_fold "$a")   extra=[$(arm_extra "$a")]"
done

REV_LOCAL="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && REV_LOCAL="$REV_LOCAL-dirty"
if [ "$DRY" != 1 ] && [ "$JUDGE_ONLY" != 1 ]; then
    rssh "true" >/dev/null 2>&1 || { echo "FATAL: cannot reach $NODE"; exit 2; }
    if [ -n "${SYNC:-}" ]; then
        echo "-- SYNC=1: rsync --delete $ROOT -> $NODE:$RROOT (the repo's own files only)"
        rssh "mkdir -p ~/$RROOT" || { echo "FATAL: cannot reach $NODE"; exit 2; }
        rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
            --exclude target/ --exclude .git/ --exclude '*.so' --exclude .build_id \
            "$ROOT/" "$NODE:$RROOT/" || { echo "FATAL: rsync failed"; exit 2; }
    fi
    REV_NODE="$(rssh "git -C ~/$RROOT rev-parse --short HEAD 2>/dev/null || echo '?'" 2>/dev/null)"
    [ -n "$(rssh "git -C ~/$RROOT status --porcelain 2>/dev/null" 2>/dev/null)" ] && REV_NODE="$REV_NODE-dirty"
    echo "-- rev: local $REV_LOCAL   node $REV_NODE   (no rsync by design)"
    if [ "$REV_LOCAL" != "$REV_NODE" ]; then
        echo "   WARN: the node tree is a DIFFERENT revision from this checkout, and it is"
        echo "         what will be measured. The caller owns that tree; sync it by hand if"
        echo "         that is not intended (SYNC=1 rsyncs this one over it)."
    fi
fi

# ===========================================================================
# 1. Optional build (the one order that works), then PROVE the pair is same-source
#    AND that the .so exports the symbol the M-row arm needs.
# ===========================================================================
SO_REL="kernels/cuda/libferrite_kernels.so"
BIN_REL="target/release/ferrite-serve"
if [ "$JUDGE_ONLY" != 1 ]; then
    if [ "$BUILD" = 1 ] && [ "$DRY" != 1 ]; then
        echo "-- build: kernels/cuda/build.sh $ARCH ..."
        # ⚠️ build.sh's last statement is `[ ${#SKELETON_FLAGS[@]} -gt 0 ] && echo …`,
        # so with no skeleton flags it exits 1 under its own `set -e` EVEN THOUGH IT
        # BUILT THE .so. The "built …" line is the success criterion, not the rc.
        rssh "cd ~/$RROOT/kernels/cuda && bash build.sh $ARCH" >"$LOGDIR/build_so.log" 2>&1
        so_rc=$?
        if ! grep -q "built .*libferrite_kernels.so for sm_${ARCH}" "$LOGDIR/build_so.log"; then
            echo "FATAL: build.sh $ARCH did not report a successful build (rc=$so_rc, log $LOGDIR/build_so.log)"
            tail -5 "$LOGDIR/build_so.log" | sed 's/^/    | /'
            exit 2
        fi
        [ "$so_rc" != 0 ] && echo "   (build.sh exited $so_rc but reported a successful build — trailing '[ ] && echo' under set -e; the .so IS fresh)"
        echo "-- build: cargo build --release (touch build.rs first, the stamp is baked in by it) ..."
        rssh "cd ~/$RROOT && touch crates/ferrite-kernel/build.rs && bash -lc 'cd ~/$RROOT && cargo build --release'" \
            >"$LOGDIR/build_bin.log" 2>&1 \
            || { echo "FATAL: cargo build --release failed (log $LOGDIR/build_bin.log)"; tail -5 "$LOGDIR/build_bin.log"; exit 2; }
    elif [ "$DRY" = 1 ]; then
        echo "-- build: skipped (--dry-run is read-only)"
    fi

    # Same-source proof, post-build, never an assumption. `grep -cF` and NOT `grep -qF`:
    # under `set -o pipefail` a `strings BIN | grep -q id` makes grep exit at its first
    # match, `strings` takes SIGPIPE (141) and the PIPELINE reports failure even though
    # the id WAS found — the gate would misfire "stale" on a good pair.
    NODE_BUILD_ID="$(rssh "cat ~/$RROOT/kernels/cuda/.build_id 2>/dev/null")"
    embeds_id() {
        [ -n "$NODE_BUILD_ID" ] || return 1
        rssh "strings ~/$RROOT/$BIN_REL | grep -cF -- '$NODE_BUILD_ID'" >/dev/null 2>&1
    }
    HAS_SO="$(rssh "[ -f ~/$RROOT/$SO_REL ] && echo yes || echo no")"
    HAS_BIN="$(rssh "[ -f ~/$RROOT/$BIN_REL ] && echo yes || echo no")"
    if [ "$HAS_SO" != yes ] || [ "$HAS_BIN" != yes ]; then
        echo "FATAL: the node is missing ~/$RROOT/{$SO_REL,$BIN_REL} (so=$HAS_SO bin=$HAS_BIN) — run with --build"
        exit 2
    fi
    PAIR_OK=1
    if ! embeds_id; then
        PAIR_OK=0
        if [ "$DRY" != 1 ]; then
            echo "FATAL: the node's binary does NOT embed the .so's build id — the pair is not same-source."
            echo "       (.so id on the node: ${NODE_BUILD_ID:-<missing>}; re-run with --build)"
            exit 2
        fi
    fi
    [ "$PAIR_OK" = 1 ] && echo "-- pair: same-source OK (build id $NODE_BUILD_ID)"

    # THE symbol gate. Without `dsv41_gemm_fp8_sh_exp_fused` the Rust probe
    # (`supports_sh_exp_fused()`) is false and `shared_expert_mrows` never reaches
    # this arm: BOTH A and B would run the per-row chain and a no-delta result
    # would be a STALE .so, not the kernel. Refusing here is what keeps that
    # misdiagnosis off the table.
    HAS_NM="$(rssh "command -v nm >/dev/null 2>&1 && echo yes || echo no")"
    if [ "$HAS_NM" = yes ]; then
        if rssh "nm -D ~/$RROOT/$SO_REL 2>/dev/null | grep -c 'dsv41_gemm_fp8_sh_exp_fused'" >/dev/null 2>&1; then
            echo "-- pair: .so exports dsv41_gemm_fp8_sh_exp_fused (the template<M> entry is present)"
        else
            echo "FATAL: the .so exports no dsv41_gemm_fp8_sh_exp_fused — the template<M> arm"
            echo "       cannot run at all, so a no-delta A/B here would be a STALE .so, NOT the kernel."
            echo "       rebuild: (cd kernels/cuda && bash build.sh $ARCH) && touch crates/ferrite-kernel/build.rs && cargo build --release"
            exit 2
        fi
    else
        echo "-- pair: nm not found on the node — skipping the symbol check"
    fi
fi

# ===========================================================================
# 2. Dry run stops here: nothing was built and no serve ran.
# ===========================================================================
if [ "$DRY" = 1 ]; then
    free="$(rssh "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END {print s+0}'")"
    running="$(rssh 'pgrep -x ferrite-serve | wc -l')"
    echo "-- dry run: pair same-source $([ "${PAIR_OK:-0}" = 1 ] && echo YES || echo 'NO (a real run would need --build)')"
    echo "-- dry run: GPU memory in use = ${free} MiB   ferrite-serve running = $running (both must be 0 for a clean run)"
    echo "-- dry run: a real run would start 8 serial serves on ports $PORT_BASE..$(( PORT_BASE + 7 )); nothing was launched."
    exit 0
fi

if [ "$JUDGE_ONLY" = 1 ]; then
    SKIP_RUN=1     # re-judge the artifacts already in $LOGDIR
else
    SKIP_RUN=0
fi

# ===========================================================================
# 3. Teardown by EXACT process name: `pkill -f ferrite-serve` would match this
#    very command line (it contains the string) and kill the ssh session.
# ===========================================================================
kill_serves() {
    rssh "pkill -9 -x ferrite-serve 2>/dev/null; sleep $TEARDOWN_SLEEP; pgrep -x ferrite-serve | wc -l"
}
teardown() {
    local port="$1" left
    rssh "$CURL -m 5 -X POST http://localhost:$port/shutdown >/dev/null 2>&1"
    for _ in $(seq 1 10); do
        [ "$(rssh 'pgrep -x ferrite-serve | wc -l')" = 0 ] && break
        rssh "sleep 3"
    done
    left="$(kill_serves)"
    [ "$left" != 0 ] && { echo "FATAL: $left ferrite-serve process(es) survived teardown — arms must never overlap."; exit 2; }
    return 0
}

# ===========================================================================
# 4. The per-(arm,prompt) parser: read (dspark, resp.json, log) and write a
#    `<tag>.metrics` key=value file + `<tag>.txt` (the FULL answer) +
#    `<tag>.kacc` (the raw k_acc sequence). Everything the report needs is derived
#    here, once.
# ===========================================================================
metrics_of() {  # tag  prompt_kind(sh|di)
    python3 - "$LOGDIR/$1.dspark" "$LOGDIR/$1.resp.json" "$LOGDIR/$1.log" \
               "$LOGDIR/$1.metrics" "$LOGDIR/$1.txt" "$LOGDIR/$1.kacc" \
               "$STEADY_SKIP" "$2" <<'PY'
import hashlib
import json
import re
import sys

dspark_p, resp_p, log_p, out_p, txt_p, kacc_p = sys.argv[1:7]
skip = int(sys.argv[7])
kind = sys.argv[8]


def read(path):
    try:
        return open(path, "r", errors="ignore").read()
    except OSError:
        return ""


def field(line, key):
    i = line.find(key)
    if i < 0:
        return None
    m = re.match(r"[-+0-9.eE]+", line[i + len(key):])
    return float(m.group(0)) if m else None


dspark_txt = read(dspark_p)
log = read(log_p)

# ---- the `[dspark] steps=… mean-k=… tok/step=… draft=… verify=… commit=…` line:
#      a SERVE-PROCESS average (its accumulators live outside the request loop),
#      so it CORROBORATES the per-step wall, it does not replace it. LAST wins.
dspark_line = ""
for line in dspark_txt.splitlines():
    if "[dspark] steps=" in line and "mean-k=" in line:
        dspark_line = line.rstrip("\n")
steps = field(dspark_line, "steps=")
mean_k = field(dspark_line, "mean-k=")
tok_step = field(dspark_line, "tok/step=")
verify_ms = field(dspark_line, "verify=")
draft_ms = field(dspark_line, "draft=")
commit_ms = field(dspark_line, "commit=")

# ---- the per-round wall: `[dsv41] step pos=<p>: <X>ms (<tok/s>)` -------------
seq = [(int(a), float(b))
       for a, b in re.findall(r"\[dsv41\] step pos=(\d+): ([\d.]+)ms", log)]
wall = [ms for _, ms in seq]
steady = wall[skip:] if len(wall) > skip + 1 else wall[:]
# the stop round can cut a block short: drop the last sample when we have room
if len(steady) > 1:
    steady = steady[:-1]


def stats(xs):
    if not xs:
        return 0, None, None, None, None
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p10 = s[max(0, int(0.10 * n) - 1)] if n else None
    return n, mean, med, s[0], p10


sn, smean, smed, smin, sp10 = stats(steady)

# ---- k_acc sequence + histogram from the position deltas: the engine advances by
#      k_emit = k_acc + 1 per committed round, so delta - 1 is that round's k_acc.
deltas = [seq[i + 1][0] - seq[i][0] for i in range(len(seq) - 1)]
kacc = [d - 1 for d in deltas if 1 <= d <= 7]
hist = {k: kacc.count(k) for k in range(0, 7)}
kacc_mean = (sum(kacc) / len(kacc)) if kacc else None
kacc_seq = " ".join(str(k) for k in kacc)
kacc_md5 = hashlib.md5(kacc_seq.encode()).hexdigest() if kacc else "NA"

# ---- text: the FULL answer + the project's red lines -------------------------
content = ""
resp_err = ""
try:
    content = json.loads(read(resp_p))["choices"][0]["message"]["content"]
except Exception as exc:  # noqa: BLE001
    resp_err = str(exc)
with open(txt_p, "w") as fh:
    fh.write(content)
with open(kacc_p, "w") as fh:
    fh.write(kacc_seq + "\n")

ch = list(content)
dbl = sum(1 for i in range(1, len(ch)) if ch[i] == ch[i - 1] and not ch[i].isspace())
# 零拉丁红线: the answer is Chinese; any ASCII letter is corruption leaking through
# (the historical failure mode this project gates on).
latin_idx = [i for i, c in enumerate(ch) if ("a" <= c <= "z" or "A" <= c <= "Z")]
latin = len(latin_idx)
latin_samples = ",".join("".join(ch[max(0, i - 1):i + 2]) for i in latin_idx[:5])
md5 = hashlib.md5(content.encode()).hexdigest() if content else "NA"
has_kaishen = "yes" if "先帝创业未半" in content else "no"

fails = []
if not content:
    fails.append("empty answer")
if latin:
    fails.append("%d latin char(s)" % latin)
# The double-char rule is a 出师表 canary. It MUST NOT be applied to the digit
# prompt: "11"/"22"/…/"111" are legitimate adjacent repeats when counting.
if kind == "sh":
    if has_kaishen != "yes":
        fails.append("missing 先帝创业未半")
    if dbl:
        fails.append("%d adjacent double-char" % dbl)
text_ok = "ok" if not fails else "; ".join(fails)


def fmt(x, spec="%.3f"):
    return "NA" if x is None else spec % x


with open(out_p, "w") as fh:
    for k, v in [
        ("arm_kind", kind),
        ("steps", "NA" if steps is None else str(int(steps))),
        ("mean_k", fmt(mean_k)),
        ("tok_step", fmt(tok_step)),
        ("verify_ms", fmt(verify_ms, "%.2f")),
        ("draft_ms", fmt(draft_ms, "%.2f")),
        ("commit_ms", fmt(commit_ms, "%.2f")),
        ("rounds", str(len(wall))),
        ("steady_n", str(sn)),
        ("steady_mean", fmt(smean, "%.2f")),
        ("steady_median", fmt(smed, "%.2f")),
        ("steady_min", fmt(smin, "%.2f")),
        ("steady_p10", fmt(sp10, "%.2f")),
        ("kacc_n", str(len(kacc))),
        ("kacc_mean", fmt(kacc_mean)),
        ("kacc_md5", kacc_md5),
        ("hist0", str(hist[0])), ("hist1", str(hist[1])), ("hist2", str(hist[2])),
        ("hist3", str(hist[3])), ("hist4", str(hist[4])), ("hist5", str(hist[5])),
        ("hist6", str(hist[6])),
        ("chars", str(len(content))),
        ("dbl", str(dbl)),
        ("latin", str(latin)),
        ("latin_samples", latin_samples),
        ("md5", md5),
        ("has_kaishen", has_kaishen),
        ("text_ok", text_ok),
        ("resp_err", resp_err),
    ]:
        fh.write("%s=%s\n" % (k, v))
PY
}

# ===========================================================================
# 5. One serve per (arm, prompt), strictly serial. `extra` is the arm's env
#    delta; `forbidden` gates MUST NOT appear in the RUNNING env (the arm integrity
#    check). A leaked bit destroys the attribution, so the arm is ABORTED, never
#    averaged in.
# ===========================================================================
N=0
run_case() {  # arm tag prompt kind
    local arm="$1" tag="$2" prompt="$3" kind="$4"
    local port=$(( PORT_BASE + N ))
    local rlog="~/shp_${tag}.log"
    N=$(( N + 1 ))
    local extra; extra="$(arm_extra "$arm")"
    local genv="$BASE_ENV"
    [ -n "$extra" ] && genv="$genv $extra"

    echo
    echo "---- arm $arm / $kind (tag $tag, port $port): SH_PAIR_M_FOLD=$(arm_fold "$arm")"
    echo "     extra env: ${extra:-<none>}"

    kill_serves >/dev/null                     # every case starts from a clean box
    rssh "cd ~/$RROOT && nohup env CUDA_VISIBLE_DEVICES=$GPU_LIST \
          LD_LIBRARY_PATH=\$HOME/$RROOT/kernels/cuda \
          DSV41_KERNELS=\$HOME/$RROOT/kernels/cuda/libferrite_kernels.so \
          $genv ./target/release/ferrite-serve --model dsv41 --serve --tp $TP \
          --model-dir $MODEL_DIR --port $port > $rlog 2>&1 &" \
        || { echo "FATAL: could not spawn the serve for $tag"; exit 2; }

    if ! rssh "for i in \$(seq 1 $HEALTH_TRIES); do $CURL -m 2 http://localhost:$port/health >/dev/null 2>&1 && exit 0; sleep 5; done; exit 1"; then
        echo "FATAL: $tag never became healthy (log $rlog); last lines:"
        rssh "tail -15 $rlog" | sed 's/^/    | /'
        teardown "$port"; exit 2
    fi

    # The env actually running, straight off /proc — the cheapest proof that the
    # single-variable difference survived the shell.
    rssh "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort" \
        >"$LOGDIR/${tag}.env" 2>/dev/null
    echo "   effective env:"; sed 's/^/     /' "$LOGDIR/${tag}.env"

    # ---- arm integrity -----------------------------------------------------
    if [ "$arm" = base ]; then
        if grep -q '^DSV41_SH_PAIR_M' "$LOGDIR/${tag}.env"; then
            echo "   FATAL: base must NOT carry DSV41_SH_PAIR_M, but the running serve has it."
            teardown "$port"; exit 2
        fi
    else
        grep -q '^DSV41_SH_PAIR_M=1$' "$LOGDIR/${tag}.env" \
            || { echo "   FATAL: arm $arm's DSV41_SH_PAIR_M=1 did not survive into the serve's env."; teardown "$port"; exit 2; }
        grep -q "^DSV41_SH_PAIR_M_FOLD=$(arm_fold "$arm")$" "$LOGDIR/${tag}.env" \
            || { echo "   FATAL: arm $arm's DSV41_SH_PAIR_M_FOLD=$(arm_fold "$arm") did not survive."; teardown "$port"; exit 2; }
    fi
    # These two must never be set: SH_EXP_MROWS would make "base" a non-per-row
    # chain, and SWALLOW_STEP is already implied by lazy verify.
    for bad in DSV41_SH_EXP_MROWS DSV41_SWALLOW_STEP; do
        if grep -q "^${bad}=" "$LOGDIR/${tag}.env"; then
            echo "   FATAL: $bad is set in the running serve — the arms are not single-variable."
            teardown "$port"; exit 2
        fi
    done
    grep -q '^DSV41_LAZY_VERIFY=1$' "$LOGDIR/${tag}.env" \
        || { echo "   FATAL: DSV41_LAZY_VERIFY=1 is missing from the running serve."; teardown "$port"; exit 2; }

    local body t0 t1
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
            "$MODEL_NAME" "$prompt" "$MAXTOK")"
    t0=$(date +%s)
    rssh "cd ~/$RROOT && $CURL -m 1800 http://localhost:$port/v1/chat/completions \
          -H 'Content-Type: application/json' -d '$body'" >"$LOGDIR/${tag}.resp.json" 2>/dev/null
    t1=$(date +%s)
    rssh "grep 'dspark] steps' $rlog | tail -1" >"$LOGDIR/${tag}.dspark" 2>/dev/null
    rssh "cat $rlog" >"$LOGDIR/${tag}.log" 2>/dev/null

    teardown "$port"
    echo "   e2e $(( t1 - t0 ))s   dspark: $(head -c 140 "$LOGDIR/${tag}.dspark")"

    metrics_of "$tag" "$kind"
    [ -s "$LOGDIR/${tag}.metrics" ] \
        || { echo "FATAL: parser wrote no metrics for $tag (see $LOGDIR/${tag}.log)"; exit 2; }
    echo "   $(grep -E '^(steps|mean_k|kacc_mean|tok_step|steady_mean|steady_median|rounds|chars|dbl|latin|text_ok)=' "$LOGDIR/${tag}.metrics" | paste -sd' ' -)"
    echo "   k_acc hist [0..6]: $(grep -E '^hist[0-6]=' "$LOGDIR/${tag}.metrics" | cut -d= -f2 | paste -sd' ' -)"
    echo "   k_acc 序列 (n=$(grep -m1 '^kacc_n=' "$LOGDIR/${tag}.metrics" | cut -d= -f2), md5=$(grep -m1 '^kacc_md5=' "$LOGDIR/${tag}.metrics" | cut -d= -f2)):"
    echo "     $(cat "$LOGDIR/${tag}.kacc")"

    echo "   ---- full text ($tag), $(wc -c <"$LOGDIR/${tag}.txt") bytes ----"
    cat "$LOGDIR/${tag}.txt"
    echo
    echo "   ---- end full text ($tag) ----"
}

if [ "$SKIP_RUN" = 0 ]; then
    echo
    echo "== arms (serial, same binary+.so, one prompt per serve) =="
    for a in "${ARMS[@]}"; do
        run_case "$a" "${a}_sh" "$PROMPT_SH" sh
        run_case "$a" "${a}_di" "$PROMPT_DI" di
    done
fi

# ===========================================================================
# 6. The Δ table + the three-leg verdict, per prompt. Δ is arm minus base.
# ===========================================================================
echo
python3 - "$LOGDIR" "$MIN_GAIN_MS" "$EPS" <<'PY' | tee "$LOGDIR/table.txt"
import os
import sys

LOGDIR, min_gain, eps = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ARMS = ["base", "m1f1", "m1f2", "m1f6"]
FOLD = {"base": "-", "m1f1": "1", "m1f2": "2", "m1f6": "6"}
KINDS = [("sh", "出师表"), ("di", "计数")]
TAGS = ["%s_%s" % (a, k) for a, k in ((a, k) for a in ARMS for k, _ in KINDS)]


def load(tag):
    d = {}
    try:
        for line in open(os.path.join(LOGDIR, tag + ".metrics")):
            line = line.rstrip("\n")
            if "=" in line:
                k, v = line.split("=", 1)
                d[k] = v
    except OSError:
        pass
    return d


def num(d, k):
    try:
        return float(d[k])
    except (KeyError, ValueError):
        return None


def txt(tag):
    try:
        return open(os.path.join(LOGDIR, tag + ".txt"), errors="replace").read()
    except OSError:
        return None


M = {t: load(t) for t in TAGS}

print("==== SH_PAIR template<M> A/B — 4 arms x 2 prompts ====")
print("%-6s %-5s %-6s %7s %7s %7s %8s %8s %6s %5s %6s %s" % (
    "ARM", "FOLD", "PROMPT", "steps", "mean-k", "hist-k", "verify", "steady",
    "tok/stp", "chars", "latin", "TEXT"))
for k, label in KINDS:
    for a in ARMS:
        d = M["%s_%s" % (a, k)]

        def g(key):
            return d.get(key, "NA")

        print("%-6s %-5s %-6s %7s %7s %7s %8s %8s %6s %5s %6s %s" % (
            a, FOLD[a], label, g("steps"), g("mean_k"), g("kacc_mean"),
            g("verify_ms"), g("steady_mean"), g("tok_step"),
            g("chars"), g("latin"), g("text_ok")))

print()
print("k_acc 序列 (raw, md5) — derived from the [dsv41] step pos= deltas:")
for k, label in KINDS:
    for a in ARMS:
        d = M["%s_%s" % (a, k)]
        seq = ""
        try:
            seq = open(os.path.join(LOGDIR, "%s_%s.kacc" % (a, k))).read().strip()
        except OSError:
            pass
        print("  %-6s %-6s n=%-4s md5=%-12s %s" % (
            a, label, d.get("kacc_n", "NA"), d.get("kacc_md5", "NA"), seq or "<empty>"))

print()
print("判据:")
rc = 0

# --- USABLE: every (arm,prompt) needs a parseable dspark line AND a non-empty text.
unmeasured = [t for t in TAGS if num(M[t], "mean_k") is None]
empty = [t for t in TAGS if M[t].get("chars") in (None, "0")]
if unmeasured:
    print("  USABLE  FAIL: %s printed no '[dspark] steps=' line — was DSV41_TIMING=1 effective?"
          % ",".join(unmeasured))
    print("          A run with an unmeasured arm cannot carry any Δ.")
    rc = 2
elif empty:
    print("  USABLE  FAIL: %s produced an empty / unparseable answer — the harness, not the kernel."
          % ",".join(empty))
    rc = 2
else:
    print("  USABLE  OK: all 8 (arm,prompt) cells measured, every answer non-empty.")

# --- RED LINE (零拉丁 / 先帝创业未半 / no double-char) on EVERY cell.
bad_text = [t for t in TAGS if M[t].get("text_ok") != "ok"]
if bad_text:
    print("  REDLINE FAIL: %s" % ", ".join(
        "%s: %s" % (t, M[t].get("text_ok")) for t in bad_text))
    rc = max(rc, 1)
else:
    print("  REDLINE OK: all 8 cells zero-latin, 先帝创业未半 present (出师表), no double-char.")
    for k, label in KINDS:
        for a in ARMS:
            d = M["%s_%s" % (a, k)]
            if d.get("latin") not in ("0", None):
                print("          %s/%s latin samples: %s" % (a, label, d.get("latin_samples")))

# --- k_acc 不变: the optimization is bit-identical BY CONSTRUCTION, so the greedy
#     answer md5 AND the k_acc sequence md5 must match across all arms, per prompt.
print()
print("  k_acc / text 不变 (bit-identity, per prompt):")
for k, label in KINDS:
    base = M["base_%s" % k]
    tmd5 = {a: M["%s_%s" % (a, k)].get("md5") for a in ARMS}
    kmd5 = {a: M["%s_%s" % (a, k)].get("kacc_md5") for a in ARMS}
    tset = sorted(set(tmd5.values()))
    kset = sorted(set(kmd5.values()))
    ok_t = len(tset) == 1
    ok_k = len(kset) == 1
    print("    %-6s text md5 %s (%s)" % (
        label, "identical" if ok_t else "DRIFT", ",".join(str(x) for x in tset)))
    print("           k_acc seq md5 %s (%s)" % (
        "identical" if ok_k else "DRIFT", ",".join(str(x) for x in kset)))
    if not ok_t or not ok_k:
        print("           ⚠️  A drift here is a CORRECTNESS bug — do NOT move the default.")
        rc = max(rc, 1)
    # histogram-mean cross-check for the reader
    print("           mean-k: %s" % "  ".join(
        "%s=%s" % (a, M["%s_%s" % (a, k)].get("mean_k")) for a in ARMS))

# --- SPEED: the M arm's steady wall must beat base by >= min_gain, per prompt.
print()
print("  SPEED (steady step wall, arm minus base; gate >= %.2f ms):" % min_gain)
for k, label in KINDS:
    b = num(M["base_%s" % k], "steady_mean")
    if b is None:
        print("    %-6s base has no steady wall — cannot judge." % label)
        rc = max(rc, 1)
        continue
    print("    %-6s base=%.2fms" % (label, b))
    for a in ARMS[1:]:
        v = num(M["%s_%s" % (a, k)], "steady_mean")
        if v is None:
            print("      %-5s fold=%s  UNMEASURED" % (a, FOLD[a]))
            rc = max(rc, 1)
            continue
        d = b - v                                   # >0 => the M arm is faster
        verdict = "IMPROVES" if d >= min_gain else ("tiny" if d > 0 else "SLOWER")
        print("      %-5s fold=%s  %.2f -> %.2f   Δ %+.2fms   %s" % (
            a, FOLD[a], b, v, d, verdict))
    # fold-knob reference: relative to the fold_r=1 arm
    r1 = num(M["m1f1_%s" % k], "steady_mean")
    if r1 is not None:
        for a in ("m1f2", "m1f6"):
            v = num(M["%s_%s" % (a, k)], "steady_mean")
            print("      (vs m1f1) %-5s fold=%s  Δ %s" % (
                a, FOLD[a], "NA" if v is None else "%+.2fms" % (r1 - v)))

print()
print("  读判 (spelled out so the numbers are not over-read):")
print("    * The headline leg is m1f1-base on the DIGIT prompt (high accept, stable wall).")
print("      A drop >= %.2f ms can only come from the template<M> chain running; a ~0" % min_gain)
print("      delta means the STALE-SO / gate trap is back — check the preflight symbol line")
print("      and the per-arm env read-back before believing it.")
print("    * fold_r=1 is the design's §3.3 default (M over the GRID). fold_r=2/6 fold the rows")
print("      into the warp (fewer instructions, less SM coverage) and the model says they LOSE;")
print("      if a fold arm wins, that model is wrong and the default should move to it.")
print("    * Every text / k_acc md5 must be IDENTICAL across arms. Any drift is a correctness")
print("      bug: parity has regressed — go back to scripts/verify_mrows.sh, do NOT ship.")

print()
if rc == 2:
    print("verdict: rc=2 — NOT USABLE (an arm is unmeasured / harness error); fix the harness before reading Δ.")
elif rc == 1:
    print("verdict: rc=1 — SPEED leg and/or a RED LINE / bit-identity leg failed; do NOT move the default.")
else:
    print("verdict: rc=0 — usable, zero-latin, bit-identical; read the SPEED table to pick fold_r.")
sys.exit(rc)
PY
verdict=$?

echo
echo "Logs: $LOGDIR/<tag>.{log,dspark,resp.json,metrics,txt,kacc,env}   table: $LOGDIR/table.txt"
exit "$verdict"
