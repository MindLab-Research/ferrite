#!/usr/bin/env bash
# l49_ab.sh — the L4-9 (CNORM / NORM dim-split) serve A/B on the 91.1 stack.
#
# WHAT IT ANSWERS
#   "Do `DSV41_CNORM_SPLIT` and `DSV41_NORM_SPLIT` buy the +0.5%/+1% the L4-9
#    design promises, on the SAME 91.1 stack the rest of the batch runs — and do
#    they do it WITHOUT moving a token?"
#
# WHY THIS IS A NEW FILE (and not an edit of sh_pair_ab.sh)
#   sh_pair_ab.sh is the SH_PAIR `template<M>` A/B: 4 arms, its own BASE_ENV, its
#   own `.so` symbol gate (`dsv41_gemm_fp8_sh_exp_fused`) and its own 3-leg
#   verdict. L4-9 shares the SKELETON (launch / health / /proc environ / teardown
#   / per-round step-wall parser) but NONE of the semantics. Rewriting the same
#   file would silently break the SH_PAIR tool; this file COPIES the skeleton and
#   keeps the SH_PAIR script intact. REUSED VERBATIM: the launch/envchk/teardown
#   shape and the `[dsv41] step pos=` parser (incl. the 1<=d<=7 k_acc delta rule).
#
# THE DESIGN'S THREE CORRECTIONS THIS FILE ENCODES
#   1. T1-C IS AN EMPTY ARM BY ITSELF. `DSV41_NORM_SPLIT` only gates
#      `Device::rmsnorm_rows` (crates/ferrite-models/src/dsv41/device.rs:5816),
#      whose ONLY caller is `ChainDev::norm_rows`, itself behind
#      `DSV41_NORM_MROWS` (chain_dev.rs:1273, default OFF). Neither the 91.1 stack
#      nor sh_pair_ab.sh:BASE_ENV sets NORM_MROWS, so `DSV41_NORM_SPLIT=1` alone
#      is a NO-OP. This file therefore ships BOTH a true control (T1-C0 =
#      NORM_MROWS only) and the real arm (T1-C = NORM_MROWS + NORM_SPLIT), and the
#      verdict for C is taken against C0, never against A.
#   2. A NOISE FLOOR IS MANDATORY. T1-A2 repeats the control verbatim; without it
#      a +0.5% reading cannot be told from noise. sigma is derived from A vs A2.
#   3. DETERMINISM IS CHECKED. T1-D2 reruns T1-D; the two counting/出师表 answers
#      must be BYTE-IDENTICAL (the split kernel's fold order is FIXED — for q in
#      0..nchunks — so a mismatch can only be device-global scratch cross-talk).
#   Plus P7 ACTIVITY: for a non-bit-identical arm, "the text did not change" does
#      NOT prove the split kernel ran. Only `nsys` kernel invocation counts do
#      (`--nsys`, a SEPARATE pass — nsys pollutes the step wall, see §7.8 of the
#      design).
#
# THE MATRIX (7 arms, strictly serial, ONE FRESH SERVE PER ARM)
#   | arm   | extra env (the ONLY variable)                              | reads as                |
#   |-------|------------------------------------------------------------|-------------------------|
#   | T1-A  | (empty)                                                    | 91.1 baseline (control) |
#   | T1-A2 | (empty, REPEAT)                                            | NOISE FLOOR sigma       |
#   | T1-B  | DSV41_CNORM_SPLIT=1                                        | collapse_norm dim-split |
#   | T1-C0 | DSV41_NORM_MROWS=1                                         | the TRUE control for C  |
#   | T1-C  | DSV41_NORM_MROWS=1 DSV41_NORM_SPLIT=1                      | rmsnorm_rows dim-split  |
#   | T1-D  | DSV41_CNORM_SPLIT=1 DSV41_NORM_MROWS=1 DSV41_NORM_SPLIT=1  | both                    |
#   | T1-D2 | = T1-D, REPEAT                                             | DETERMINISM (byte-equal)|
#   Comparisons: B vs A; C vs C0; D vs C0 (and D vs A, shown alongside);
#   A2 vs A = sigma; D2 vs D = P8.
#
# ONE SERVE PER ARM, TWO REQUESTS INSIDE IT
#   The gates are `OnceLock` process-global (read once per process), so a fresh
#   serve per arm is non-negotiable. Within the arm's serve we send the COUNTING
#   prompt first (P1 + P4 throughput) and the 出师表 prompt second (P2 + P6 content).
#   ⚠️ This deviates from sh_pair_ab.sh's "ONE PROMPT PER SERVE" 口径, whose only
#   casualty is the process-level `[dspark] steps=` accumulator (never reset, so
#   the last line mixes both requests — it is REPORTED, never judged). The
#   `[dsv41] step pos=` per-round wall IS per-request (the chain resets on the
#   next prefill), so the counting request's wall is recovered by splitting the
#   log at the request boundary — which is exactly why this file snapshots the
#   log AFTER request 1 (`<tag>.di.log`) and keeps the whole serve log
#   (`<tag>.log`), deriving `sh.log = tail -n +(lines(di.log)+1)`.
#
# 口径 (the project rules this script encodes)
#   1. SERIAL, ONE SERVE AT A TIME, torn down between every arm. Two serves on
#      the same 8 GPUs give MEANINGLESS numbers.
#   2. SAME BINARY + SAME `.so` ON EVERY ARM, PROVEN (.build_id embed check).
#   3. THE RUNNING ENV IS READ BACK OFF /proc AND CHECKED (not what we typed):
#      an arm's three gates must be EXACTLY its arm_extra set — a leaked bit
#      ABORTS the arm, it is never averaged in.
#   4. Preflight symbol gate: BOTH `dsv41_hc_collapse_norm_split` and
#      `dsv41_rmsnorm_rows_split` must be exported by the `.so`, else a no-delta
#      A/B is a STALE `.so`, not the kernel.
#
# 判据 (printed as a per-arm verdict; automated)
#   P0 USABLE  — every arm printed a `[dsv41] step pos=` wall AND a non-empty
#                counting answer.
#   P1 计数     — the first 61 counting lines are exactly "1".."61"; first_bad
#                (1-based) >= max(61, first_bad(T1-A)).
#   P2 出师表    — latin(arm) <= latin(T1-A), no new latin sample, and
#                `先帝创业未半` present.
#   P3 k_acc    — the first-10 k_acc sequence md5 equals T1-A's; |dmean_k| < 0.05.
#   P4 吞吐     — steady step wall (STEADY_SKIP rounds dropped) vs the DESIGN
#                control, with the sigma floor from T1-A vs T1-A2.
#   P5 hang     — zero `ar5-hang` / fault lines in the serve log.
#   P6 内容     — leading-char agreement vs T1-A >= TOKEN_MATCH_CHARS for BOTH
#                the counting and the 出师表 answer. >=10 = PASS, 1..9 = WARN,
#                0 = FAIL. (A latin-only check is a FALSE NEGATIVE here: the
#                dim-split drift is NUMERIC — digits change, the charset does not.)
#   P7 活性     — (`--nsys` only) split-kernel invocation count > 0.
#   P8 确定性   — T1-D and T1-D2 byte-identical on both answers.
#   DECISION (design §5.3): Δ% = (steady_ctrl - steady_arm)/steady_ctrl * 100.
#      sigma >= 1%              -> 噪声地板过高, skip
#      Δ% <= -0.3%              -> OFF (negative)
#      -0.3% < Δ% <= max(sigma, 0.3%) -> skip (in-noise)
#      Δ% >  max(sigma, 0.3%)   -> 入栈 (gate stays ON)
#
# USAGE
#   bash scripts/l49_ab.sh                 # 7 serial serves, reuse the pair on the node
#   bash scripts/l49_ab.sh --build         # rebuild BOTH products first, then run
#   bash scripts/l49_ab.sh --dry-run       # pre-flight only; launches nothing
#   bash scripts/l49_ab.sh --judge-only    # re-judge the existing $LOGDIR artifacts
#   bash scripts/l49_ab.sh --nsys          # separate P7 pass (B / C / D under nsys)
#   SYNC=1 bash scripts/l49_ab.sh          # rsync THIS tree to the node first
#   MAXTOK=1000 PORT_BASE=8260 bash scripts/l49_ab.sh
#
# OUTPUT: per arm — the counting first_bad, the 出师表 latin/prefix, the steady
#   wall (mean/median/min/p10) and the est tok/s; then the Δ table (vs control and
#   vs sigma) and the P1..P8 + DECISION block.
#   Logs: $LOGDIR/<tag>.{env,di.log,sh.log,log,di.resp.json,sh.resp.json,
#                        di.txt,sh.txt,metrics,kacc,.di.txt}
#   Exit: 0 = usable and every red line held; 1 = a red line / determinism failed;
#         2 = the run is NOT usable (unmeasured arm / leaked gate bit / missing
#             symbol / harness error).
#
# ⚠️ SYNC=1 DOES rsync --delete THE REPO OVER THE NODE'S WORKING TREE. The node
#    tree is normally the caller's deployment, so it is OFF by default.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

NODE="${NODE:-ubuntu@43.202.208.136}"
ARCH="${ARCH:-103a}"
RROOT="${RROOT:-ferrite}"
LOGDIR="${L49_LOGDIR:-/tmp/l49_ab}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
CURL="curl -s --noproxy '*'"

PORT_BASE="${PORT_BASE:-8260}"
MAXTOK="${MAXTOK:-1000}"                        # both prompts: full answer, no truncation
REQ_TIMEOUT="${REQ_TIMEOUT:-300}"               # <= 5 min per request (the user's cap)
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"              # x5s
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"
STEADY_SKIP="${STEADY_SKIP:-20}"                # warm-up rounds dropped before the steady stats
MIN_GAIN_PCT="${MIN_GAIN_PCT:-0.3}"             # the 0.3% leg of the decision rule
SIGMA_FLOOR="${SIGMA_FLOOR:-1.0}"               # sigma >= this -> skip everything
TOKEN_MATCH_CHARS="${TOKEN_MATCH_CHARS:-10}"    # P6 leading-char agreement (tcgen05_smoke.sh default)

# 出师表 is the project's long-prompt yardstick (dspark_verify.rs PROMPTS); the
# digit task is the high-accept / throughput side. PROMPT_DI is kept VERBATIM
# from sh_pair_ab.sh:145 so the numbers stay comparable with the historical stack.
PROMPT_SH="${PROMPT_SH:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"
PROMPT_DI="${PROMPT_DI:-请从 1 数到 1000，每个数字单独一行，只输出数字本身，不要任何解释。}"

BUILD=""    # empty = neither --build nor --no-build: prove the pair, never rebuild
DRY=0
JUDGE_ONLY=0
NSYS_MODE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --build)      BUILD=1 ;;
        --no-build)   BUILD=0 ;;
        --dry-run)    DRY=1 ;;
        --judge-only) JUDGE_ONLY=1 ;;
        --nsys)       NSYS_MODE=1 ;;
        -h|--help)    sed -n '2,130p' "$0"; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)"; exit 2 ;;
    esac
    shift
done

mkdir -p "$LOGDIR"

rssh() { ssh "${SSH_OPTS[@]}" "$NODE" "$1"; }

# One writer at a time, across invocations too.
LOCK="$LOGDIR/.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "FATAL: another l49_ab.sh holds $LOCK — arms must never overlap."
    exit 2
fi

# ---------------------------------------------------------------------------
# The FIXED base env — identical in all seven arms (only the L4-9 bits move).
# This is the 91.1 stack (design §2, batch plan §4, verbatim).
# NOT set on purpose: DSV41_CNORM_SPLIT / DSV41_NORM_SPLIT / DSV41_NORM_MROWS
#   (the three L4-9 gates — the arms under test) and DSV41_NORM_SPLIT_NC (the
#   runtime chunk-count knob, default (dim+1023)/1024 = 5).
# NCCL_NVLS_ENABLE / CUDA_VISIBLE_DEVICES are on the launch line, not here.
# ---------------------------------------------------------------------------
BASE_ENV="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1 \
DSV41_SH_EXP_MROWS=1 DSV41_SH_PAIR_M=1 \
DSV41_ATTN_LIN_FUSE=1 DSV41_MARKOV_SLICED=1 DSV41_LAZY_SDR=1 \
DSV41_VERIFY_FORK=1 DSV41_RING_WIN_FUSE=1 \
DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1 DSV41_VERIFY_AR_FOLD=1 \
DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_COMPRESSOR_MROWS=1 \
DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 DSV41_DRAFT_P3A=1"

# The three L4-9 gates — the ONLY bits that may differ between arms.
L49_GATES=(DSV41_CNORM_SPLIT DSV41_NORM_MROWS DSV41_NORM_SPLIT)

# The seven arms: the ARM's extra env. T1-A's is empty (the reference).
ARMS=(T1-A T1-A2 T1-B T1-C0 T1-C T1-D T1-D2)
arm_extra() {
    case "$1" in
        T1-A)  echo "" ;;
        T1-A2) echo "" ;;
        T1-B)  echo "DSV41_CNORM_SPLIT=1" ;;
        T1-C0) echo "DSV41_NORM_MROWS=1" ;;
        T1-C)  echo "DSV41_NORM_MROWS=1 DSV41_NORM_SPLIT=1" ;;
        T1-D)  echo "DSV41_CNORM_SPLIT=1 DSV41_NORM_MROWS=1 DSV41_NORM_SPLIT=1" ;;
        T1-D2) echo "DSV41_CNORM_SPLIT=1 DSV41_NORM_MROWS=1 DSV41_NORM_SPLIT=1" ;;
    esac
}
# The DESIGN control each arm is judged against (single-variable rule).
arm_ctrl() {
    case "$1" in
        T1-A2) echo "T1-A" ;;
        T1-B)  echo "T1-A" ;;
        T1-C0) echo "T1-A" ;;
        T1-C)  echo "T1-C0" ;;
        T1-D)  echo "T1-C0" ;;
        T1-D2) echo "T1-D" ;;
        *)     echo "-" ;;
    esac
}
# The three gates an arm MUST carry (space-joined); empty = none.
arm_gates() {
    case "$1" in
        T1-A|T1-A2) echo "" ;;
        T1-B)       echo "DSV41_CNORM_SPLIT" ;;
        T1-C0)      echo "DSV41_NORM_MROWS" ;;
        T1-C)       echo "DSV41_NORM_MROWS DSV41_NORM_SPLIT" ;;
        T1-D|T1-D2) echo "DSV41_CNORM_SPLIT DSV41_NORM_MROWS DSV41_NORM_SPLIT" ;;
    esac
}
# A short human label.
arm_desc() {
    case "$1" in
        T1-A)  echo "control (91.1)" ;;
        T1-A2) echo "control REPEAT (noise floor)" ;;
        T1-B)  echo "+CNORM_SPLIT" ;;
        T1-C0) echo "+NORM_MROWS (C's true control)" ;;
        T1-C)  echo "+NORM_MROWS+NORM_SPLIT" ;;
        T1-D)  echo "+CNORM+NORM_MROWS+NORM_SPLIT" ;;
        T1-D2) echo "= T1-D REPEAT (determinism)" ;;
    esac
}

# ===========================================================================
# 0. Node reachability + revision (a mismatch is a WARNING: with SYNC unset the
#    node tree is the caller's and measuring it is legitimate).
# ===========================================================================
echo "== L4-9 (CNORM / NORM dim-split) A/B — 7 arms, 91.1 stack, strictly serial =="
echo "-- node $NODE   arch $ARCH   tp $TP   ports $PORT_BASE..$(( PORT_BASE + 6 ))"
echo "-- counting max_tokens=$MAXTOK  出师表 max_tokens=$MAXTOK   request timeout ${REQ_TIMEOUT}s"
echo "-- steady-skip=$STEADY_SKIP   decision: Δ% > max(1σ, ${MIN_GAIN_PCT}%) -> 入栈   (σ>=${SIGMA_FLOOR}% -> skip)"
echo "-- P6 leading-char agreement >= $TOKEN_MATCH_CHARS (counting + 出师表 vs T1-A)"
echo "-- ⚠️  T1-C alone is an EMPTY ARM: NORM_SPLIT needs NORM_MROWS=1 first (device.rs:5816 -> chain_dev.rs:1273)."
echo "-- base env (identical in all seven arms):"
echo "     $BASE_ENV"
for a in "${ARMS[@]}"; do
    echo "-- arm $a: $(arm_desc "$a")   extra=[$(arm_extra "$a")]   ctrl=$(arm_ctrl "$a")"
done

REV_LOCAL="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && REV_LOCAL="$REV_LOCAL-dirty"
if [ "$DRY" != 1 ] && [ "$JUDGE_ONLY" != 1 ] && [ "$NSYS_MODE" != 1 ]; then
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
# 1. Optional build (the one order that works), then PROVE the pair is
#    same-source AND that the `.so` exports BOTH split entries.
# ===========================================================================
SO_REL="kernels/cuda/libferrite_kernels.so"
BIN_REL="target/release/ferrite-serve"
if [ "$JUDGE_ONLY" != 1 ]; then
    if [ "$BUILD" = 1 ] && [ "$DRY" != 1 ] && [ "$NSYS_MODE" != 1 ]; then
        echo "-- build: kernels/cuda/build.sh $ARCH ..."
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

    # Same-source proof, post-build, never an assumption (`grep -cF` not `-qF`:
    # under pipefail a `strings | grep -q` makes strings take SIGPIPE and the
    # PIPELINE reports failure even though the id WAS found).
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

    # THE symbol gate. Without these two exports the `cnorm_split_wanted()` /
    # `norm_split_wanted()` probes see `None` and BOTH arms run the ORIGINAL
    # launch — a no-delta A/B would then be a STALE `.so`, not the kernel.
    HAS_NM="$(rssh "command -v nm >/dev/null 2>&1 && echo yes || echo no")"
    if [ "$HAS_NM" = yes ]; then
        miss=""
        for sym in dsv41_hc_collapse_norm_split dsv41_rmsnorm_rows_split; do
            rssh "nm -D ~/$RROOT/$SO_REL 2>/dev/null | grep -c '$sym'" >/dev/null 2>&1 || miss="$miss $sym"
        done
        if [ -n "$miss" ]; then
            echo "FATAL: the .so exports no$miss — the L4-9 arm cannot run at all, so a"
            echo "       no-delta A/B here would be a STALE .so, NOT the kernel."
            echo "       rebuild: (cd kernels/cuda && bash build.sh $ARCH) && touch crates/ferrite-kernel/build.rs && cargo build --release"
            exit 2
        fi
        echo "-- pair: .so exports dsv41_hc_collapse_norm_split + dsv41_rmsnorm_rows_split (both split entries present)"
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
    echo "-- dry run: a real run would start 7 serial serves on ports $PORT_BASE..$(( PORT_BASE + 6 )); nothing was launched."
    exit 0
fi

if [ "$JUDGE_ONLY" = 1 ]; then
    SKIP_RUN=1     # re-judge the artifacts already in $LOGDIR
else
    SKIP_RUN=0
fi

# ===========================================================================
# 3. Teardown by EXACT process name (`pkill -f ferrite-serve` would match this
#    very command line and kill the ssh session).
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
# 4. The per-arm parser: read (di.log, sh.log, di.resp.json, sh.resp.json) and
#    write `<tag>.metrics` + `<tag>.di.txt` + `<tag>.sh.txt` + `<tag>.kacc`.
#    The `[dsv41] step pos=` parser is sh_pair_ab.sh's, verbatim (1<=d<=7).
# ===========================================================================
parse_arm() {  # tag
    python3 - "$LOGDIR/$1.di.log" "$LOGDIR/$1.sh.log" \
               "$LOGDIR/$1.di.resp.json" "$LOGDIR/$1.sh.resp.json" \
               "$LOGDIR/$1.metrics" "$LOGDIR/$1.di.txt" "$LOGDIR/$1.sh.txt" \
               "$LOGDIR/$1.kacc" "$STEADY_SKIP" <<'PY'
import hashlib
import json
import re
import sys

di_log_p, sh_log_p, di_resp_p, sh_resp_p, out_p, di_txt_p, sh_txt_p, kacc_p = sys.argv[1:9]
skip = int(sys.argv[9])


def read(path):
    try:
        return open(path, "r", errors="ignore").read()
    except OSError:
        return ""


def stats(xs):
    if not xs:
        return 0, None, None, None, None
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p10 = s[max(0, int(0.10 * n) - 1)]
    return n, mean, med, s[0], p10


def wall_of(log):
    seq = [(int(a), float(b))
           for a, b in re.findall(r"\[dsv41\] step pos=(\d+): ([\d.]+)ms", log)]
    wall = [ms for _, ms in seq]
    steady = wall[skip:] if len(wall) > skip + 1 else wall[:]
    if len(steady) > 1:                        # the stop round can cut a block short
        steady = steady[:-1]
    return seq, stats(steady), (sum(wall) / len(wall) if wall else None)


di_log, sh_log = read(di_log_p), read(sh_log_p)
di_seq, di_stats, di_wall_all = wall_of(di_log)
_, sh_stats, sh_wall_all = wall_of(sh_log)

# ---- k_acc sequence + histogram from the COUNTING request's position deltas:
#      the engine advances by k_emit = k_acc + 1 per committed round.
deltas = [di_seq[i + 1][0] - di_seq[i][0] for i in range(len(di_seq) - 1)]
kacc = [d - 1 for d in deltas if 1 <= d <= 7]
hist = {k: kacc.count(k) for k in range(0, 7)}
kacc_mean = (sum(kacc) / len(kacc)) if kacc else None
kacc_seq = " ".join(str(k) for k in kacc)
kacc_md5 = hashlib.md5(kacc_seq.encode()).hexdigest() if kacc else "NA"
kacc10 = " ".join(str(k) for k in kacc[:10])
kacc10_md5 = hashlib.md5(kacc10.encode()).hexdigest() if kacc10 else "NA"


def content(path):
    try:
        return json.loads(read(path))["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        return ""


di = content(di_resp_p)
sh = content(sh_resp_p)
with open(di_txt_p, "w") as fh:
    fh.write(di)
with open(sh_txt_p, "w") as fh:
    fh.write(sh)
with open(kacc_p, "w") as fh:
    fh.write(kacc_seq + "\n")

# ---- P1: the leading correct counting lines (>= 61) --------------------------
nz = [l.strip() for l in di.splitlines() if l.strip()]
ok_lines = 0
for i, l in enumerate(nz):
    if l == str(i + 1):
        ok_lines += 1
    else:
        break
first_bad = ok_lines + 1
fb_line = nz[ok_lines][:24] if ok_lines < len(nz) else ""


def text_stats(c):
    ch = list(c)
    dbl = sum(1 for i in range(1, len(ch)) if ch[i] == ch[i - 1] and not ch[i].isspace())
    li = [i for i, x in enumerate(ch) if ("a" <= x <= "z" or "A" <= x <= "Z")]
    samples = ",".join("".join(ch[max(0, i - 1):i + 2]) for i in li[:5])
    return dict(chars=len(c), dbl=dbl, latin=len(li), latin_samples=samples,
                md5=hashlib.md5(c.encode()).hexdigest() if c else "NA",
                has_kaishen=("yes" if "先帝创业未半" in c else "no"))


di_t, sh_t = text_stats(di), text_stats(sh)


def fmt(x, spec="%.3f"):
    return "NA" if x is None else spec % x


rows = [
    ("di_steps", str(len(di_seq))),
    ("di_steady_n", str(di_stats[0])),
    ("di_steady_mean", fmt(di_stats[1], "%.2f")),
    ("di_steady_median", fmt(di_stats[2], "%.2f")),
    ("di_steady_min", fmt(di_stats[3], "%.2f")),
    ("di_steady_p10", fmt(di_stats[4], "%.2f")),
    ("di_wall_all", fmt(di_wall_all, "%.2f")),
    ("di_ok_lines", str(ok_lines)),
    ("di_first_bad", str(first_bad)),
    ("di_first_bad_line", fb_line),
    ("di_kacc_n", str(len(kacc))),
    ("di_kacc_mean", fmt(kacc_mean)),
    ("di_kacc_md5", kacc_md5),
    ("di_kacc10_md5", kacc10_md5),
    ("di_kacc10", kacc10),
    ("di_chars", str(di_t["chars"])),
    ("di_latin", str(di_t["latin"])),
    ("di_dbl", str(di_t["dbl"])),
    ("di_md5", di_t["md5"]),
    ("sh_steady_n", str(sh_stats[0])),
    ("sh_steady_mean", fmt(sh_stats[1], "%.2f")),
    ("sh_wall_all", fmt(sh_wall_all, "%.2f")),
    ("sh_chars", str(sh_t["chars"])),
    ("sh_latin", str(sh_t["latin"])),
    ("sh_latin_samples", sh_t["latin_samples"]),
    ("sh_dbl", str(sh_t["dbl"])),
    ("sh_md5", sh_t["md5"]),
    ("sh_has_kaishen", sh_t["has_kaishen"]),
]
for k in range(0, 7):
    rows.append(("di_hist%d" % k, str(hist[k])))

with open(out_p, "w") as fh:
    for k, v in rows:
        fh.write("%s=%s\n" % (k, v))
PY
}

# ===========================================================================
# 5. One serve per ARM, strictly serial. The counting request runs FIRST (its
#    step wall is the P4 metric), then the 出师表 request (P2/P6 content). The log
#    is snapshotted between them so the counting wall is cleanly isolated.
#    The running env is read back and its three L4-9 gates must be EXACTLY the
#    arm's set — a leaked bit ABORTS the arm, it is never averaged in.
# ===========================================================================
N=0
run_arm() {  # arm
    local arm="$1" tag="$1"
    local port=$(( PORT_BASE + N ))
    local rlog="~/l49_${tag}.log"
    N=$(( N + 1 ))
    local extra; extra="$(arm_extra "$arm")"
    local genv="$BASE_ENV"
    [ -n "$extra" ] && genv="$genv $extra"

    echo
    echo "---- arm $arm / $(arm_desc "$arm")  (tag $tag, port $port)"
    echo "     extra env: ${extra:-<none>}   ctrl=$(arm_ctrl "$arm")"

    kill_serves >/dev/null                     # every arm starts from a clean box
    rssh "cd ~/$RROOT && nohup env -u FERRITE_P2P CUDA_VISIBLE_DEVICES=$GPU_LIST NCCL_NVLS_ENABLE=0 \
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

    # The env actually running, straight off /proc.
    rssh "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort" \
        >"$LOGDIR/${tag}.env" 2>/dev/null
    echo "   effective env:"; sed 's/^/     /' "$LOGDIR/${tag}.env"

    # ---- arm integrity: the three L4-9 gates must be EXACTLY the arm's set ----
    local want; want=" $(arm_gates "$arm") "
    for g in "${L49_GATES[@]}"; do
        if [[ "$want" == *" $g "* ]]; then
            grep -q "^${g}=1$" "$LOGDIR/${tag}.env" \
                || { echo "   FATAL: arm $arm must carry ${g}=1, but it did not survive into the serve's env."
                     teardown "$port"; exit 2; }
        else
            if grep -q "^${g}=" "$LOGDIR/${tag}.env"; then
                echo "   FATAL: arm $arm must NOT carry $g, but the running serve has it."
                echo "          A leaked gate destroys the single-variable attribution. ABORT."
                teardown "$port"; exit 2
            fi
        fi
    done
    # The 91.1 stack markers must have survived (sanity, not the variable).
    for g in DSV41_LAZY_VERIFY=1 DSV41_TIMING=1 DSV41_SH_PAIR_M=1 DSV41_SH_EXP_MROWS=1 DSV41_HC_VERIFY_FUSE=1; do
        grep -q "^${g}$" "$LOGDIR/${tag}.env" \
            || { echo "   FATAL: the 91.1 stack marker $g is missing from the running serve."; teardown "$port"; exit 2; }
    done
    grep -q '^DSV41_FUSE_B1=0$' "$LOGDIR/${tag}.env" \
        && { echo "   FATAL: DSV41_FUSE_B1=0 would make T1-B an EMPTY ARM (CNORM is on the fuse_b1 path)."; teardown "$port"; exit 2; }

    local body t0 t1
    # ---- request 1: COUNTING (P1 + P4 throughput + P3 k_acc) ----
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
            "$MODEL_NAME" "$PROMPT_DI" "$MAXTOK")"
    t0=$(date +%s)
    rssh "cd ~/$RROOT && $CURL -m $REQ_TIMEOUT http://localhost:$port/v1/chat/completions \
          -H 'Content-Type: application/json' -d '$body'" >"$LOGDIR/${tag}.di.resp.json" 2>/dev/null
    t1=$(date +%s)
    rssh "cat $rlog" >"$LOGDIR/${tag}.di.log" 2>/dev/null      # snapshot BEFORE 出师表
    echo "   counting: e2e $(( t1 - t0 ))s, resp $(wc -c <"$LOGDIR/${tag}.di.resp.json") bytes"

    # ---- request 2: 出师表 (P2 + P6 content) ----
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
            "$MODEL_NAME" "$PROMPT_SH" "$MAXTOK")"
    t0=$(date +%s)
    rssh "cd ~/$RROOT && $CURL -m $REQ_TIMEOUT http://localhost:$port/v1/chat/completions \
          -H 'Content-Type: application/json' -d '$body'" >"$LOGDIR/${tag}.sh.resp.json" 2>/dev/null
    t1=$(date +%s)
    rssh "cat $rlog" >"$LOGDIR/${tag}.log" 2>/dev/null
    local lines1; lines1="$(wc -l <"$LOGDIR/${tag}.di.log")"
    tail -n +$(( lines1 + 1 )) "$LOGDIR/${tag}.log" >"$LOGDIR/${tag}.sh.log"
    echo "   出师表: e2e $(( t1 - t0 ))s, resp $(wc -c <"$LOGDIR/${tag}.sh.resp.json") bytes"

    teardown "$port"

    parse_arm "$tag"
    [ -s "$LOGDIR/${tag}.metrics" ] \
        || { echo "FATAL: parser wrote no metrics for $tag (see $LOGDIR/${tag}.di.log)"; exit 2; }
    echo "   $(grep -E '^(di_ok_lines|di_first_bad|di_steady_mean|di_kacc_mean|sh_chars|sh_latin|sh_has_kaishen)=' "$LOGDIR/${tag}.metrics" | paste -sd' ' -)"
    echo "   k_acc hist [0..6]: $(grep -E '^di_hist[0-6]=' "$LOGDIR/${tag}.metrics" | cut -d= -f2 | paste -sd' ' -)"
    echo "   counting text (first 3 lines): $(head -3 "$LOGDIR/${tag}.di.txt" | paste -sd'|' -)"
    echo "   出师表 text (first 40 chars): $(head -c 40 "$LOGDIR/${tag}.sh.txt")"
}

if [ "$SKIP_RUN" = 0 ] && [ "$NSYS_MODE" = 0 ]; then
    echo
    echo "== arms (serial, same binary+.so, one fresh serve per arm) =="
    for a in "${ARMS[@]}"; do
        run_arm "$a"
    done
fi

# ===========================================================================
# 6. OPTIONAL P7: activity proof. nsys POLLUTES the step wall (design §7.8), so
#    this is a SEPARATE pass — split arms only, kernel counts only.
#    The `.so`'s host-barrier AR mode is pinned (DSV41_AR_V5=0 DSV41_GRAPH_STEP=0):
#    the device-side publish kernel spins, and under nsys that spin is amplified
#    ~300x. Neither pin gates the norm kernels, so "did the split kernel run" is
#    unaffected (nsys_wave1.sh's header documents the hazard).
# ===========================================================================
NSYS_ARMS=(T1-B T1-C T1-D)
if [ "$NSYS_MODE" = 1 ]; then
    NSYS_BIN="${NSYS:-/usr/local/cuda-13.2/bin/nsys}"
    echo
    echo "== P7 activity pass (nsys, ${NSYS_ARMS[*]}; split-kernel invocation counts) =="
    echo "-- nsys binary: $NSYS_BIN   (nsys pollutes timing: do NOT read ms from this pass)"
    for a in "${NSYS_ARMS[@]}"; do
        port=$(( PORT_BASE + 6 + N )); N=$(( N + 1 ))
        rep="~/l49_nsys_$a"
        extra="$(arm_extra "$a")"
        kill_serves >/dev/null
        rssh "cd ~/$RROOT && nohup env -u FERRITE_P2P CUDA_VISIBLE_DEVICES=$GPU_LIST NCCL_NVLS_ENABLE=0 \
              DSV41_AR_V5=0 DSV41_GRAPH_STEP=0 \
              LD_LIBRARY_PATH=\$HOME/$RROOT/kernels/cuda \
              DSV41_KERNELS=\$HOME/$RROOT/kernels/cuda/libferrite_kernels.so \
              $BASE_ENV $extra \
              $NSYS_BIN profile --trace=cuda,nvtx --sample=none --output=$rep --force-overwrite=true \
              ./target/release/ferrite-serve --model dsv41 --serve --tp $TP \
              --model-dir $MODEL_DIR --port $port > $rep.log 2>&1 &" \
            || { echo "FATAL: could not spawn the nsys pass for $a"; exit 2; }
        if ! rssh "for i in \$(seq 1 $HEALTH_TRIES); do $CURL -m 2 http://localhost:$port/health >/dev/null 2>&1 && exit 0; sleep 5; done; exit 1"; then
            echo "WARN: the nsys pass for $a never became healthy; P7 for this arm is INCONCLUSIVE."
            rssh "tail -10 $rep.log" | sed 's/^/    | /'
            rssh "pkill -9 -x nsys 2>/dev/null; pkill -9 -x ferrite-serve 2>/dev/null; sleep 3"
            continue
        fi
        body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
                "$MODEL_NAME" "$PROMPT_DI" "${NSYS_MAXTOK:-120}")"
        rssh "cd ~/$RROOT && $CURL -m $REQ_TIMEOUT http://localhost:$port/v1/chat/completions \
              -H 'Content-Type: application/json' -d '$body'" >"$LOGDIR/${a}.nsys.resp.json" 2>/dev/null
        rssh "$CURL -m 5 -X POST http://localhost:$port/shutdown >/dev/null 2>&1"
        rssh "pkill -INT -x nsys 2>/dev/null; sleep 12; pkill -9 -x nsys 2>/dev/null; pkill -9 -x ferrite-serve 2>/dev/null; sleep 3"
        rssh "$NSYS_BIN stats --report cuda_gpu_kern_sum --format csv $rep.nsys-rep 2>/dev/null" >"$LOGDIR/${a}.kern.csv"
        python3 - "$LOGDIR/${a}.kern.csv" "$LOGDIR/${a}.p7" "$a" <<'PY'
import csv, sys
csv_p, out_p, arm = sys.argv[1], sys.argv[2], sys.argv[3]
want = []
if arm in ("T1-B", "T1-D"):
    want.append("hc_collapse_norm_split")
if arm in ("T1-C", "T1-D"):
    want.append("rmsnorm_rows_split")
counts = {w: 0 for w in want}
try:
    for r in csv.reader(open(csv_p, errors="ignore")):
        if len(r) < 3:
            continue
        name = r[-1].strip()
        try:
            inst = int(r[2])
        except ValueError:
            continue
        for w in want:
            if w in name:
                counts[w] += inst
except OSError:
    pass
lines = ["%s inst=%d" % (w, counts[w]) for w in want]
ok = all(counts[w] > 0 for w in want) if want else False
lines.append("P7: %s" % ("PASS" if ok else "FAIL"))
open(out_p, "w").write("\n".join(lines) + "\n")
print("   P7 %s: %s" % (arm, "  ".join(lines[:-1]) + "  -> " + lines[-1]))
PY
    done
    echo "-- P7 artifacts: $LOGDIR/<arm>.{kern.csv,p7}"
    echo
    echo "NOTE: --nsys is a SEPARATE pass (it pollutes the step wall). It ran no A/B arm,"
    echo "      so the Δ judge is NOT run here — run the plain (no --nsys) pass for P0..P6/P8,"
    echo "      and this pass for P7. Both write into $LOGDIR, which is why --judge-only"
    echo "      can fold the two together afterwards."
    exit 0
fi

# ===========================================================================
# 7. The Δ table + the P0..P8 verdict + the §5.3 decision, per arm.
# ===========================================================================
echo
python3 - "$LOGDIR" "$MIN_GAIN_PCT" "$SIGMA_FLOOR" "$TOKEN_MATCH_CHARS" <<'PY' | tee "$LOGDIR/table.txt"
import hashlib
import os
import sys

LOGDIR = sys.argv[1]
MIN_GAIN_PCT = float(sys.argv[2])       # the 0.3% leg
SIGMA_FLOOR = float(sys.argv[3])        # sigma >= this -> skip everything
NEED = int(sys.argv[4])                 # P6 leading-char agreement

ARMS = ["T1-A", "T1-A2", "T1-B", "T1-C0", "T1-C", "T1-D", "T1-D2"]
CTRL = {"T1-A2": "T1-A", "T1-B": "T1-A", "T1-C0": "T1-A",
        "T1-C": "T1-C0", "T1-D": "T1-C0", "T1-D2": "T1-D"}
DESC = {"T1-A": "control 91.1", "T1-A2": "control x2", "T1-B": "+CNORM",
        "T1-C0": "+MROWS", "T1-C": "+MROWS+SPLIT", "T1-D": "+both",
        "T1-D2": "D x2"}
SPLIT_ARM = {"T1-B": True, "T1-C": True, "T1-D": True}


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


def txt(tag, kind):
    try:
        return open(os.path.join(LOGDIR, "%s.%s.txt" % (tag, kind)), errors="replace").read()
    except OSError:
        return None


def prefix(base, arm):
    if base is None or arm is None:
        return None
    n = min(len(base), len(arm))
    i = 0
    while i < n and base[i] == arm[i]:
        i += 1
    return i


M = {a: load(a) for a in ARMS}
CT = {"di": txt("T1-A", "di"), "sh": txt("T1-A", "sh")}

print("==== L4-9 (CNORM/NORM dim-split) A/B — 7 arms, 91.1 stack ====")
print("%-6s %-16s %8s %8s %7s %6s %7s %6s %6s %s" % (
    "ARM", "VARIANT", "ok_lines", "firstbd", "steady", "estT/s", "kacc_md", "latin", "sh_lat", "TEXT"))
rc = 0

rows = {}
for a in ARMS:
    d = M[a]

    def g(k):
        return d.get(k, "NA")

    steady = num(d, "di_steady_mean")
    km = num(d, "di_kacc_mean")
    # est tok/s = (k_acc+1) tokens per round / (ms per round)
    est = None
    if steady and steady > 0 and km is not None:
        est = (km + 1.0) / (steady / 1000.0)
    rows[a] = dict(steady=steady, est=est, ok=num(d, "di_ok_lines"),
                   fb=num(d, "di_first_bad"))
    print("%-6s %-16s %8s %8s %7s %6s %7s %6s %6s %s" % (
        a, DESC[a], g("di_ok_lines"), g("di_first_bad"),
        g("di_steady_mean"), "NA" if est is None else "%.1f" % est,
        g("di_kacc_md5")[:8], g("di_latin"), g("sh_latin"), g("sh_has_kaishen")))

print()
print("P6 leading-char agreement (vs T1-A control):")
P6 = {}
for a in ARMS:
    di_p = prefix(CT["di"], txt(a, "di"))
    sh_p = prefix(CT["sh"], txt(a, "sh"))
    P6[a] = (di_p, sh_p)
    print("  %-6s counting=%s  出师表=%s" % (
        a, "NA" if di_p is None else di_p, "NA" if sh_p is None else sh_p))

# ---- P0 USABLE --------------------------------------------------------------
print()
print("判据:")
unmeasured = [a for a in ARMS if rows[a]["steady"] is None]
empty = [a for a in ARMS if M[a].get("di_chars") in (None, "0", "")]
if unmeasured:
    print("  P0 USABLE  FAIL: %s printed no countable `[dsv41] step pos=` wall "
          "(was DSV41_TIMING=1 effective?) — an unmeasured arm cannot carry any Δ." % ",".join(unmeasured))
    rc = 2
elif empty:
    print("  P0 USABLE  FAIL: %s produced an empty counting answer — the harness, not the kernel." % ",".join(empty))
    rc = 2
else:
    print("  P0 USABLE  OK: all 7 arms measured, every counting answer non-empty.")

# ---- P1 计数 ----------------------------------------------------------------
p1_fail = []
for a in ARMS:
    ok = rows[a]["ok"] or 0
    ctrl = CTRL.get(a)
    ctrl_ok = (rows[ctrl]["ok"] or 0) if ctrl else 0
    if ok < 61 or (ctrl and ok < ctrl_ok):
        p1_fail.append(a)
print("  P1 计数   %s: first 61 lines == 1..61 (per arm ok_lines): %s" % (
    "OK" if not p1_fail else "FAIL -> " + ",".join(p1_fail),
    " ".join("%s=%s" % (a, M[a].get("di_ok_lines")) for a in ARMS)))
if p1_fail:
    for a in p1_fail:
        print("            %s first_bad=%s (line %r)" % (
            a, M[a].get("di_first_bad"), M[a].get("di_first_bad_line")))
    rc = max(rc, 1)

# ---- P2 出师表 ---------------------------------------------------------------
ctrl_lat = num(M["T1-A"], "sh_latin") or 0
p2_fail = [a for a in ARMS
           if (num(M[a], "sh_latin") or 0) > ctrl_lat or M[a].get("sh_has_kaishen") != "yes"]
print("  P2 出师表  %s: latin(arm) <= latin(T1-A)=%d and 先帝创业未半 present" % (
    "OK" if not p2_fail else "FAIL -> " + ",".join(p2_fail), ctrl_lat))
if p2_fail:
    for a in p2_fail:
        print("            %s latin=%s has_kaishen=%s samples=%r" % (
            a, M[a].get("sh_latin"), M[a].get("sh_has_kaishen"), M[a].get("sh_latin_samples")))
    rc = max(rc, 1)

# ---- P3 k_acc ---------------------------------------------------------------
base10 = M["T1-A"].get("di_kacc10_md5")
base_k = num(M["T1-A"], "di_kacc_mean")
p3_fail = []
for a in ARMS:
    if a == "T1-A":
        continue
    if M[a].get("di_kacc10_md5") != base10:
        p3_fail.append(a)
    elif base_k is not None and num(M[a], "di_kacc_mean") is not None \
            and abs(num(M[a], "di_kacc_mean") - base_k) >= 0.05:
        p3_fail.append(a)
print("  P3 k_acc  %s: first-10 seq md5 == T1-A (%s) and |Δmean_k| < 0.05" % (
    "OK" if not p3_fail else "FAIL -> " + ",".join(p3_fail), base10))
if p3_fail:
    rc = max(rc, 1)

# ---- P5 hang ----------------------------------------------------------------
hang = {}
for a in ARMS:
    try:
        log = open(os.path.join(LOGDIR, a + ".log"), errors="ignore").read()
    except OSError:
        log = ""
    hang[a] = log.count("ar5-hang") + log.count("step err")
p5_fail = [a for a in ARMS if hang[a] > 0]
print("  P5 hang   %s: %s" % (
    "OK" if not p5_fail else "FAIL -> " + ",".join(p5_fail),
    " ".join("%s=%d" % (a, hang[a]) for a in ARMS)))
if p5_fail:
    rc = max(rc, 1)

# ---- P6 内容 ----------------------------------------------------------------
p6_fail = [a for a in ARMS if P6[a][0] == 0 or P6[a][1] == 0]
p6_warn = [a for a in ARMS
           if (P6[a][0] is not None and 0 < P6[a][0] < NEED)
           or (P6[a][1] is not None and 0 < P6[a][1] < NEED)]
print("  P6 内容   %s: leading chars >= %d on BOTH counting and 出师表 vs T1-A" % (
    "OK" if not p6_fail else "FAIL -> " + ",".join(p6_fail), NEED))
if p6_warn:
    print("            WARN (inspect by eye): %s" % ",".join(p6_warn))
if p6_fail:
    rc = max(rc, 1)

# ---- P7 活性 (only when --nsys produced the artifacts) ----------------------
have_p7 = any(os.path.exists(os.path.join(LOGDIR, a + ".p7")) for a in SPLIT_ARM)
if have_p7:
    for a in SPLIT_ARM:
        p = os.path.join(LOGDIR, a + ".p7")
        if not os.path.exists(p):
            print("  P7 活性   %s: MISSING (.p7 absent) — the split kernel's activity is UNPROVEN" % a)
            rc = max(rc, 1)
            continue
        body = open(p).read().strip().replace("\n", "  ")
        bad = "FAIL" in body
        print("  P7 活性   %s: %s" % (a, body))
        if bad:
            rc = max(rc, 1)
else:
    print("  P7 活性   SKIPPED: run `--nsys` for the split-kernel invocation proof "
          "(a text-identical arm does NOT prove the split kernel ran).")

# ---- P8 确定性 ---------------------------------------------------------------
d, d2 = M["T1-D"], M["T1-D2"]
det = (d.get("di_md5") == d2.get("di_md5") and d.get("sh_md5") == d2.get("sh_md5"))
print("  P8 确定性  %s: T1-D vs T1-D2 byte-identical (counting md5 %s/%s, 出师表 md5 %s/%s)" % (
    "OK" if det else "FAIL",
    str(d.get("di_md5"))[:8], str(d2.get("di_md5"))[:8],
    str(d.get("sh_md5"))[:8], str(d2.get("sh_md5"))[:8]))
if not det:
    print("            A mismatch can only be device-global scratch cross-talk / a race — do NOT ship.")
    rc = max(rc, 1)

# ---- 噪声地板 sigma ----------------------------------------------------------
sa, sa2 = rows["T1-A"]["steady"], rows["T1-A2"]["steady"]
sigma = None
if sa and sa2:
    sigma = abs(sa - sa2) / sa * 100.0
print()
sig_s = "NA" if sigma is None else "%.3f%%" % sigma
print("  噪声地板: steady(T1-A)=%s ms  steady(T1-A2)=%s ms  sigma=%s%s" % (
    M["T1-A"].get("di_steady_mean"), M["T1-A2"].get("di_steady_mean"), sig_s,
    "  (>= %.2f%% -> 噪声地板过高, skip everything)" % SIGMA_FLOOR if sigma is not None and sigma >= SIGMA_FLOOR else ""))

# ---- DECISION (§5.3) --------------------------------------------------------
print()
print("  决策 (§5.3): Δ%% = (steady_ctrl - steady_arm)/steady_ctrl * 100;  Δ%% > max(1σ, %.1f%%) -> 入栈" % MIN_GAIN_PCT)
if sigma is None:
    print("    sigma unmeasured (T1-A or T1-A2 missing) — cannot decide.")
    rc = max(rc, 2)
else:
    threshold = max(sigma, MIN_GAIN_PCT)
    if sigma >= SIGMA_FLOOR:
        print("    σ=%.3f%% >= %.1f%% -> 噪声地板过高: SKIP the whole batch (design §5.3 exit)." % (sigma, SIGMA_FLOOR))
    for a in ("T1-B", "T1-C0", "T1-C", "T1-D"):
        ctrl = CTRL[a]
        sc, sm = rows[ctrl]["steady"], rows[a]["steady"]
        if sc is None or sm is None:
            print("    %-6s vs %-6s: UNMEASURED" % (a, ctrl))
            rc = max(rc, 2)
            continue
        delta = (sc - sm) / sc * 100.0
        if sigma >= SIGMA_FLOOR:
            verdict = "SKIP (噪声地板过高)"
        elif delta <= -MIN_GAIN_PCT:
            verdict = "OFF (负收益)"
        elif delta <= threshold:
            verdict = "SKIP (噪声内)"
        else:
            verdict = "入栈 (gate 保持 ON)"
        redline = "redline-FAIL" if (a in p1_fail or a in p2_fail or a in p3_fail or a in p6_fail) else "redline-ok"
        print("    %-6s vs %-6s: %s -> %s ms  Δ%%+ = %+.3f%%  (%s)  [%s]" % (
            a, ctrl, "%.2f" % sc, "%.2f" % sm, delta, verdict, redline))
    # C0 is its own (entry) reading vs A, shown for the §5.4 note.
    if rows["T1-C0"]["steady"] and rows["T1-A"]["steady"]:
        dl = (rows["T1-A"]["steady"] - rows["T1-C0"]["steady"]) / rows["T1-A"]["steady"] * 100.0
        print("    (note) T1-C0 vs T1-A: %+.3f%%  — the NORM_MROWS entry itself" % dl)
    # D vs A, alongside.
    if rows["T1-D"]["steady"] and rows["T1-A"]["steady"]:
        dl = (rows["T1-A"]["steady"] - rows["T1-D"]["steady"]) / rows["T1-A"]["steady"] * 100.0
        print("    (note) T1-D vs T1-A: %+.3f%%" % dl)

print()
if rc == 2:
    print("verdict: rc=2 — NOT USABLE (an arm is unmeasured / harness error); fix the harness before reading Δ.")
elif rc == 1:
    print("verdict: rc=1 — a RED LINE / P3 / P6 / P8 leg failed; do NOT move the default.")
else:
    print("verdict: rc=0 — usable, red lines held; read the Δ table + DECISION to pick the gate.")
sys.exit(rc)
PY
verdict=$?

echo
echo "Logs: $LOGDIR/<arm>.{env,di.log,sh.log,log,di.resp.json,sh.resp.json,di.txt,sh.txt,metrics,kacc}   table: $LOGDIR/table.txt"
exit "$verdict"
