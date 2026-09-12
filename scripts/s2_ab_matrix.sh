#!/usr/bin/env bash
# s2_ab_matrix.sh — S2: the SINGLE-VARIABLE A/B of TAP_BF16 and DRAFT_ATTN_BF16.
#
# WHY THIS SCRIPT EXISTS
#   The 0.898 regression (`dspark-correctness-chain.md` §1339) was a 3-gate bundle
#   (`SEED_POS + DRAFT_ATTN_BF16 + TAP_BF16`) measured on a build whose SEED_POS arm
#   had a q/o-vs-kv RoPE phase mismatch. That mismatch is a first-order corruption on
#   the draft's attention rows, so it swamped whatever the two dtype gates did — the
#   "these gates must stay OFF" conclusion has NO isolated measurement behind it.
#   S1 removed the confound. This script is the first clean 2x2 factorial on the
#   fixed baseline: TAP_BF16 x DRAFT_ATTN_BF16, one variable per arm.
#
# THE MATRIX (4 arms, exactly one bit differs from A in each of B/C/D)
#   | arm | DSV41_TAP_BF16 | DSV41_DRAFT_ATTN_BF16 | reads as                     |
#   |-----|----------------|-----------------------|------------------------------|
#   |  A  | OFF            | OFF                   | the baseline (reference)     |
#   |  B  | ON             | OFF                   | TAP_BF16 alone               |
#   |  C  | OFF            | ON                    | DRAFT_ATTN_BF16 alone        |
#   |  D  | ON             | ON                    | the combination              |
#   Verdicts are B-A, C-A, D-A (and, for the interaction, D-(B+C-A)).
#
# ⚠️ SEED_POS AND THE S1 RESULT — READ BEFORE SETTING BASE_SEED_POS
#   The S2 brief (accept-first-strategy §3-S2) was written BEFORE S1 was run, and it
#   asked for `DSV41_SEED_POS=1` in every arm. S1 has since been run and its result is
#   committed (e27ad40 / `dspark-correctness-chain.md` §1604): **SEED_POS with the q/o
#   fix DEGRADES accept (mean-k 1.067 vs the best 1.214), zero-latin still holds, and
#   the repo's own conclusion is "SEED_POS stays OFF; next is S2 (TAP_BF16/DRAFT_ATTN
#   single-variable A/B, 无 SEED_POS)".**
#   Running S2 with SEED_POS=1 would therefore measure the two gates on a *degraded*
#   baseline — exactly the confound this script exists to remove. The default is
#   therefore BASE_SEED_POS=0 (the 1.214 baseline). To reproduce the brief literally:
#       BASE_SEED_POS=1 bash scripts/s2_ab_matrix.sh
#   The run prints a banner naming which mode it is in; the reader is never left to
#   reconstruct it.
#
# FIXED BASE (identical in all four arms — only the two gate bits move)
#   DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1   (spec decoding — required)
#   DSV41_EXPERT_ACT_E4M3=1 DSV41_SH_EXP_MROWS=1 DSV41_DRAFT_P3A=1  (canonical lazy arm)
#   DSV41_TIMING=1                                       (REQUIRED: no `[dspark] steps=` line without it)
#   DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1             (the S1/lazy arm)
#   DSV41_BF16_TRUNCATE=1                                (零拉丁红线, non-negotiable)
#   DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1          (P0-3 + P1-5 = the 1.214 levers)
#   [+ DSV41_SEED_POS=1  iff  BASE_SEED_POS=1]
#
# 口径 (the project rules this script encodes)
#   1. SERIAL, ONE SERVE AT A TIME, torn down between arms — two serves on the same
#      8 GPUs do not give noisy numbers, they give meaningless ones.
#   2. ONE PROMPT PER SERVE: the `[dspark]` accumulators are process-level and never
#      reset per request, so two requests in one serve would mix averages.
#   3. SAME BINARY + SAME .so ON EVERY ARM, PROVEN (the loader refuses a mismatched
#      pair; `cargo build` alone can never heal a stale `.so`).
#   4. The running env is read back off /proc and checked: the arm's own gate MUST be
#      present and the OTHER gate MUST be absent. A leaked bit means the 2x2 is not a
#      2x2 — the arm is aborted, not silently averaged in.
#   5. NO rsync. The tree on the node is the caller's and is assumed already synced
#      (手动同步已确认代码一致); a revision mismatch is a loud WARNING, never a sync.
#
# k_acc 直方图
#   `DSV41_LAZY_VERIFY` has no dedicated histogram print, but the per-round
#   `[dsv41] step pos=…` lines carry it for free: the engine advances the position by
#   `k_emit = k_acc + 1` per committed round, so the DELTA between consecutive
#   positions MINUS ONE is that round's k_acc. The parser buckets those (0..6) and
#   reports the histogram-derived mean-k beside the serve's own `mean-k` as a
#   cross-check — no new env knob needed.
#
# USAGE
#   bash scripts/s2_ab_matrix.sh                 # 4 serial serves, reuse the pair on the node
#   bash scripts/s2_ab_matrix.sh --build         # rebuild BOTH products first, then run
#   bash scripts/s2_ab_matrix.sh --dry-run       # pre-flight only; launches nothing
#   BASE_SEED_POS=1 bash scripts/s2_ab_matrix.sh # the brief's literal config (see the banner)
#   MAXTOK=1000 PORT_BASE=8230 bash scripts/s2_ab_matrix.sh
#
# OUTPUT: per arm — the full 出师表 answer, the k_acc histogram, mean-k, tok/step and
#   the red-line check; then the 2x2 Δ table and the verdict framework (B-A, C-A,
#   D-A, interaction). Logs: $LOGDIR/<tag>.{log,dspark,metrics,txt,env,resp.json}
#   Exit: 0 = all four arms usable and red-line clean; 1 = a red line broke;
#         2 = the run is not usable (unmeasured arm / leaked gate bit / harness error).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

NODE="${NODE:-ubuntu@43.202.208.136}"
ARCH="${ARCH:-103a}"
RROOT="${RROOT:-ferrite}"
LOGDIR="${S2_LOGDIR:-/tmp/s2_ab_matrix}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
CURL="curl -s --noproxy '*'"

PORT_BASE="${PORT_BASE:-8230}"
MAXTOK="${MAXTOK:-1000}"                        # 出师表, full answer, no truncation
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"              # x5s
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"
STEADY_SKIP="${STEADY_SKIP:-20}"                # warm-up rounds dropped before the steady stats
EPS="${EPS:-0.05}"                              # |Δmean-k| below this is reported as "null"

# The S1 outcome is committed (see the banner above): SEED_POS OFF is the 1.214
# baseline. Flip to 1 only to reproduce the brief's literal (pre-S1) config.
BASE_SEED_POS="${BASE_SEED_POS:-0}"

# 出师表 is the project's long-prompt yardstick (dspark_verify.rs PROMPTS): a long
# prompt + recitation, so it is sensitive to cumulative corruption AND runs long
# enough (>= 50 dspark steps) to print the `[dspark] steps=` line.
PROMPT="${PROMPT:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"

BUILD=0; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --build)    BUILD=1 ;;
        --no-build) BUILD=0 ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,80p' "$0"; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)"; exit 2 ;;
    esac
    shift
done

mkdir -p "$LOGDIR"

rssh() { ssh "${SSH_OPTS[@]}" "$NODE" "$1"; }

# One writer at a time, across invocations too: a second serve while the first is up
# is exactly the failure this script exists to prevent.
LOCK="$LOGDIR/.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "FATAL: another s2_ab_matrix.sh holds $LOCK — arms must never overlap."
    exit 2
fi

# The FIXED base, one variable per arm. A string so it travels the ssh env intact.
BASE_ENV="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
DSV41_EXPERT_ACT_E4M3=1 DSV41_SH_EXP_MROWS=1 DSV41_DRAFT_P3A=1 \
DSV41_TIMING=1 DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1 \
DSV41_BF16_TRUNCATE=1 DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1"
[ "$BASE_SEED_POS" = 1 ] && BASE_ENV="$BASE_ENV DSV41_SEED_POS=1"

echo "== S2: TAP_BF16 x DRAFT_ATTN_BF16 single-variable A/B (4 arms) =="
echo "-- node $NODE   arch $ARCH   tp $TP   ports $PORT_BASE..$(( PORT_BASE + 3 ))"
echo "-- prompt: 出师表 max_tokens=$MAXTOK   steady-skip=$STEADY_SKIP   |Δ| gate ${EPS}"
if [ "$BASE_SEED_POS" = 1 ]; then
    echo "-- ⚠️  BASE_SEED_POS=1: this measures the two gates on the DEGRADED SEED_POS"
    echo "--     baseline (S1 result: mean-k 1.067 vs 1.214). Kept only to reproduce"
    echo "--     the pre-S1 brief; the default (0) is the clean 1.214 baseline."
else
    echo "-- BASE_SEED_POS=0: the clean baseline (S1 result: SEED_POS stays OFF, best 1.214)."
fi
echo "-- base env (identical in all four arms): $BASE_ENV"
echo "-- arm A: +<none>                    (TAP_BF16=0 DRAFT_ATTN_BF16=0)"
echo "-- arm B: +DSV41_TAP_BF16=1          (TAP_BF16=1 DRAFT_ATTN_BF16=0)"
echo "-- arm C: +DSV41_DRAFT_ATTN_BF16=1   (TAP_BF16=0 DRAFT_ATTN_BF16=1)"
echo "-- arm D: +DSV41_TAP_BF16=1 DSV41_DRAFT_ATTN_BF16=1"

# ---------------------------------------------------------------------------
# 0. Node reachability + revision on each side (a mismatch is a WARNING: this
#    script does NOT rsync — the node tree is the caller's, already synced).
# ---------------------------------------------------------------------------
REV_LOCAL="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && REV_LOCAL="$REV_LOCAL-dirty"
if [ "$DRY" != 1 ]; then
    rssh "true" >/dev/null 2>&1 || { echo "FATAL: cannot reach $NODE"; exit 2; }
fi
REV_NODE="$(rssh "git -C ~/$RROOT rev-parse --short HEAD 2>/dev/null || echo '?'" 2>/dev/null)"
[ -n "$(rssh "git -C ~/$RROOT status --porcelain 2>/dev/null" 2>/dev/null)" ] && REV_NODE="$REV_NODE-dirty"
echo "-- rev: local $REV_LOCAL   node $REV_NODE   (no rsync by design)"
if [ "$REV_LOCAL" != "$REV_NODE" ]; then
    echo "   WARN: the node tree is a DIFFERENT revision from this checkout, and it is"
    echo "         what will be measured. The caller owns that tree; sync it by hand if"
    echo "         that is not intended (this script never rsyncs)."
fi

# ---------------------------------------------------------------------------
# 1. Optional build (OFF by default — the caller confirms the pair is already
#    synced), then PROVE the pair is same-source.
# ---------------------------------------------------------------------------
SO_REL="kernels/cuda/libferrite_kernels.so"
BIN_REL="target/release/ferrite-serve"
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
# the id WAS found — the gate would misfire on a good pair.
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

# ---------------------------------------------------------------------------
# 2. Dry run stops here: nothing was built and no serve ran.
# ---------------------------------------------------------------------------
if [ "$DRY" = 1 ]; then
    free="$(rssh "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END {print s+0}'")"
    running="$(rssh 'pgrep -x ferrite-serve | wc -l')"
    echo "-- dry run: pair same-source $([ "$PAIR_OK" = 1 ] && echo YES || echo 'NO (a real run would need --build)')"
    echo "-- dry run: GPU memory in use = ${free} MiB   ferrite-serve running = $running (both must be 0 for a clean run)"
    echo "-- dry run: a real run would start 4 serial serves on ports $PORT_BASE..$(( PORT_BASE + 3 )); nothing was launched."
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. Teardown by EXACT process name: `pkill -f ferrite-serve` would match this very
#    command line (it contains the string) and kill the ssh session.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 4. The per-arm parser: read (dspark, resp.json, log) and write a
#    `<tag>.metrics` key=value file + `<tag>.txt` (the FULL answer, no truncation).
#    Everything numeric the report needs is derived here, once.
# ---------------------------------------------------------------------------
metrics_of() {  # tag
    python3 - "$LOGDIR/$1.dspark" "$LOGDIR/$1.resp.json" "$LOGDIR/$1.log" \
               "$LOGDIR/$1.metrics" "$LOGDIR/$1.txt" "$STEADY_SKIP" <<'PY'
import hashlib
import json
import re
import sys

dspark_p, resp_p, log_p, out_p, txt_p = sys.argv[1:6]
skip = int(sys.argv[6])


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
#      the LAST one is the run's cumulative average (printed every 50 steps).
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

# ---- k_acc histogram from the position deltas: the engine advances by
#      k_emit = k_acc + 1 per committed round, so delta - 1 is that round's k_acc.
deltas = [seq[i + 1][0] - seq[i][0] for i in range(len(seq) - 1)]
kacc = [d - 1 for d in deltas if 1 <= d <= 7]
hist = {k: kacc.count(k) for k in range(0, 7)}
kacc_mean = (sum(kacc) / len(kacc)) if kacc else None

# ---- text: the FULL answer + the project's three red lines -------------------
content = ""
resp_err = ""
try:
    content = json.loads(read(resp_p))["choices"][0]["message"]["content"]
except Exception as exc:  # noqa: BLE001
    resp_err = str(exc)
with open(txt_p, "w") as fh:
    fh.write(content)

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
if has_kaishen != "yes":
    fails.append("missing 先帝创业未半")
if dbl:
    fails.append("%d adjacent double-char" % dbl)
if latin:
    fails.append("%d latin char(s)" % latin)
text_ok = "ok" if not fails else "; ".join(fails)


def fmt(x, spec="%.3f"):
    return "NA" if x is None else spec % x


with open(out_p, "w") as fh:
    for k, v in [
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
        ("kacc_mean", fmt(kacc_mean)),
        ("kacc_n", str(len(kacc))),
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

# ---------------------------------------------------------------------------
# 5. One serve per arm, strictly serial. `extra` is the arm's single-variable
#    delta (empty for A, one bit for B/C, both bits for D); `forbidden` is the gate
#    that MUST NOT appear in the running env (the 2x2 integrity check). A leaked bit
#    turns B/C into D and destroys the attribution, so the arm is aborted, never
#    averaged in.
# ---------------------------------------------------------------------------
run_case() {  # arm tag port extra forbidden_gate
    local arm="$1" tag="$2" port="$3" extra="$4" forbidden="$5"
    local rlog="~/s2_${tag}.log"
    local genv="$BASE_ENV"
    [ -n "$extra" ] && genv="$genv $extra"

    echo
    echo "---- arm $arm ($tag, port $port): TAP_BF16=$( [[ "$genv" == *DSV41_TAP_BF16=1* ]] && echo 1 || echo 0 ) DRAFT_ATTN_BF16=$( [[ "$genv" == *DSV41_DRAFT_ATTN_BF16=1* ]] && echo 1 || echo 0 )"

    kill_serves >/dev/null                    # every case starts from a clean box
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

    # 2x2 integrity: the other gate must NOT be in the running env, and every bit in
    # `extra` must have survived into it.
    if [ -n "$forbidden" ] && grep -q "^${forbidden}=" "$LOGDIR/${tag}.env"; then
        echo "   FATAL: $forbidden is set in the running serve but arm $arm must have it OFF."
        teardown "$port"; exit 2
    fi
    for gate in $extra; do
        if ! grep -q "^${gate}$" "$LOGDIR/${tag}.env"; then
            echo "   FATAL: arm $arm's bit $gate did not survive into the serve's env."
            teardown "$port"; exit 2
        fi
    done

    local body t0 t1
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
            "$MODEL_NAME" "$PROMPT" "$MAXTOK")"
    t0=$(date +%s)
    rssh "cd ~/$RROOT && $CURL -m 1800 http://localhost:$port/v1/chat/completions \
          -H 'Content-Type: application/json' -d '$body'" >"$LOGDIR/${tag}.resp.json" 2>/dev/null
    t1=$(date +%s)
    rssh "grep 'dspark] steps' $rlog | tail -1" >"$LOGDIR/${tag}.dspark" 2>/dev/null
    rssh "cat $rlog" >"$LOGDIR/${tag}.log" 2>/dev/null

    teardown "$port"
    echo "   e2e $(( t1 - t0 ))s   dspark: $(head -c 160 "$LOGDIR/${tag}.dspark")"

    metrics_of "$tag"
    [ -s "$LOGDIR/${tag}.metrics" ] \
        || { echo "FATAL: parser wrote no metrics for $tag (see $LOGDIR/${tag}.log)"; exit 2; }
    echo "   $(grep -E '^(steps|mean_k|kacc_mean|tok_step|steady_mean|rounds|chars|dbl|latin|text_ok)=' "$LOGDIR/${tag}.metrics" | paste -sd' ' -)"
    echo "   k_acc hist [0..6]: $(grep -E '^hist[0-6]=' "$LOGDIR/${tag}.metrics" | cut -d= -f2 | paste -sd' ' -)"

    echo "   ---- full text ($tag), $(wc -c <"$LOGDIR/${tag}.txt") bytes ----"
    cat "$LOGDIR/${tag}.txt"
    echo
    echo "   ---- end full text ($tag) ----"
}

echo
echo "== arms (serial, same binary+.so, one serve per arm) =="
#  arm  tag  port                extra env (the arm's delta)                                  forbidden (must stay OFF)
run_case A A0 "$PORT_BASE"                ""                                                    ""
run_case B B0 "$(( PORT_BASE + 1 ))"      "DSV41_TAP_BF16=1"                                    DSV41_DRAFT_ATTN_BF16
run_case C C0 "$(( PORT_BASE + 2 ))"      "DSV41_DRAFT_ATTN_BF16=1"                             DSV41_TAP_BF16
run_case D D0 "$(( PORT_BASE + 3 ))"      "DSV41_TAP_BF16=1 DSV41_DRAFT_ATTN_BF16=1"            ""

# ---------------------------------------------------------------------------
# 6. The 2x2 table + the verdict framework. Δ is arm minus A (the reference).
# ---------------------------------------------------------------------------
echo
python3 - "$LOGDIR" "$EPS" "$BASE_SEED_POS" <<'PY' | tee "$LOGDIR/table.txt"
import os
import sys

LOGDIR, eps, seed_pos = sys.argv[1], float(sys.argv[2]), sys.argv[3]
CASES = [("A0", "A", "0", "0"), ("B0", "B", "1", "0"),
         ("C0", "C", "0", "1"), ("D0", "D", "1", "1")]


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


M = {tag: load(tag) for tag, _, _, _ in CASES}
A = M["A0"]

print("== S2 2x2 — TAP_BF16 x DRAFT_ATTN_BF16 (baseline SEED_POS=%s) ==" % seed_pos)
print("%-3s %-9s %-15s %7s %7s %8s %8s %6s %5s %6s %s" % (
    "ARM", "TAP_BF16", "DRAFT_ATTN_BF16", "steps", "mean-k", "hist-k", "tok/step",
    "chars", "dbl", "latin", "TEXT"))
for tag, arm, tap, attn in CASES:
    d = M[tag]

    def g(k):
        return d.get(k, "NA")

    print("%-3s %-9s %-15s %7s %7s %8s %8s %6s %5s %6s %s" % (
        arm, tap, attn, g("steps"), g("mean_k"), g("kacc_mean"), g("tok_step"),
        g("chars"), g("dbl"), g("latin"), g("text_ok")))

print()
print("k_acc histogram (0..6) — derived from the [dsv41] step pos= deltas:")
for tag, arm, tap, attn in CASES:
    d = M[tag]
    h = " ".join("%d:%s" % (k, d.get("hist%d" % k, "0")) for k in range(7))
    print("  arm%s (TAP=%s ATTN=%s)  n=%-4s mean-k(hist)=%-6s  %s"
          % (arm, tap, attn, d.get("kacc_n", "NA"), d.get("kacc_mean", "NA"), h))

print()
print("判据:")
rc = 0

# --- USABLE: every arm needs the `[dspark] steps=` line (DSV41_TIMING effective).
unmeasured = [arm for tag, arm, _, _ in CASES if num(M[tag], "mean_k") is None]
if unmeasured:
    print("  USABLE  FAIL: arm(s) %s printed no '[dspark] steps=' line — was DSV41_TIMING=1 effective?"
          % ",".join(unmeasured))
    print("          A run with an unmeasured arm cannot carry any Δ.")
    rc = 2
else:
    print("  USABLE  OK: all four arms printed '[dspark] steps=' (DSV41_TIMING effective).")

# --- RED LINE: zero latin / 先帝创业未半 / no double-char, on EVERY arm.
bad_text = [arm for tag, arm, _, _ in CASES if M[tag].get("text_ok") != "ok"]
if bad_text:
    print("  REDLINE FAIL: arm(s) %s — %s" % (
        ",".join(bad_text), "; ".join("%s: %s" % (a, M[t].get("text_ok"))
                                      for t, a, _, _ in CASES if M[t].get("text_ok") != "ok")))
    rc = max(rc, 1)
else:
    print("  REDLINE OK: all four arms zero-latin, 先帝创业未半 present, no double-char.")
    for tag, arm, _, _ in CASES:
        if M[tag].get("latin") not in ("0", None):
            print("          arm%s latin samples: %s" % (arm, M[tag].get("latin_samples")))

# --- EFFECTS: B-A (TAP), C-A (ATTN), D-A (combo); interaction = D-(B+C-A).
def delta(tag):
    a, b = num(A, "mean_k"), num(M[tag], "mean_k")
    return None if (a is None or b is None) else b - a


def verdict(d):
    if d is None:
        return "UNMEASURED"
    if d > eps:
        return "IMPROVES (+%.3f)" % d
    if d < -eps:
        return "DEGRADES (%+.3f)" % d
    return "NULL (%.3f, |Δ|<=%.2f)" % (d, eps)


dB, dC, dD = delta("B0"), delta("C0"), delta("D0")
print()
print("  EFFECTS (mean-k, arm minus A; eps=%.2f):" % eps)
print("    B-A  TAP_BF16 alone        : A %s -> B %s   Δ %s   %s"
      % (A.get("mean_k"), M["B0"].get("mean_k"),
         "NA" if dB is None else "%+.3f" % dB, verdict(dB)))
print("    C-A  DRAFT_ATTN_BF16 alone : A %s -> C %s   Δ %s   %s"
      % (A.get("mean_k"), M["C0"].get("mean_k"),
         "NA" if dC is None else "%+.3f" % dC, verdict(dC)))
print("    D-A  combination           : A %s -> D %s   Δ %s   %s"
      % (A.get("mean_k"), M["D0"].get("mean_k"),
         "NA" if dD is None else "%+.3f" % dD, verdict(dD)))
if None not in (dB, dC, dD):
    inter = dD - (dB + dC)
    print("    interaction D-(B+C-A)      : %+.3f   (%s)"
          % (inter, "≈additive" if abs(inter) <= eps else
             "super-additive" if inter > 0 else "sub-additive (overlap/cancellation)"))

print()
print("  读判 (the S2 hypothesis, spelled out so the numbers are not over-read):")
print("    * If B-A and C-A are BOTH <=0 (null or negative), the 'append the dtype gate'")
print("      idea is dead on a CLEAN baseline — the earlier 0.898 was NOT just SEED_POS")
print("      masking a real gain. Send the budget to S3/S4 instead.")
print("    * If B-A or C-A > 0, that gate is a real accept lever and belongs in the")
print("      400 matrix; re-run it there for the throughput number.")
print("    * The single-sided-alignment warning (accept-first-strategy §1.1) predicts a")
print("      DEGRADING gate when the verify side is not moved with it; a positive Δ here")
print("      would falsify that for the gate in question (it is the first isolated test).")
print("    * D-A vs (B-A)+(C-A): non-additivity means the two gates share a mechanism.")

print()
if rc == 2:
    print("verdict: rc=2 — NOT USABLE (an arm is unmeasured); fix the harness before reading Δ.")
elif rc == 1:
    print("verdict: rc=1 — a RED LINE broke; the accept numbers are second-order until it is fixed.")
else:
    print("verdict: rc=0 — usable and red-line clean; read the EFFECTS table above.")
sys.exit(rc)
PY
verdict=$?

echo
if [ "$verdict" = 2 ]; then
    echo "s2_ab_matrix: NOT USABLE — read the USABLE line above and the per-arm logs."
    exit 2
fi
if [ "$verdict" = 1 ]; then
    echo "s2_ab_matrix: a red line broke — investigate before trusting any Δ."
    exit 1
fi
echo "s2_ab_matrix: 4 arms usable, red-line clean. Baseline SEED_POS=$BASE_SEED_POS."
echo "Logs: $LOGDIR/<tag>.{log,dspark,metrics,txt,env,resp.json}   table: $LOGDIR/table.txt"
exit 0
