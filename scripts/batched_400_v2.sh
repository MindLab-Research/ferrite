#!/usr/bin/env bash
# batched_400_v2.sh — the REBUILT batched-400 comprehensive run.
#
# WHY v2 EXISTS
#   The first batched-400 run (e6fc5ad7) did NOT rebuild both products before it
#   measured: the serve may have loaded a stale binary/.so pair, so its numbers
#   ("verify 37.80ms ~= 37.31 baseline, all mrows gates declined") are not
#   trustworthy evidence about HEAD. This script makes the rebuild the FIRST
#   class citizen: it syncs THIS tree to the node, runs the kernel build
#   (`build.sh 103a`) AND the Rust build (`cargo build --release`), proves the
#   pair is same-source (the binary MUST embed the .so's .build_id), and only
#   then starts ONE serve with the batched-400-final gate matrix.
#
#   口径 (the project rules this script encodes)
#     1. ONE serve at a time, torn down between runs — two serves on the same 8
#        GPUs do not give noisy numbers, they give meaningless ones.
#     2. ONE PROMPT PER SERVE: the `[dspark]` accumulators are process-level and
#        never reset per request, so two requests in one serve would mix averages.
#     3. SAME-SOURCE PAIR, PROVEN: `build.sh` writes `.build_id`; the Rust loader
#        embeds whatever it read, and this script refuses to run unless the
#        binary's embedded id equals the .so's on-disk id. A bare `cargo build`
#        can never heal a stale `.so`, and a bare `build.sh` can never heal a
#        stale binary — BOTH must be rebuilt (see `final-400-config.md` §7).
#
# THE GATE MATRIX (batched-400-final: A+B groups + SWALLOW_STEP)
#   DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1
#   DSV41_EXPERT_ACT_E4M3=1
#   DSV41_BF16_TRUNCATE=1                     # 零拉丁红线 (non-negotiable)
#   DSV41_SH_EXP_MROWS=1 DSV41_MROWS_SMALL_N_ADAPTIVE=1
#   DSV41_GATE_MROWS=1 DSV41_VERIFY_HEAD_MROWS=1
#   DSV41_INDEXER_MROWS=1 DSV41_NORM_MROWS=1
#   DSV41_COMPRESSOR_MROWS=1
#   DSV41_DRAFT_GRAPH=1 DSV41_DRAFT_P3A=1
#   DSV41_VERIFY_GRAPH=1
#   DSV41_SWALLOW_STEP=1                      # without it the step is +6.15ms and
#                                             # the 400 target is out of reach
#   + the two DIAGNOSTIC gates requirement 3 needs (they gate OUTPUT, not
#     performance): DSV41_TIMING=1 (the `[dsv41] step pos=` and `[dspark] steps=`
#     lines do not exist without it) and DSV41_DSPARK_DEBUG=1 (the per-step trace).
#
#   DELIBERATELY NOT SET (each one would silently pick a different path):
#     DSV41_LAZY_VERIFY    — this is the BATCHED path; lazy routes to m=1 rows.
#     DSV41_HC_VERIFY_FUSE — pending the mrows investigation; it also carries
#                            BF16_TRUNCATE into the verify chain (the 050c7fd
#                            baseline-break), so it stays out of this matrix.
#     DSV41_HC_FRONT_ROWS  — the A2 arm has the same open bf16_truncate hole
#                            (chain_dev.rs:11404) and is not cleared to run.
#   The script FAILS if any of those three is present in the launched serve's
#   environment (a shell export would otherwise change the measured path).
#
# WHAT IT PRINTS (requirement 3)
#   * the FULL 1000-token 出师表 answer;
#   * the accept numbers: mean-k, the k_acc HISTOGRAM (derived from the
#     `[dsv41] step pos=` deltas, since `k_emit = k_acc + 1`), tok/step;
#   * verify_ms / draft_ms / commit_ms (the `[dspark] steps=` line);
#   * the step wall: steady-state mean / median / min / p10 over `[dsv41] step`;
#   * the GRAPH state: which `verify_graph_m{m}` shapes captured, the capture /
#     replay / failed counters, and whether the draft graph captured — plus the
#     designed-silent-degradation case (`capture FAILED`) called out by name.
#
# USAGE
#   bash scripts/batched_400_v2.sh                 # sync + full rebuild + run
#   bash scripts/batched_400_v2.sh --dry-run       # pre-flight only; launches nothing
#   bash scripts/batched_400_v2.sh --no-build      # reuse the pair on the node (re-run only)
#   bash scripts/batched_400_v2.sh --no-sync       # measure the node tree as-is
#   MAXTOK=1000 PORT=8691 bash scripts/batched_400_v2.sh
#   NODE=ubuntu@1.2.3.4 bash scripts/batched_400_v2.sh
#
# OUTPUT: $LOGDIR/<tag>.{log,dspark,metrics,txt,env,resp.json,build_*.log}
#   Exit: 0 = usable AND both red lines hold; 1 = a red line broke (拉丁 / 双字 /
#   缺句 / 空答); 2 = the run is NOT usable (no rebuild proof / no measurement).
#
# ⚠️ SYNC (default ON) DOES `rsync --delete` THIS TREE OVER THE NODE'S WORKING
#    TREE — that is the POINT (the node tree in this repo is a deployment mirror
#    and it lags; measuring a lagging tree is exactly the e6fc5ad7 failure). Set
#    SYNC=0 / --no-sync to measure the node tree as-is instead.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

NODE="${NODE:-ubuntu@43.202.208.136}"
ARCH="${ARCH:-103a}"
RROOT="${RROOT:-ferrite}"
LOGDIR="${B400_LOGDIR:-/tmp/batched_400_v2}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
CURL="curl -s --noproxy '*'"

PORT="${PORT:-8691}"
MAXTOK="${MAXTOK:-1000}"                          # 出师表, full answer, no truncation
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"                # x5s = 5 min
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"
STEADY_SKIP="${STEADY_SKIP:-20}"                  # warm-up rounds dropped before the steady stats
REQ_TIMEOUT="${REQ_TIMEOUT:-1800}"                # 1000 tok at ~20-50ms/step

SYNC="${SYNC:-1}"                                 # default ON: the tree must be the latest
BUILD=1
DRY=0

# 出师表 is the project's long-prompt yardstick (dspark_verify.rs PROMPTS): a long
# prompt + recitation, so it is sensitive to cumulative corruption AND runs long
# enough (>= 50 dspark steps) to print the `[dspark] steps=` line.
PROMPT="${PROMPT:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"

while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --no-sync)  SYNC=0 ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,90p' "$0"; exit 0 ;;
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
    echo "FATAL: another batched_400_v2.sh holds $LOCK — runs must never overlap."
    exit 2
fi

# The gate matrix, ONE line, printed verbatim so a reader never reconstructs it.
GATES="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
DSV41_EXPERT_ACT_E4M3=1 \
DSV41_BF16_TRUNCATE=1 \
DSV41_SH_EXP_MROWS=1 DSV41_MROWS_SMALL_N_ADAPTIVE=1 \
DSV41_GATE_MROWS=1 DSV41_VERIFY_HEAD_MROWS=1 \
DSV41_INDEXER_MROWS=1 DSV41_NORM_MROWS=1 \
DSV41_COMPRESSOR_MROWS=1 \
DSV41_DRAFT_GRAPH=1 DSV41_DRAFT_P3A=1 \
DSV41_VERIFY_GRAPH=1 \
DSV41_SWALLOW_STEP=1 \
DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1"
# folded to one line for `env`-style use
GATES_ONELINE="$(echo "$GATES" | tr '\n' ' ' | tr -s ' ')"
# These would each silently select a DIFFERENT path than this matrix intends.
FORBIDDEN="DSV41_LAZY_VERIFY DSV41_HC_VERIFY_FUSE DSV41_HC_FRONT_ROWS"

# ---------------------------------------------------------------------------
# OPT-IN ARM: the tcgen05 e4m3 GROUPED routed gate/up (default OFF).
#
# WHY AN OPT-IN ARM AND NOT IN THE MATRIX ABOVE. The arm changes
# `DSV41_EXPERT_ILV` (a LOAD-TIME weight-layout decision) from the production
# default ON to OFF, so it does NOT measure the shipped layout — folding it into
# `GATES` would silently turn the documented batched-400 matrix into a different
# experiment (exactly the class of mistake `FORBIDDEN` guards against).
#
# WHAT IT RUNS. `moe_rows` (the verify path) dispatches the group-indexed masked
# M=128 e4m3 tile (`dsv41_expert_gemm_e4m3_grouped`, `tc5::e4x`) at
# `moe_experts_grouped_gate_up` when ALL of these hold:
#   * `DSV41_EXPERT_ACT_E4M3=1`   — already in the matrix; the kernel eats e4m3
#                                   activation bytes (`kind::f8f6f4`);
#   * `DSV41_EXPERT_TCGEN05_E4M3=1` — the e4m3 tcgen05 family's runtime gate (the
#                                   MXF4 gate is NOT involved: `kind::mxf4` is
#                                   e2m1 x e2m1, a different kernel);
#   * `DSV41_EXPERT_GROUPED=1`    — builds the permuted layout the kernel indexes;
#   * `DSV41_GATEUP_FUSE=0`       — the e4x epilogue only clamps, it never fuses
#                                   swiglu, so the fused shape declines the arm;
#   * `DSV41_EXPERT_ILV=0`        — the plain (non-interleaved) w1/w3 planes; the
#                                   interleaved layout is unreadable by this arm.
# The `.so` must carry the symbol, which `build.sh 103a` compiles in BY DEFAULT
# (`DSV41_TCGEN05_GATEUP_E4M3_SKELETON`; opt out with `DSV41_BUILD_TCGEN05_E4M3=0`).
#
# USAGE:  B400_TCGEN05_E4M3_GROUPED=1 bash scripts/batched_400_v2.sh
# ONE-LINE GATE CHAIN (the arm's additions, on top of the matrix above):
#   DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0
TCGEN05_E4M3_GROUPED="${B400_TCGEN05_E4M3_GROUPED:-0}"
if [ "$TCGEN05_E4M3_GROUPED" = 1 ]; then
    TC5_GATES="DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0"
    GATES_ONELINE="$GATES_ONELINE $TC5_GATES"
fi

echo "== BATCHED-400 v2 (rebuilt) comprehensive run =="
echo "-- node $NODE   arch $ARCH   port $PORT   tp $TP"
echo "-- prompt: 出师表 max_tokens=$MAXTOK   steady-skip=$STEADY_SKIP"
echo "-- gates: $GATES_ONELINE"
echo "-- NOT set (would change the path): $FORBIDDEN"

# ---------------------------------------------------------------------------
# 0. Node reachability + revisions (SYNC makes them equal; --no-sync only warns).
# ---------------------------------------------------------------------------
REV_LOCAL="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && REV_LOCAL="$REV_LOCAL-dirty"
rssh "true" >/dev/null 2>&1 || { echo "FATAL: cannot reach $NODE"; exit 2; }

if [ "$SYNC" = 1 ] && [ "$DRY" != 1 ]; then
    echo "-- sync: rsync --delete $ROOT -> $NODE:$RROOT (source + .git; target/ and *.so excluded)"
    rssh "mkdir -p ~/$RROOT" || { echo "FATAL: cannot create ~/$RROOT on $NODE"; exit 2; }
    # .git IS synced (unlike the other A/B scripts): the recorded revision in
    # .build_id must be EXACT, and a stale node .git would make the id's
    # revision half misleading even though the cu-hash half is accurate.
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
        --exclude 'target/' --exclude '*.so' --exclude '.build_id' \
        "$ROOT/" "$NODE:$RROOT/" || { echo "FATAL: rsync failed"; exit 2; }
elif [ "$DRY" = 1 ] && [ "$SYNC" = 1 ]; then
    echo "-- sync: skipped (--dry-run is read-only; a real run would rsync $ROOT -> $NODE:$RROOT)"
fi
REV_NODE="$(rssh "git -C ~/$RROOT rev-parse --short HEAD 2>/dev/null || echo '?'" 2>/dev/null)"
[ -n "$(rssh "git -C ~/$RROOT status --porcelain 2>/dev/null" 2>/dev/null)" ] && REV_NODE="$REV_NODE-dirty"
echo "-- rev: local $REV_LOCAL   node $REV_NODE"
if [ "$REV_LOCAL" != "$REV_NODE" ]; then
    echo "   WARN: the node tree is a DIFFERENT revision and is what will be measured."
    echo "         (a run without sync CANNOT claim the latest code was compiled —"
    echo "          that is the e6fc5ad7 failure this script exists to prevent.)"
fi

# ---------------------------------------------------------------------------
# 1. Full rebuild, in the ONLY order that works, then PROVE the pair matches.
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
    # The `touch` is load-bearing: a bare incremental `cargo build` does NOT pick
    # up a rebuilt .so, because the .build_id stamp is baked in by build.rs and
    # cargo has no reason to re-run it (final-400-config.md §7: "单独 cargo build
    # 永远修不好（增量 + build.rs 戳记）").
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
    echo "FATAL: the node is missing ~/$RROOT/{$SO_REL,$BIN_REL} (so=$HAS_SO bin=$HAS_BIN) — run without --no-build"
    exit 2
fi
PAIR_OK=1
if ! embeds_id; then
    PAIR_OK=0
    if [ "$DRY" != 1 ]; then
        echo "FATAL: the node's binary does NOT embed the .so's build id — the pair is not same-source."
        echo "       (.so id on the node: ${NODE_BUILD_ID:-<missing>})"
        exit 2
    fi
fi
if [ "$PAIR_OK" = 1 ]; then
    echo "-- pair: same-source OK  build id: $NODE_BUILD_ID"
    rssh "stat -c '   %n  %s bytes  mtime=%y' ~/$RROOT/$SO_REL ~/$RROOT/$BIN_REL" 2>/dev/null | sed 's/^/ /'
fi

# ---------------------------------------------------------------------------
# 2. Dry run stops here: nothing was built and no serve ran.
# ---------------------------------------------------------------------------
if [ "$DRY" = 1 ]; then
    free="$(rssh "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END {print s+0}'")"
    running="$(rssh 'pgrep -x ferrite-serve | wc -l')"
    echo "-- dry run: pair same-source $([ "$PAIR_OK" = 1 ] && echo YES || echo 'NO (a real run would rebuild)')"
    echo "-- dry run: GPU memory in use = ${free} MiB   ferrite-serve running = $running (both must be 0 for a clean run)"
    echo "-- dry run: a real run would rebuild, then start ONE serve on port $PORT; nothing was launched."
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. Teardown by EXACT process name: `pkill -f ferrite-serve` would match this
#    very command line (it contains the string) and kill the ssh session.
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
    [ "$left" != 0 ] && { echo "FATAL: $left ferrite-serve process(es) survived teardown — runs must never overlap."; exit 2; }
    return 0
}

# ---------------------------------------------------------------------------
# 4. The parser: read (dspark, resp.json, log) and write a `<tag>.metrics`
#    key=value file + `<tag>.txt` (the FULL answer, no truncation). Everything
#    numeric the report needs is derived here, once.
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
#      a SERVE-PROCESS average (its accumulators are declared outside the request
#      loop), so the step-wall below is the per-step truth and this corroborates it.
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

# ---- the GRAPH state -------------------------------------------------------
#  Three states look identical from the outside and only one is a measurement:
#    1. gate closed     -> no `[verify_graph]` line at all;
#    2. capture failed  -> `capture FAILED (…)` (a DESIGNED silent degradation);
#    3. engaged         -> `captured verify_graph_m{m}`, then every later replay.
#  The batched path runs m=5 on the FIRST round (legacy bootstrap) and m=6 on
#  every later one once SWALLOW_STEP is armed, so BOTH names can legitimately
#  appear — the pool holds VERIFY_GRAPH_SLOTS=3 shape slots.
if "[verify_graph] captured" in log:
    vg_engaged = "captured"
elif "[verify_graph] capture FAILED" in log:
    vg_engaged = "failed"
else:
    vg_engaged = "no"
vg_shapes = ",".join(
    "m=%d" % x for x in sorted({int(x) for x in
                                re.findall(r"\[verify_graph\] captured verify_graph_m(\d+)", log)}))
vg_failed_shapes = ",".join(
    "m=%d" % x for x in sorted({int(x) for x in
                                re.findall(r"\[verify_graph\] \S+ capture FAILED \(m=(\d+)", log)}))
prev = re.search(
    r"\[verify_graph\] previous request: captures=(\d+) replays=(\d+) failed=(\w+) shapes=\[([^\]]*)\]",
    log,
)
if prev:
    vg_captures, vg_replays, vg_failed, vg_prev_shapes = prev.groups()
else:
    vg_captures, vg_replays, vg_failed, vg_prev_shapes = "", "", "", ""

if "[draft_graph] captured the draft chain" in log:
    dg_engaged = "captured"
elif "[draft_graph] capture FAILED" in log:
    dg_engaged = "failed"
else:
    dg_engaged = "no"
dg_pos = ""
m = re.search(r"\[draft_graph\] captured the draft chain at pos=(\d+)", log)
if m:
    dg_pos = m.group(1)

# ---- text: the FULL answer + the project's red lines -----------------------
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
        ("vg_engaged", vg_engaged),
        ("vg_shapes", vg_shapes),
        ("vg_failed_shapes", vg_failed_shapes),
        ("vg_captures", vg_captures),
        ("vg_replays", vg_replays),
        ("vg_failed", vg_failed),
        ("vg_prev_shapes", vg_prev_shapes),
        ("dg_engaged", dg_engaged),
        ("dg_pos", dg_pos),
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
    for k in range(0, 7):
        fh.write("hist%d=%d\n" % (k, hist[k]))
PY
}

# ---------------------------------------------------------------------------
# 5. ONE serve, strictly serial.
# ---------------------------------------------------------------------------
run_case() {  # tag port
    local tag="$1" port="$2"
    local rlog="~/b400_${tag}.log"

    echo
    echo "---- case $tag (port $port): the batched-400-final matrix"

    kill_serves >/dev/null                    # every case starts from a clean box
    rssh "cd ~/$RROOT && nohup env CUDA_VISIBLE_DEVICES=$GPU_LIST \
          LD_LIBRARY_PATH=\$HOME/$RROOT/kernels/cuda \
          DSV41_KERNELS=\$HOME/$RROOT/kernels/cuda/libferrite_kernels.so \
          $GATES_ONELINE ./target/release/ferrite-serve --model dsv41 --serve --tp $TP \
          --model-dir $MODEL_DIR --port $port > $rlog 2>&1 &" \
        || { echo "FATAL: could not spawn the serve for $tag"; exit 2; }

    if ! rssh "for i in \$(seq 1 $HEALTH_TRIES); do $CURL -m 2 http://localhost:$port/health >/dev/null 2>&1 && exit 0; sleep 5; done; exit 1"; then
        echo "FATAL: $tag never became healthy (log $rlog); last lines:"
        rssh "tail -15 $rlog" | sed 's/^/    | /'
        teardown "$port"; exit 2
    fi
    # The env actually running, straight off /proc — the cheapest proof that the
    # matrix survived the shell (and that the FORBIDDEN three did NOT creep in).
    rssh "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort" \
        >"$LOGDIR/${tag}.env" 2>/dev/null
    echo "   effective env:"; sed 's/^/     /' "$LOGDIR/${tag}.env"
    local bad=0
    for f in $FORBIDDEN; do
        if grep -q "^${f}=" "$LOGDIR/${tag}.env"; then
            echo "   FATAL: $f is set in the running serve — it selects a DIFFERENT path than this matrix."
            bad=1
        fi
    done
    [ "$bad" = 1 ] && { teardown "$port"; exit 2; }

    local body t0 t1
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
            "$MODEL_NAME" "$PROMPT" "$MAXTOK")"
    t0=$(date +%s)
    rssh "cd ~/$RROOT && $CURL -m $REQ_TIMEOUT http://localhost:$port/v1/chat/completions \
          -H 'Content-Type: application/json' -d '$body'" >"$LOGDIR/${tag}.resp.json" 2>/dev/null
    t1=$(date +%s)
    rssh "grep 'dspark] steps' $rlog | tail -1" >"$LOGDIR/${tag}.dspark" 2>/dev/null
    rssh "cat $rlog" >"$LOGDIR/${tag}.log" 2>/dev/null

    teardown "$port"
    echo "   e2e $(( t1 - t0 ))s   dspark: $(head -c 160 "$LOGDIR/${tag}.dspark")"

    metrics_of "$tag"
    [ -s "$LOGDIR/${tag}.metrics" ] \
        || { echo "FATAL: parser wrote no metrics for $tag (see $LOGDIR/${tag}.log)"; exit 2; }
}

echo
echo "== run (one serve, one prompt) =="
run_case run "$PORT"

# ---------------------------------------------------------------------------
# 6. The report + the 判据. Requirement 3 asks for FOUR things: the full text,
#    k_acc, verify_ms, the step wall — and the graph state.
# ---------------------------------------------------------------------------
echo
python3 - "$LOGDIR" "$MAXTOK" "$STEADY_SKIP" <<'PY'
import os
import sys

LOGDIR = sys.argv[1]
maxtok = sys.argv[2]
warm = sys.argv[3]

d = {}
try:
    for line in open(os.path.join(LOGDIR, "run.metrics")):
        line = line.rstrip("\n")
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
except OSError:
    pass


def g(k, default="NA"):
    return d.get(k, default) or default


def fnum(k):
    try:
        return float(d[k])
    except (KeyError, ValueError):
        return None


print("================ batched-400 v2 report ================")
print("--- accept ---")
print("  steps          %s   (the serve's own counter)" % g("steps"))
print("  mean-k         %s   (serve average)" % g("mean_k"))
print("  kacc_mean      %s   n=%s  (derived from [dsv41] step pos deltas)"
      % (g("kacc_mean"), g("kacc_n")))
print("  k_acc hist 0..6  %s" % " ".join("%d:%s" % (k, g("hist%d" % k, "0")) for k in range(7)))
print("  tok/step       %s   (4 tok/step @ step<=10ms is the 400 path)" % g("tok_step"))
print("--- phase times (ms, per step) ---")
print("  verify         %s   <-- the 400 lever" % g("verify_ms"))
print("  draft          %s" % g("draft_ms"))
print("  commit         %s" % g("commit_ms"))
print("--- step wall (ms, [dsv41] step pos=; first %s warm-up rounds dropped) ---" % warm)
print("  rounds=%s  steady n=%s  mean=%s  median=%s  min=%s  p10=%s"
      % (g("rounds"), g("steady_n"), g("steady_mean"), g("steady_median"),
         g("steady_min"), g("steady_p10")))
print("--- GRAPH state ---")
print("  verify_graph   engaged=%s  captured shapes=[%s]  failed shapes=[%s]"
      % (g("vg_engaged"), g("vg_shapes"), g("vg_failed_shapes", "none")))
print("                 counters: captures=%s replays=%s failed=%s shapes=[%s]"
      % (g("vg_captures"), g("vg_replays"), g("vg_failed"), g("vg_prev_shapes")))
print("  draft_graph    engaged=%s  pos=%s" % (g("dg_engaged"), g("dg_pos")))
print("--- text red lines ---")
print("  chars=%s  md5=%s  先帝创业未半=%s  double-char=%s  latin=%s"
      % (g("chars"), g("md5"), g("has_kaishen"), g("dbl"), g("latin")))
if g("latin") not in ("NA", "0"):
    print("  latin samples: %s" % g("latin_samples"))
if g("resp_err") not in ("NA", ""):
    print("  resp decode error: %s" % g("resp_err"))
print("  max_tokens=%s (requested 1000-tok 出师表 full text)" % maxtok)

print()
print("---- FULL 出师表 text ----")
txt = ""
try:
    txt = open(os.path.join(LOGDIR, "run.txt"), "r", errors="ignore").read()
except OSError:
    pass
print(txt)
print("---- end full text (%d chars) ----" % len(txt))

print()
print("判据:")
rc = 0
steps = fnum("steps")
rounds = fnum("rounds")
if steps is None or rounds is None or rounds == 0:
    print("  USABLE  FAIL: no `[dsv41] step pos=` lines and/or no `[dspark] steps=` line —"
          " nothing was measured (check DSV41_TIMING=1 and the serve log).")
    rc = 2
else:
    print("  USABLE  OK: %d dspark steps over %d per-step samples." % (int(steps), int(rounds)))
    if fnum("steady_mean") is None:
        print("  USABLE  FAIL: the steady wall is empty (every sample was warm-up?).")
        rc = 2

if rc == 0:
    fails = []
    if g("has_kaishen") != "yes":
        fails.append("missing 先帝创业未半")
    if fnum("dbl") is None or fnum("dbl") != 0:
        fails.append("%s adjacent double-char" % g("dbl"))
    if fnum("latin") is None or fnum("latin") != 0:
        fails.append("%s latin char(s)" % g("latin"))
    if fnum("chars") in (None, 0):
        fails.append("empty answer")
    if fails:
        print("  REDLINE FAIL: %s" % "; ".join(fails))
        rc = 1
    else:
        print("  REDLINE OK: zero latin, 0 double-char, 先帝创业未半 present, non-empty.")

# The graph is an optimisation; a refusal is a DESIGNED silent degradation. Report
# it as a note, not a failure — but never let "capture FAILED" pass unremarked.
if g("vg_engaged") == "no":
    print("  GRAPH   NOTE: no [verify_graph] line — DSV41_VERIFY_GRAPH did NOT engage"
          " (gate closed or never reached).")
elif g("vg_engaged") == "failed":
    print("  GRAPH   NOTE: verify graph capture FAILED (shapes [%s]) — the request ran on"
          " the direct launches; the -15ms submit saving was NOT realised." % g("vg_failed_shapes"))
else:
    print("  GRAPH   OK: verify graph captured shape(s) [%s] (captures=%s replays=%s)."
          % (g("vg_shapes"), g("vg_captures"), g("vg_replays")))
if g("dg_engaged") == "no":
    print("  GRAPH   NOTE: no [draft_graph] line — DSV41_DRAFT_GRAPH did NOT engage"
          " (the draft stage is gated on pos>=win, so a short answer may never arm it).")
elif g("dg_engaged") == "failed":
    print("  GRAPH   NOTE: draft graph capture FAILED — the draft ran on the direct launches.")
else:
    print("  GRAPH   OK: draft graph captured at pos=%s." % g("dg_pos"))

if rc == 2:
    print("verdict: rc=2 — NOT usable (no measurement).")
elif rc == 1:
    print("verdict: rc=1 — usable, but a red line broke (see REDLINE above).")
else:
    print("verdict: rc=0 — rebuilt-HEAD batched-400 run: usable, red lines hold.")
sys.exit(rc)
PY
verdict=$?

echo
echo "Logs: $LOGDIR/run.{log,dspark,metrics,txt,env,resp.json}   build: $LOGDIR/build_{so,bin}.log"
exit $verdict
