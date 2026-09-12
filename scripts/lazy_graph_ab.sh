#!/usr/bin/env bash
# lazy_graph_ab.sh — the LAZY verify's m=1 CUDA-graph A/B.
#
#   arm 0: the lazy config WITHOUT DSV41_VERIFY_GRAPH   (the m=1 BARE chain)
#   arm 1: the SAME config WITH DSV41_VERIFY_GRAPH=1    (the m=1 GRAPH slot)
#
# WHY THIS SCRIPT EXISTS
#   commit 23885a7 lifted `VERIFY_GRAPH_SLOTS` 2 -> 3 so that lazy's per-row
#   `step_rows(m = 1)` can own a graph slot of its own. Before it, the two slots
#   were claimed by the batched shapes (m = 5 / m = 6) and every m = 1 row fell
#   through to the BARE chain. A bare-chain row pays the submit of each of the
#   ~thousand streaming launches; the graphed row pays one `graph_launch`. That
#   is measured at ~9.5 ms/row bare vs ~6.15 ms/row graphed — a 3.35 ms/row gap.
#   A lazy round runs `k_emit = k_acc + 1` rows, so at the observed ~2.4
#   rows/step the gap is ~8-9 ms/step: lazy's step wall should fall from
#   ~24 ms to ~15 ms.
#
#   The switch is opt-in (default OFF) and an ARMED graph can still refuse to
#   capture (a driver refusal is a DESIGNED silent degradation), so three states
#   look identical from the outside and only one of them is a measurement:
#     1. gate closed     → no `[verify_graph]` line at all (that is arm 0's job);
#     2. capture failed  → `[verify_graph] … capture FAILED (…)` and the request
#                          finishes on the direct launches anyway;
#     3. engaged         → `[verify_graph] captured verify_graph_m1 at pos=…`,
#                          then every later m = 1 verify replays.
#   Only (3) makes arm 1 evidence about the m = 1 graph.
#
# 判据 (printed as a verdict; all three legs must hold)
#   * ENGAGE — arm 1 MUST print `[verify_graph] captured verify_graph_m1` AND
#              arm 0 MUST print no `[verify_graph]` line at all.
#   * SPEED  — the STEADY-STATE step wall (`[dsv41] step pos=…: Xms`, first
#              STEADY_SKIP rounds dropped) must drop by >= MIN_GAIN_MS (default
#              6 ms; the plan's expectation is ~9 ms, printed for the reader).
#              The `[dspark] steps=` `verify=` field is reported as
#              corroboration only: its accumulators live OUTSIDE the request
#              loop, so it is a SERVE-PROCESS average, not the per-step wall.
#   * TEXT   — the graph is an optimisation: it may not move a single token.
#              The FULL answer is saved per arm and compared (char count + md5),
#              and the project's two red lines are checked (no adjacent double
#              character; 出师表 must contain 先帝创业未半).
#
# 口径 (the project rules this script encodes)
#   1. SERIAL, ONE SERVE AT A TIME, torn down between arms — two serves on the
#      same 8 GPUs do not give noisy numbers, they give meaningless ones.
#   2. ONE PROMPT PER SERVE: the `[dspark]` accumulators are process-level and
#      never reset per request, so two requests in one serve would mix averages.
#   3. SAME BINARY + SAME .so ON EVERY ARM, PROVEN (the loader refuses a
#      mismatched pair; `cargo build` alone can never heal a stale `.so`).
#   4. `DSV41_AR_V5` is deliberately NOT set here: under TP8 both `ar_v5()` legs
#      default ON, so the graph is reachable either way. If it is ever forced
#      OFF, the ARMED graph never engages and the ENGAGE leg fails loudly.
#
# k_acc 直方图
#   `DSV41_LAZY_VERIFY` has no dedicated histogram print, but the per-round
#   `[dsv41] step pos=…` lines carry it for free: the engine advances the
#   position by `k_emit = k_acc + 1` per committed round, so the DELTA between
#   consecutive positions MINUS ONE is that round's k_acc. The parser buckets
#   those (0..6) and reports the histogram-derived mean-k beside the serve's own
#   `mean-k` as a cross-check — no new env knob needed.
#
# USAGE
#   bash scripts/lazy_graph_ab.sh              # build + 2 serial serves
#   bash scripts/lazy_graph_ab.sh --no-build   # reuse the pair on the node
#   bash scripts/lazy_graph_ab.sh --dry-run    # pre-flight only; launches nothing
#   SYNC=1 bash scripts/lazy_graph_ab.sh       # rsync THIS tree to the node first
#   MAXTOK=1000 PORT_BASE=8210 STEADY_SKIP=20 MIN_GAIN_MS=6 bash scripts/lazy_graph_ab.sh
#
# OUTPUT: per arm — steps / mean-k (+ the histogram mean-k) / tok-step / verify_ms
#   / draft_ms / the steady step wall (mean, median, min, p10) / the k_acc
#   histogram / the engagement proof / the full answer text; then the Δ table and
#   the verdict. Logs: $LOGDIR/<tag>.{log,dspark,metrics,txt,resp.json}
#   Exit: 0 = all three legs hold; 1 = SPEED or TEXT leg failed; 2 = the run is
#   not usable (unmeasured / engagement proof missing / harness error).
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
LOGDIR="${LGA_LOGDIR:-/tmp/lazy_graph_ab}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
CURL="curl -s --noproxy '*'"

PORT_BASE="${PORT_BASE:-8210}"
MAXTOK="${MAXTOK:-1000}"                        # 出师表, full answer, no truncation
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"              # x5s
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"
STEADY_SKIP="${STEADY_SKIP:-20}"                # warm-up rounds dropped before the steady stats
MIN_GAIN_MS="${MIN_GAIN_MS:-6}"                 # the acceptance gate
EXPECT_GAIN_MS="${EXPECT_GAIN_MS:-9}"           # the plan's expectation (~24 -> ~15 ms)

# 出师表 is the project's long-prompt yardstick (dspark_verify.rs PROMPTS): a
# long prompt + recitation, so it is sensitive to cumulative corruption AND runs
# long enough (>= 50 dspark steps) to print the `[dspark] steps=` line.
PROMPT="${PROMPT:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"

BUILD=1; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,95p' "$0"; exit 0 ;;
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
    echo "FATAL: another lazy_graph_ab.sh holds $LOCK — arms must never overlap."
    exit 2
fi

echo "== LAZY verify m=1 graph A/B (no DSV41_VERIFY_GRAPH vs DSV41_VERIFY_GRAPH=1) =="
echo "-- node $NODE   arch $ARCH   ports $PORT_BASE..$(( PORT_BASE + 1 ))   tp $TP"
echo "-- prompt: 出师表 max_tokens=$MAXTOK   steady-skip=$STEADY_SKIP   gate >= ${MIN_GAIN_MS}ms"

# The two arms differ by EXACTLY one variable; print it so a reader never has to
# reconstruct the difference from the code.
ARM_ENV="DSV41_LAZY_VERIFY=1 DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_SH_EXP_MROWS=1 DSV41_DRAFT_P3A=1 DSV41_TIMING=1"
echo "-- arm0 env: $ARM_ENV"
echo "-- arm1 env: $ARM_ENV DSV41_VERIFY_GRAPH=1"

# ---------------------------------------------------------------------------
# 0. Node reachability + revision on each side (a mismatch is a WARNING: with
#    SYNC unset the node tree is the caller's and measuring it is legitimate —
#    as long as the reader KNOWS the two differ).
# ---------------------------------------------------------------------------
REV_LOCAL="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && REV_LOCAL="$REV_LOCAL-dirty"
if [ -n "${SYNC:-}" ]; then
    echo "-- SYNC=1: rsync --delete $ROOT -> $NODE:$RROOT (the repo's own files only)"
    rssh "mkdir -p ~/$RROOT" || { echo "FATAL: cannot reach $NODE"; exit 2; }
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
        --exclude target/ --exclude .git/ --exclude '*.so' --exclude .build_id \
        "$ROOT/" "$NODE:$RROOT/" || { echo "FATAL: rsync failed"; exit 2; }
fi
REV_NODE="$(rssh "git -C ~/$RROOT rev-parse --short HEAD 2>/dev/null || echo '?'" 2>/dev/null)"
[ -n "$(rssh "git -C ~/$RROOT status --porcelain 2>/dev/null" 2>/dev/null)" ] && REV_NODE="$REV_NODE-dirty"
echo "-- rev: local $REV_LOCAL   node $REV_NODE"
if [ "$REV_LOCAL" != "$REV_NODE" ] && [ -z "${SYNC:-}" ]; then
    echo "   WARN: the node tree is a DIFFERENT revision and is what will be measured."
    echo "         (SYNC=1 rsyncs this tree over it; the caller owns that tree.)"
fi

# ---------------------------------------------------------------------------
# 1. Build BOTH products in the order that works, then PROVE they match.
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
[ "$PAIR_OK" = 1 ] && echo "-- pair: same-source OK (build id $NODE_BUILD_ID)"

# ---------------------------------------------------------------------------
# 2. Dry run stops here: nothing was built and no serve ran.
# ---------------------------------------------------------------------------
if [ "$DRY" = 1 ]; then
    free="$(rssh "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END {print s+0}'")"
    echo "-- dry run: pair same-source $([ "$PAIR_OK" = 1 ] && echo YES || echo 'NO (a real run would rebuild)')"
    echo "-- dry run: GPU memory in use = ${free} MiB (a serve needs the box to itself)"
    echo "-- dry run: a real run would start 2 serial serves on ports $PORT_BASE..$(( PORT_BASE + 1 )); nothing was launched."
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
    [ "$left" != 0 ] && { echo "FATAL: $left ferrite-serve process(es) survived teardown — arms must never overlap."; exit 2; }
    return 0
}

# ---------------------------------------------------------------------------
# 4. The per-arm parser: read (dspark, resp.json, log) and write a
#    `<tag>.metrics` key=value file + `<tag>.txt` (the FULL answer, no
#    truncation). Everything numeric the report needs is derived here, once.
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
#      loop), so it corroborates the per-step wall, it does not replace it.
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

# ---- the engagement proof ---------------------------------------------------
if "[verify_graph] captured" in log:
    engaged = "captured"
elif "[verify_graph] capture FAILED" in log:
    engaged = "failed"
else:
    engaged = "no"
captured_m1 = "yes" if "[verify_graph] captured verify_graph_m1" in log else "no"
# Which shapes actually captured. If lazy's route stayed on the batched arm the
# log shows m=5/m=6 instead of m=1 — the reader sees that here, not just "failed".
found = sorted({int(x) for x in re.findall(r"\[verify_graph\] captured verify_graph_m(\d+)", log)})
captured_shapes = ",".join("m=%d" % x for x in found) if found else ""
prev = re.search(
    r"\[verify_graph\] previous request: captures=(\d+) replays=(\d+) failed=(\w+) shapes=\[([^\]]*)\]",
    log,
)
if prev:
    captures, replays, failed, shapes = prev.group(1), prev.group(2), prev.group(3), prev.group(4)
else:
    captures, replays, failed, shapes = "", "", "", ""

# ---- text: the FULL answer + the project's two red lines --------------------
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
md5 = hashlib.md5(content.encode()).hexdigest() if content else "NA"
has_kaishen = "yes" if "先帝创业未半" in content else "no"

fails = []
if not content:
    fails.append("empty answer")
if has_kaishen != "yes":
    fails.append("missing 先帝创业未半")
if dbl:
    fails.append("%d adjacent double-char" % dbl)
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
        ("engaged", engaged),
        ("captured_m1", captured_m1),
        ("captured_shapes", captured_shapes),
        ("captures", captures), ("replays", replays),
        ("failed", failed), ("shapes", shapes),
        ("chars", str(len(content))),
        ("dbl", str(dbl)),
        ("md5", md5),
        ("has_kaishen", has_kaishen),
        ("text_ok", text_ok),
        ("resp_err", resp_err),
    ]:
        fh.write("%s=%s\n" % (k, v))
PY
}

# ---------------------------------------------------------------------------
# 5. One serve per arm, strictly serial.
# ---------------------------------------------------------------------------
run_case() {  # arm tag port
    local arm="$1" tag="$2" port="$3"
    local rlog="~/lga_${tag}.log"
    local genv="$ARM_ENV"
    [ "$arm" = 1 ] && genv="$genv DSV41_VERIFY_GRAPH=1"

    echo
    echo "---- case $tag (port $port): arm=$arm  $( [ "$arm" = 1 ] && echo 'DSV41_VERIFY_GRAPH=1' || echo 'no DSV41_VERIFY_GRAPH' )"

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
    echo "   $(grep -E '^(steps|mean_k|kacc_mean|verify_ms|steady_mean|steady_median|rounds|engaged|captured_m1|chars|dbl|text_ok)=' "$LOGDIR/${tag}.metrics" | paste -sd' ' -)"
    echo "   k_acc hist [0..6]: $(grep -E '^hist[0-6]=' "$LOGDIR/${tag}.metrics" | cut -d= -f2 | paste -sd' ' -)"

    echo "   ---- full text ($tag), $(wc -c <"$LOGDIR/${tag}.txt") bytes ----"
    cat "$LOGDIR/${tag}.txt"
    echo
    echo "   ---- end full text ($tag) ----"
}

echo
echo "== cases (serial, same binary+.so, one serve per arm) =="
run_case 0 a0 "$PORT_BASE"
run_case 1 a1 "$(( PORT_BASE + 1 ))"

# ---------------------------------------------------------------------------
# 6. The table + the 判据. Δ is arm1 minus arm0.
# ---------------------------------------------------------------------------
echo
python3 - "$LOGDIR" "$MIN_GAIN_MS" "$EXPECT_GAIN_MS" <<'PY' | tee "$LOGDIR/table.txt"
import os
import sys

LOGDIR, min_gain, expect = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
CASES = [("a0", "0"), ("a1", "1")]


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


M = {tag: load(tag) for tag, _ in CASES}

print("%-4s %-3s %7s %8s %8s %7s %9s %12s %9s %9s %-9s %-11s %6s %5s %s" % (
    "TAG", "ARM", "steps", "mean-k", "hist-k", "tok/st", "verify_ms",
    "steady_mean", "steady_med", "steady_min", "ENGAGED", "captured_m1", "chars", "dbl", "TEXT"))
for tag, arm in CASES:
    d = M[tag]
    def g(k):
        return d.get(k, "NA")
    print("%-4s %-3s %7s %8s %8s %7s %9s %12s %9s %9s %-9s %-11s %6s %5s %s" % (
        tag, arm, g("steps"), g("mean_k"), g("kacc_mean"), g("tok_step"), g("verify_ms"),
        g("steady_mean"), g("steady_median"), g("steady_min"),
        g("engaged"), g("captured_m1"), g("chars"), g("dbl"), g("text_ok")))

print()
print("k_acc histogram (0..6) — derived from the [dsv41] step pos= deltas:")
for tag, arm in CASES:
    d = M[tag]
    h = " ".join("%d:%s" % (k, d.get("hist%d" % k, "0")) for k in range(7))
    print("  arm%s  n=%-4s mean-k(hist)=%-6s  %s"
          % (arm, d.get("kacc_n", "NA"), d.get("kacc_mean", "NA"), h))

print()
print("判据 (arm1 minus arm0):")
rc = 0

# --- ENGAGE ----------------------------------------------------------------
engage_ok = True
if M["a1"].get("engaged") != "captured" or M["a1"].get("captured_m1") != "yes":
    print("  ENGAGE  FAIL: arm1 did not print '[verify_graph] captured verify_graph_m1' "
          "(engaged=%s captured_m1=%s) — arm1 is NOT evidence about the m=1 graph"
          % (M["a1"].get("engaged"), M["a1"].get("captured_m1")))
    engage_ok = False
if M["a0"].get("engaged") != "no":
    print("  ENGAGE  FAIL: arm0 printed a [verify_graph] line (engaged=%s) — the env was not honoured"
          % M["a0"].get("engaged"))
    engage_ok = False
if engage_ok:
    print("  ENGAGE  OK: arm1 captured verify_graph_m1 (captured_shapes=[%s] captures=%s replays=%s); arm0 printed none"
          % (M["a1"].get("captured_shapes"), M["a1"].get("captures"), M["a1"].get("replays")))
else:
    # An unengaged arm1 is NOT a measurement — it is the run's usability failure,
    # so it dominates every other leg (rc 2, never 0/1).
    rc = 2
    if M["a1"].get("captured_shapes") == "m=5" or M["a1"].get("captured_shapes") == "m=6":
        print("  ENGAGE  HINT: arm1 only captured the BATCHED shape (%s) — lazy's route stayed on the "
              "batched arm, so the m=1 graph never ran (check DSV41_LAZY_THRESHOLD / mean-k)."
              % M["a1"].get("captured_shapes"))

# --- SPEED -----------------------------------------------------------------
a0, a1 = num(M["a0"], "steady_mean"), num(M["a1"], "steady_mean")
speed_ok = True
if a0 is None or a1 is None:
    print("  SPEED   UNMEASURED: arm0=%s arm1=%s — a serve printed no '[dsv41] step pos=' lines"
          % (a0, a1))
    rc = 2
    speed_ok = False
else:
    gain = a0 - a1
    ok = gain >= min_gain
    print("  SPEED   steady wall %.2f -> %.2f ms (Δ%+.2f, gain %+.2f ms; gate >= %.1f ms %s;"
          " plan expectation ~%.1f ms %s)"
          % (a0, a1, a1 - a0, gain, min_gain, "PASS" if ok else "FAIL",
             expect, "PASS" if gain >= expect else "below"))
    v0, v1 = num(M["a0"], "verify_ms"), num(M["a1"], "verify_ms")
    if v0 is not None and v1 is not None:
        print("  SPEED   (corroboration) serve-avg verify_ms %.2f -> %.2f ms (Δ%+.2f ms)"
              % (v0, v1, v1 - v0))
    if not ok:
        rc = max(rc, 1)
        speed_ok = False

# --- TEXT ------------------------------------------------------------------
text_ok = True
if M["a0"].get("text_ok") != "ok" or M["a1"].get("text_ok") != "ok":
    print("  TEXT    FAIL: arm0='%s' arm1='%s'" % (M["a0"].get("text_ok"), M["a1"].get("text_ok")))
    text_ok = False
elif M["a0"].get("md5") != M["a1"].get("md5"):
    print("  TEXT    FAIL: the arms differ — arm0 %s chars md5=%s | arm1 %s chars md5=%s"
          % (M["a0"].get("chars"), M["a0"].get("md5"), M["a1"].get("chars"), M["a1"].get("md5")))
    text_ok = False
else:
    print("  TEXT    OK: identical answer both arms (%s chars, %s double-chars, md5 %s)"
          % (M["a0"].get("chars"), M["a0"].get("dbl"), M["a0"].get("md5")))
if not text_ok:
    rc = max(rc, 1)

print()
if rc == 2:
    print("verdict: rc=2 — the run is NOT usable (unmeasured / engagement proof missing).")
elif rc == 1:
    print("verdict: rc=1 — engaged and text-consistent, but the SPEED leg did not hold.")
else:
    print("verdict: rc=0 — the m=1 graph engaged, the text is identical, and the steady step wall dropped by >= %.1f ms." % min_gain)
sys.exit(rc)
PY
verdict=$?

echo
if [ "$verdict" = 2 ]; then
    echo "lazy_graph_ab: NOT USABLE — read the ENGAGE/SPEED lines above and the [verify_graph] lines in the logs."
    exit 2
fi
if [ "$verdict" = 1 ]; then
    echo "lazy_graph_ab: the m=1 graph engaged but the speed/text leg failed — compare the table, do not trust a single number."
    exit 1
fi
echo "lazy_graph_ab: m=1 verify graph engaged, text-consistent, steady step wall down >= ${MIN_GAIN_MS}ms."
echo "Logs: $LOGDIR/<tag>.{log,dspark,metrics,txt,env,resp.json}   table: $LOGDIR/table.txt"
exit 0
