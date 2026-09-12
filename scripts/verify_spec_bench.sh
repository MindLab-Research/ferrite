#!/usr/bin/env bash
# verify_spec_bench.sh — the DSpark verify A/B/C: is the CUDA graph (and, when it
# is its own gate, the multi-row MoE) actually faster?
#
# THE 口径 THIS SCRIPT ENCODES (three project rules, each cost a cycle to learn)
#   1. SAME SESSION, BACK TO BACK, SERIAL. Every arm runs in one invocation, one
#      after the other. Two serves at once on the same 8 GPUs do not give "noisy"
#      numbers, they give MEANINGLESS ones — a flock makes a second concurrent
#      run impossible and every arm is fully torn down before the next starts.
#   2. THE SAME BINARY AND .so ON EVERY ARM. The kernel sources' hash is folded
#      into the .so's build id, which is embedded in the binary; the loader
#      REFUSES a mismatched pair. This script builds BOTH products in the only
#      order that works (build.sh 103a -> touch build.rs -> cargo build --release,
#      see dsv41_serve_ab.sh's header) and then PROVES the pair is same-source
#      before spawning anything.
#   3. THE NUMBER IS THE LAST "[dspark] steps=… verify=…" LINE, i.e. the run's
#      cumulative per-step average (printed every 50 steps). Never a segment
#      average, never a hand-timed wall clock.
#
# ARMS
#   C1 baseline        DSV41_VERIFY_GRAPH=0
#   C2 graph           DSV41_VERIFY_GRAPH=1
#   C3 graph+multirow  C2 plus the multi-row gate, AUTO-DETECTED (see MROWS_GATE).
#       If this tree has NO such gate the multi-row kernels are the DEFAULT, so
#       C3 would merely re-measure C2 — the arm then SKIPS with that reason
#       instead of printing a fake tie. FORCE_C3=1 runs it anyway (same env as
#       C2, for the record).
#
# EACH ARM: serve (DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1 + the arm's gate)
#   -> ONE fixed long prompt (出师表) at MAXTOK -> read the last dspark line
#   -> POST /shutdown -> exact-PID teardown. Serial, always.
#
# USAGE
#   bash scripts/verify_spec_bench.sh                # build + all arms, serial
#   bash scripts/verify_spec_bench.sh --dry-run      # pre-flight only: node, pair
#                                                    #   same-source, gate, port.
#                                                    #   Builds/launches NOTHING.
#   bash scripts/verify_spec_bench.sh --no-build     # reuse the pair on the node
#   SYNC=1 bash scripts/verify_spec_bench.sh         # rsync THIS tree to the node
#                                                    #   first (see the warning)
#   MROWS_GATE=DSV41_VERIFY_MROWS bash scripts/verify_spec_bench.sh
#   MAXTOK=300 PORT_BASE=8160 bash scripts/verify_spec_bench.sh
#
# OUTPUT: the comparison table (verify_ms / draft_ms / mean-k / tok·step-1 / steps
#   and Δverify vs C1), the raw last dspark line per arm, and — as a first-class
#   column, not an afterthought — the TEXT check on each arm's answer (the 出师表
#   anchor "先帝创业未半" and the adjacent-double-character rule, lifted from
#   crates/ferrite-models/src/dsv41/dspark_verify.rs and inlined so this script
#   needs no Rust build).
#   Exit: 0 = every arm produced a readable number AND its text passed;
#         1 = a text check failed (correctness, decide the A/B on the table);
#         2 = harness error (build / pair / serve / parse) — the table is then
#             NOT evidence about the kernels.
#
# ⚠️ SYNC=1 DOES rsync --delete THE REPO OVER THE NODE'S WORKING TREE. The node
#    tree is normally the caller's own deployment, so syncing is OFF by default
#    and the script merely WARNS when the two revisions differ. Only set SYNC=1
#    when you intend the measurement to be of THIS revision.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
K="$ROOT/kernels/cuda"

NODE="${NODE:-ubuntu@43.202.208.136}"
ARCH="${ARCH:-103a}"
RROOT="${RROOT:-ferrite}"                # the tree on the node, relative to $HOME
LOGDIR="${SPEC_LOGDIR:-/tmp/verify_spec_bench}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
CURL="curl -s --noproxy '*'"

PORT_BASE="${PORT_BASE:-8160}"
MAXTOK="${MAXTOK:-300}"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
# 出师表 is the project's long-prompt yardstick (dspark_verify.rs PROMPTS); 300
# tokens is long enough that the cumulative average is past the warm-up.
PROMPT="${PROMPT:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"       # x5s
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"

# Candidate multi-row gates, most specific first. DSV41_GATEUP_ROWS is
# deliberately NOT here: it re-packs warps per CTA inside one launch and does not
# change how many activation rows a call covers (see dsv41_experts_mxf4.cu).
GATE_CANDIDATES=(DSV41_VERIFY_MROWS DSV41_VERIFY_ROWS DSV41_SPEC_MROWS DSV41_MULTIROW
                 DSV41_MROWS DSV41_MOE_ROWS DSV41_EXPERT_MROWS DSV41_GEMM_MROWS)

BUILD=1; DRY=0; FORCE_C3="${FORCE_C3:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,62p' "$0"; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)"; exit 2 ;;
    esac
    shift
done

mkdir -p "$LOGDIR"

rssh() { ssh "${SSH_OPTS[@]}" "$NODE" "$1"; }

# One writer at a time, across invocations too: a second run while the first is
# serving is the exact failure this script exists to prevent.
LOCK="$LOGDIR/.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "FATAL: another verify_spec_bench.sh holds $LOCK — arms must never overlap."
    exit 2
fi

echo "== DSV41 spec verify A/B/C =="
echo "-- node $NODE   arch $ARCH   ports $PORT_BASE..   max_tokens $MAXTOK"

# ---------------------------------------------------------------------------
# 0. Node reachability + which revision is on each side. A revision mismatch is
#    a WARNING, not a failure: with SYNC unset the node tree is the caller's and
#    measuring it is legitimate — as long as the reader KNOWS the two differ.
# ---------------------------------------------------------------------------
REV_LOCAL="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
dirty_local=""; [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && dirty_local="-dirty"
if [ -n "${SYNC:-}" ]; then
    echo "-- SYNC=1: rsync --delete $ROOT -> $NODE:$RROOT (the repo's own files only)"
    rssh "mkdir -p ~/$RROOT" || { echo "FATAL: cannot reach $NODE"; exit 2; }
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
        --exclude target/ --exclude .git/ --exclude '*.so' --exclude .build_id \
        "$ROOT/" "$NODE:$RROOT/" || { echo "FATAL: rsync failed"; exit 2; }
fi
REV_NODE="$(rssh "git -C ~/$RROOT rev-parse --short HEAD 2>/dev/null || echo '?'" 2>/dev/null)"
[ -n "$(rssh "git -C ~/$RROOT status --porcelain 2>/dev/null" 2>/dev/null)" ] && REV_NODE="$REV_NODE-dirty"
echo "-- rev: local $REV_LOCAL$dirty_local   node $REV_NODE"
if [ "$REV_LOCAL" != "$REV_NODE" ] && [ -z "${SYNC:-}" ]; then
    echo "   WARN: the node tree is a DIFFERENT revision and is what will be measured."
    echo "         (SYNC=1 rsyncs this tree over it; the caller owns that tree.)"
fi

# ---------------------------------------------------------------------------
# 1. Build BOTH products, in the order that works, then PROVE they match.
#    `cargo build` alone can never heal a stale pair (cargo is incremental and
#    the stamp is baked in by ferrite-kernel/build.rs), which is why the .so
#    build is followed by a forced build.rs touch.
# ---------------------------------------------------------------------------
SO_REL="kernels/cuda/libferrite_kernels.so"
BIN_REL="target/release/ferrite-serve"
if [ "$BUILD" = 1 ] && [ "$DRY" != 1 ]; then
    echo "-- build: kernels/cuda/build.sh $ARCH ..."
    # ⚠️ build.sh's LAST statement is `[ ${#SKELETON_FLAGS[@]} -gt 0 ] && echo …`,
    # so with no skeleton flags exported the script exits 1 under its own `set -e`
    # EVEN THOUGH IT BUILT THE .so (measured 2026-09-12). The exit code is NOT the
    # success criterion here — the "built …" line is.
    rssh "cd ~/$RROOT/kernels/cuda && bash build.sh $ARCH" >"$LOGDIR/build_so.log" 2>&1
    so_rc=$?
    if ! grep -q "built .*libferrite_kernels.so for sm_${ARCH}" "$LOGDIR/build_so.log"; then
        echo "FATAL: build.sh $ARCH did not report a successful build (rc=$so_rc, log $LOGDIR/build_so.log)"
        tail -5 "$LOGDIR/build_so.log" | sed 's/^/    | /'
        exit 2
    fi
    [ "$so_rc" != 0 ] && echo "   (build.sh exited $so_rc but reported a successful build — trailing '[ ] && echo' under set -e; the .so IS fresh)"
    echo "-- build: cargo build --release (touch build.rs first, the stamp is baked in by it) ..."
    # The node's non-interactive PATH has no cargo: go through a login shell.
    rssh "cd ~/$RROOT && touch crates/ferrite-kernel/build.rs && bash -lc 'cd ~/$RROOT && cargo build --release'" \
        >"$LOGDIR/build_bin.log" 2>&1 \
        || { echo "FATAL: cargo build --release failed (log $LOGDIR/build_bin.log)"; tail -5 "$LOGDIR/build_bin.log"; exit 2; }
elif [ "$DRY" = 1 ]; then
    echo "-- build: skipped (--dry-run is read-only; it never rewrites .build_id or the pair)"
fi

# Same-source proof (post-build, never an assumption). The .so and its .build_id
# both live on the NODE — the source box has no nvcc, so it never holds one, and
# comparing against a local file would always "fail" a perfectly good pair.
#
# `grep -cF` and NOT `grep -qF`: under `set -o pipefail` a `strings BIN | grep -q id`
# makes grep exit at its first match, `strings` then takes SIGPIPE (141) and the
# PIPELINE reports failure even though the id WAS found — the gate would misfire
# on a good pair.
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
        echo "       rerun without --no-build so the pair is rebuilt in the one working order."
        exit 2
    fi
fi
[ "$PAIR_OK" = 1 ] && echo "-- pair: same-source OK (build id $NODE_BUILD_ID)"

# ---------------------------------------------------------------------------
# 2. Which gate, if any, turns the multi-row MoE on. No gate => it IS the default
#    and C3 is not a distinct arm.
# ---------------------------------------------------------------------------
GATE="${MROWS_GATE:-}"; GATE_SRC="${MROWS_GATE:+MROWS_GATE env}"
if [ -z "$GATE" ]; then
    NODE_GATES="$(rssh "grep -rhoE --include='*.rs' 'DSV41_[A-Z0-9_]+' ~/$RROOT/crates/ 2>/dev/null | sort -u")"
    for c in "${GATE_CANDIDATES[@]}"; do
        if echo "$NODE_GATES" | grep -qx "$c"; then GATE="$c"; GATE_SRC="auto-detected in the tree's Rust sources"; break; fi
    done
fi
if [ -n "$GATE" ]; then
    echo "-- C3 gate: $GATE ($GATE_SRC)"
else
    echo "-- C3 gate: NONE found. The multi-row kernels are the DEFAULT in this tree,"
    echo "   so C3 would re-measure C2 — the arm is skipped (FORCE_C3=1 to run it anyway)."
fi

# ---------------------------------------------------------------------------
# 3. Dry run stops here: nothing was built and no serve ran. What it answers is
#    "is this box ready to be measured on" — node, revision, pair, gate, GPU.
# ---------------------------------------------------------------------------
if [ "$DRY" = 1 ]; then
    [ "$PAIR_OK" = 1 ] && echo "-- dry run: pair same-source YES" \
                       || echo "-- dry run: pair same-source NO (binary does not embed ${NODE_BUILD_ID:-<?>}) — a real run would rebuild it"
    free="$(rssh "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END {print s+0}'")"
    echo "-- dry run: GPU memory in use = ${free} MiB (a serve needs the box to itself)"
    echo "-- dry run: ports $PORT_BASE..$(( PORT_BASE + 2 )) would be used; nothing was launched."
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. The arms, strictly serial.
# ---------------------------------------------------------------------------
BODY="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
        "$MODEL_NAME" "$PROMPT" "$MAXTOK")"

declare -a A_TAG A_VERIFY A_DRAFT A_MEANK A_TOKSTEP A_STEPS A_TEXT A_NOTE
unmeasured=0    # set when an arm produced no cumulative dspark line at all

# Teardown by EXACT process name: `pkill -f ferrite-serve` would match this very
# command line (it contains the string) and kill the ssh session. Then prove no
# serve survives — a leftover from arm N would corrupt arm N+1's reading.
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

run_arm() {
    local tag="$1" env_extra="$2" note="$3"
    local port=$(( PORT_BASE + ${#A_TAG[@]} ))
    local rlog="~/spec_${tag}.log"
    echo
    echo "---- arm $tag (port $port): ${env_extra:-<no extra gate>}"

    kill_serves >/dev/null                       # arm starts from a clean box
    rssh "cd ~/$RROOT && nohup env CUDA_VISIBLE_DEVICES=$GPU_LIST \
          LD_LIBRARY_PATH=\$HOME/$RROOT/kernels/cuda \
          DSV41_KERNELS=\$HOME/$RROOT/kernels/cuda/libferrite_kernels.so \
          DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1 $env_extra \
          ./target/release/ferrite-serve --model dsv41 --serve --tp $TP \
          --model-dir $MODEL_DIR --port $port > $rlog 2>&1 &" \
        || { echo "FATAL: could not spawn the serve for $tag"; exit 2; }

    if ! rssh "for i in \$(seq 1 $HEALTH_TRIES); do $CURL -m 2 http://localhost:$port/health >/dev/null 2>&1 && exit 0; sleep 5; done; exit 1"; then
        echo "FATAL: $tag never became healthy (log $rlog); last lines:"
        rssh "tail -15 $rlog" | sed 's/^/    | /'
        teardown "$port"; exit 2
    fi

    local t0 t1
    t0=$(date +%s)
    rssh "cd ~/$RROOT && $CURL -m 900 http://localhost:$port/v1/chat/completions \
          -H 'Content-Type: application/json' -d '$BODY'" >"$LOGDIR/${tag}.resp.json" 2>/dev/null
    t1=$(date +%s)
    rssh "grep 'dspark] steps' $rlog | tail -1" >"$LOGDIR/${tag}.dspark" 2>/dev/null
    rssh "cat $rlog" >"$LOGDIR/${tag}.log" 2>/dev/null

    teardown "$port"
    echo "   wall ${t1}s->${t1}s / e2e $(( t1 - t0 ))s   $(head -c 120 "$LOGDIR/${tag}.dspark")"

    # Numbers + text check, one python pass over the two artifacts.
    local parsed
    parsed="$(python3 - "$LOGDIR/${tag}.dspark" "$LOGDIR/${tag}.resp.json" "$LOGDIR/${tag}.log" <<'PY'
import json, re, sys
dspark, resp, log = sys.argv[1], sys.argv[2], sys.argv[3]

def field(line, key):
    i = line.find(key)
    if i < 0: return None
    rest = line[i + len(key):]
    m = re.match(r'[-+0-9.eE]+', rest)
    return float(m.group(0)) if m else None

line = ''
try:
    for l in open(dspark):
        if 'steps=' in l: line = l.rstrip('\n')
except OSError: pass
v = field(line, 'verify='); d = field(line, 'draft=')
k = field(line, 'mean-k='); t = field(line, 'tok/step=')
s = field(line, 'steps=')

# The text rules of crates/ferrite-models/src/dsv41/dspark_verify.rs, inlined so
# this script has no Rust dependency. Whitespace repeats are NOT corruption.
def has_double_char(txt):
    ch = list(txt)
    for i in range(1, len(ch)):
        if ch[i] == ch[i-1] and not ch[i].isspace():
            return (i, '…' + ''.join(ch[max(0, i-6):i+6]) + '…')
    return None

fails, content = [], ''
try:
    doc = json.load(open(resp))
    content = doc['choices'][0]['message']['content']
except Exception as e:
    fails.append('response not parseable: %s' % e)
if content:
    dc = has_double_char(content)
    if dc: fails.append('adjacent double char at char %d: %s' % dc)
    if '先帝创业未半' not in content:
        fails.append('missing "先帝创业未半" — got: %r' % content[:160])
    if len(content) < 40:
        fails.append('answer suspiciously short (%d chars)' % len(content))
faults = sum(1 for l in open(log, errors='ignore')
             if ('rank' in l and ' err' in l) or 'did not answer' in l
             or re.search(r'\bfault\b', l))
if faults: fails.append('%d fault line(s) in the log' % faults)

def num(x, spec):
    # 'NA', never -1: a missing field must not look like a (negative) measurement
    # — the shell keys its "no dspark line, this arm is unusable" branch on it.
    return 'NA' if x is None else (spec % x)

print('%s %s %s %s %s %s %s' % (
    num(v, '%.2f'), num(d, '%.2f'), num(k, '%.3f'), num(t, '%.3f'),
    'NA' if s is None else '%d' % int(s),
    'OK' if not fails else 'FAIL',
    '; '.join(fails) if fails else ''))
PY
)"
    read -r avv ad2 akk att ass atxt anote <<<"$parsed"

    A_TAG+=("$tag"); A_VERIFY+=("$avv"); A_DRAFT+=("$ad2"); A_MEANK+=("$akk")
    A_TOKSTEP+=("$att"); A_STEPS+=("$ass"); A_TEXT+=("$atxt"); A_NOTE+=("${anote:-$note}")
    if [ "$avv" = NA ]; then
        # No cumulative dspark line => no number for this arm. NOT a slow arm: an
        # unusable one, and the table must not read as a result.
        echo "   UNMEASURED: no '[dspark] steps=…' line — was DSV41_TIMING=1 effective? (log $LOGDIR/${tag}.log)"
        unmeasured=1
    fi
    [ "$atxt" = FAIL ] && echo "   TEXT: $anote"
}

echo
echo "== arms (serial, same session, same binary+.so) =="
run_arm C1 "DSV41_VERIFY_GRAPH=0" "baseline (no verify graph)"
run_arm C2 "DSV41_VERIFY_GRAPH=1" "verify graph"
if [ -n "$GATE" ]; then
    run_arm C3 "DSV41_VERIFY_GRAPH=1 $GATE=1" "verify graph + multi-row ($GATE)"
elif [ -n "$FORCE_C3" ]; then
    run_arm C3 "DSV41_VERIFY_GRAPH=1" "graph again (no multi-row gate: C3 == C2)"
else
    A_TAG+=("C3"); A_VERIFY+=("SKIP"); A_DRAFT+=("-"); A_MEANK+=("-"); A_TOKSTEP+=("-")
    A_STEPS+=("-"); A_TEXT+=("-")
    A_NOTE+=("skipped: no multi-row env gate in the tree (it is the default) — C3 == C2")
fi

# ---------------------------------------------------------------------------
# 5. The table. Δ verify is vs C1 (negative = faster).
# ---------------------------------------------------------------------------
echo
python3 - "${A_TAG[@]}" -- "${A_VERIFY[@]}" -- "${A_DRAFT[@]}" -- "${A_MEANK[@]}" -- \
        "${A_TOKSTEP[@]}" -- "${A_STEPS[@]}" -- "${A_TEXT[@]}" -- "${A_NOTE[@]}" <<'PY' \
    | tee "$LOGDIR/table.txt"
import sys
argv = sys.argv[1:]
groups = [[]]          # the tag list comes BEFORE the first '--'
for a in argv:
    if a == '--': groups.append([])
    else: groups[-1].append(a)
tag, ver, dra, mek, tok, ste, txt, not_ = groups

def f(x):
    try: return float(x)
    except ValueError: return None

base = f(ver[0]) if ver else None
print('%-4s %-38s %9s %9s %7s %9s %7s %9s %s' %
      ('ARM', 'NOTE', 'verify_ms', 'draft_ms', 'mean-k', 'tok/step', 'steps', 'Δverify', 'TEXT'))
for i in range(len(tag)):
    v = f(ver[i])
    dv = ('%+.2f' % (v - base)) if (v is not None and base) else '—'
    print('%-4s %-38s %9s %9s %7s %9s %7s %9s %s' %
          (tag[i], not_[i][:38],
           ver[i] if v is None else '%.2f' % v,
           dra[i] if f(dra[i]) is None else '%.2f' % float(dra[i]),
           mek[i] if f(mek[i]) is None else '%.3f' % float(mek[i]),
           tok[i] if f(tok[i]) is None else '%.3f' % float(tok[i]),
           ste[i], dv, txt[i]))
print()
print('accept red line: mean-k > 1.0 (dspark_verify.rs) — a real commit must beat the single-row path.')
PY

text_fail=0
for t in "${A_TEXT[@]}"; do [ "$t" = FAIL ] && text_fail=1; done
if [ "$unmeasured" = 1 ]; then
    echo "verify_spec_bench: at least one arm produced NO number — the table is not evidence about the kernels."
    exit 2
fi
if [ "$text_fail" = 1 ]; then
    echo "verify_spec_bench: a TEXT check FAILED — the speed numbers are not usable for a verdict."
    exit 1
fi
echo "verify_spec_bench: all arms measured and text-clean. Logs: $LOGDIR/<arm>.{log,dspark,resp.json}"
exit 0
