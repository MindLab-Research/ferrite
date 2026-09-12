#!/usr/bin/env bash
# verify_graph_ab.sh — the DSpark verify CUDA graph A/B: `DSV41_VERIFY_GRAPH=0` vs `=1`.
#
# WHY THIS SCRIPT EXISTS (read before trusting a number)
#   The verify block is ~7000 launches per verify and each streaming launch pays
#   its ~2.9us of submit time on top of the graph's ~0.4us/node dispatch floor, so
#   the graph is the single biggest lever under `verify_ms` (the 400 tok/s plan's
#   performance leg). The switch is opt-in (default OFF) and the capture is gated
#   (`chain_dev.rs::verify_graph_gate`) — which means an A/B can come back
#   "no change" for a graph that NEVER RAN. Three distinct states look identical
#   from the outside, so this script refuses to report a verdict unless the
#   engagement proof is present:
#     1. gate closed      → no `[verify_graph]` line at all;
#     2. capture failed   → `[verify_graph] capture FAILED (…): <reason>`, and the
#                           request finishes on the direct launches (failure is a
#                           DESIGNED silent degradation — only the print makes it
#                           visible);
#     3. engaged          → `[verify_graph] captured m=… at pos=…`, then replays.
#   Only (3) makes the `=1` arm a measurement of the graph.
#
# 口径 (the project rules this script encodes)
#   1. SERIAL, ONE SERVE AT A TIME, torn down between arms — two serves on the
#      same 8 GPUs do not give noisy numbers, they give meaningless ones.
#   2. THE `[dspark] steps=…` LINE IS A **SERVE-PROCESS** AVERAGE. Its
#      accumulators (`dspark_steps`/`dspark_verify_ms`, serve.rs) are declared
#      OUTSIDE the request loop and are never reset per request, so two prompts in
#      one serve would report a mixed average in the last line. Each (arm, prompt)
#      therefore gets its OWN serve: 4 serves, strictly serial.
#   3. SAME BINARY + SAME .so ON EVERY ARM, PROVEN (the loader refuses a
#      mismatched pair; see dsv41_serve_ab.sh's header for why `cargo build` alone
#      can never heal a stale pair).
#   4. AR v5: both arms export `DSV41_AR_V5=1` for the record, but NOTE this is
#      NOT the deciding flag — `ar_v5()` is `DSV41_GRAPH_STEP != 0 ||
#      DSV41_AR_V5 != 0` with both legs defaulting ON (tp.rs), so under TP8 it is
#      true unless `DSV41_GRAPH_STEP=0` AND `DSV41_AR_V5=0` are BOTH set. A
#      "graph never engaged" result is therefore never explained by the AR clause
#      alone; read the `[verify_graph]` lines instead.
#
# 判据 (the acceptance rule, printed as a verdict at the end)
#   * engage: the `=1` arm MUST print `[verify_graph] captured …` (else exit 2 —
#     the run is not evidence about the graph);
#   * speed : verify_ms(graph=1) <= verify_ms(graph=0) - 10 ms (the plan's
#     expected ≥10 ms saving; a smaller/positive delta fails the leg);
#   * text  : the DOUBLE-CHAR COUNT must be identical between the two arms of the
#     same prompt (the graph may not change a single emitted token), AND the digit
#     prompt must contain no adjacently repeated number (the historical
#     self-repetition shape).
#
# USAGE
#   bash scripts/verify_graph_ab.sh                # build + 4 serial serves
#   bash scripts/verify_graph_ab.sh --no-build     # reuse the pair on the node
#   bash scripts/verify_graph_ab.sh --dry-run      # pre-flight only; launches nothing
#   SYNC=1 bash scripts/verify_graph_ab.sh         # rsync THIS tree to the node first
#   MAXTOK_SH=300 MAXTOK_DI=200 PORT_BASE=8180 bash scripts/verify_graph_ab.sh
#
# OUTPUT: per (arm, prompt) verify_ms / draft_ms / mean-k / tok/step / steps, the
#   double-char count + the engagement proof, the Δ table, the raw last dspark
#   line, and the verdict. Logs: $LOGDIR/<tag>.{log,dspark,resp.json}
#   Exit: 0 = both legs of the 判据 hold; 1 = TEXT/verdict failure (compare the
#   table, do NOT trust the speed numbers); 2 = harness/unusable run.
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
LOGDIR="${VG_LOGDIR:-/tmp/verify_graph_ab}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
CURL="curl -s --noproxy '*'"

PORT_BASE="${PORT_BASE:-8180}"
MAXTOK_SH="${MAXTOK_SH:-300}"             # 出师表: the long-prompt yardstick
MAXTOK_DI="${MAXTOK_DI:-200}"             # the digit task
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"        # x5s
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"
MIN_GAIN_MS="${MIN_GAIN_MS:-10}"          # the plan's expected verify_ms saving

# 出师表 is the project's long-prompt yardstick (dspark_verify.rs PROMPTS); the
# digit task is the repetition detector — the shape that used to double every
# number, so a per-token difference between the two arms shows up here first.
PROMPT_SH="${PROMPT_SH:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"
PROMPT_DI="${PROMPT_DI:-请从 1 数到 100，每个数字单独占一行，只输出数字本身，不要任何解释。}"

BUILD=1; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,70p' "$0"; exit 0 ;;
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
    echo "FATAL: another verify_graph_ab.sh holds $LOCK — arms must never overlap."
    exit 2
fi

echo "== DSpark verify-graph A/B (DSV41_VERIFY_GRAPH=0 vs 1) =="
echo "-- node $NODE   arch $ARCH   ports $PORT_BASE..$(( PORT_BASE + 3 ))   tp $TP"
echo "-- prompts: 出师表 max_tokens=$MAXTOK_SH   digits max_tokens=$MAXTOK_DI"

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
    echo "-- dry run: a real run would start 4 serves on ports $PORT_BASE..$(( PORT_BASE + 3 )); nothing was launched."
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
# 4. One serve per (arm, prompt), strictly serial.
# ---------------------------------------------------------------------------
declare -a T_TAG T_ARM T_PROMPT T_VERIFY T_DRAFT T_MEANK T_TOKSTEP T_STEPS T_ENGAGED T_NOTE
unmeasured=0     # an arm produced no cumulative dspark line at all
engaged_missing=0
N=0

run_case() {
    local arm="$1" prompt="$2" max_tok="$3" tag="$4" port=$(( PORT_BASE + N ))
    local rlog="~/vg_${tag}.log"
    N=$(( N + 1 ))
    echo
    echo "---- case $tag (port $port): DSV41_VERIFY_GRAPH=$arm  prompt=$prompt"

    kill_serves >/dev/null                     # every case starts from a clean box
    rssh "cd ~/$RROOT && nohup env CUDA_VISIBLE_DEVICES=$GPU_LIST \
          LD_LIBRARY_PATH=\$HOME/$RROOT/kernels/cuda \
          DSV41_KERNELS=\$HOME/$RROOT/kernels/cuda/libferrite_kernels.so \
          DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1 \
          DSV41_VERIFY_GRAPH=$arm DSV41_AR_V5=1 \
          ./target/release/ferrite-serve --model dsv41 --serve --tp $TP \
          --model-dir $MODEL_DIR --port $port > $rlog 2>&1 &" \
        || { echo "FATAL: could not spawn the serve for $tag"; exit 2; }

    if ! rssh "for i in \$(seq 1 $HEALTH_TRIES); do $CURL -m 2 http://localhost:$port/health >/dev/null 2>&1 && exit 0; sleep 5; done; exit 1"; then
        echo "FATAL: $tag never became healthy (log $rlog); last lines:"
        rssh "tail -15 $rlog" | sed 's/^/    | /'
        teardown "$port"; exit 2
    fi

    local body t0 t1
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
            "$MODEL_NAME" "$prompt" "$max_tok")"
    t0=$(date +%s)
    rssh "cd ~/$RROOT && $CURL -m 900 http://localhost:$port/v1/chat/completions \
          -H 'Content-Type: application/json' -d '$body'" >"$LOGDIR/${tag}.resp.json" 2>/dev/null
    t1=$(date +%s)
    rssh "grep 'dspark] steps' $rlog | tail -1" >"$LOGDIR/${tag}.dspark" 2>/dev/null
    rssh "cat $rlog" >"$LOGDIR/${tag}.log" 2>/dev/null

    teardown "$port"
    echo "   e2e $(( t1 - t0 ))s   $(head -c 120 "$LOGDIR/${tag}.dspark")"

    # Numbers + engagement proof + text check, one python pass over the artifacts.
    local parsed
    parsed="$(python3 - "$LOGDIR/${tag}.dspark" "$LOGDIR/${tag}.resp.json" "$LOGDIR/${tag}.log" "$prompt" <<'PY'
import json, re, sys
dspark, resp, log, prompt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def field(line, key):
    i = line.find(key)
    if i < 0: return None
    m = re.match(r'[-+0-9.eE]+', line[i + len(key):])
    return float(m.group(0)) if m else None

line = ''
try:
    for l in open(dspark):
        if 'steps=' in l: line = l.rstrip('\n')
except OSError: pass
v = field(line, 'verify='); d = field(line, 'draft=')
k = field(line, 'mean-k='); t = field(line, 'tok/step='); s = field(line, 'steps=')

txt = ''
try:
    log_txt = open(log, errors='ignore').read()
except OSError:
    log_txt = ''

# The engagement proof. `captured` is the ONLY line that makes the `=1` arm a
# measurement of the graph; `FAILED` is the designed silent degradation made loud.
engaged = 'captured' if '[verify_graph] captured' in log_txt else \
          ('failed' if '[verify_graph] capture FAILED' in log_txt else 'no')

content = ''
try:
    content = json.load(open(resp))['choices'][0]['message']['content']
except Exception as e:
    print('NA NA NA NA NA %s resp-not-parseable: %s' % (engaged, e))
    sys.exit(0)

# The project's double-char rule (dspark_verify.rs), inlined: whitespace repeats
# are NOT corruption, an adjacent identical character is.
ch = list(content)
dbl = sum(1 for i in range(1, len(ch)) if ch[i] == ch[i-1] and not ch[i].isspace())

fails = []
if not content: fails.append('empty answer')
if len(content) < 40: fails.append('answer suspiciously short (%d chars)' % len(content))
if '出师表' in prompt or '先帝创业未半' in prompt:
    if '先帝创业未半' not in content:
        fails.append('missing "先帝创业未半"')
else:
    nums = [int(x) for x in re.findall(r'\d+', content)]
    if len(nums) < 50: fails.append('only %d numbers in the digit answer' % len(nums))
    rep = [i for i in range(1, len(nums)) if nums[i] == nums[i-1]]
    if rep: fails.append('digit %d repeated adjacently (the self-repetition shape)' % nums[rep[0]])

print('%s %s %s %s %s %s %d %d %s' % (
    'NA' if v is None else '%.2f' % v,
    'NA' if d is None else '%.2f' % d,
    'NA' if k is None else '%.3f' % k,
    'NA' if t is None else '%.3f' % t,
    'NA' if s is None else '%d' % int(s),
    engaged, dbl, len(content),
    'OK' if not fails else '; '.join(fails)))
PY
)"
    read -r av ad ak at as aen adbl alen anote <<<"$parsed"

    T_TAG+=("$tag"); T_ARM+=("$arm"); T_PROMPT+=("$prompt")
    T_VERIFY+=("$av"); T_DRAFT+=("$ad"); T_MEANK+=("$ak"); T_TOKSTEP+=("$at")
    T_STEPS+=("$as"); T_ENGAGED+=("$aen"); T_NOTE+=("${anote:-}")
    if [ "$av" = NA ]; then
        echo "   UNMEASURED: no '[dspark] steps=…' line — was DSV41_TIMING=1 effective? (log $LOGDIR/${tag}.log)"
        unmeasured=1
    fi
    echo "   engagement=$aen  dbl=$adbl chars=$alen  ${anote:-ok}"
    if [ "$arm" = 1 ] && [ "$aen" != captured ]; then
        echo "   ⛔ the graph did NOT engage (engagement=$aen) — this arm is not evidence about the graph."
        engaged_missing=1
    fi
    if [ "$arm" = 0 ] && [ "$aen" = captured ]; then
        echo "   ⛔ DSV41_VERIFY_GRAPH=0 captured a graph?! the env was not honoured."
        engaged_missing=1
    fi
}

echo
echo "== cases (serial, same binary+.so, one serve per case) =="
run_case 0 "$PROMPT_SH" "$MAXTOK_SH" g0_sh
run_case 1 "$PROMPT_SH" "$MAXTOK_SH" g1_sh
run_case 0 "$PROMPT_DI" "$MAXTOK_DI" g0_di
run_case 1 "$PROMPT_DI" "$MAXTOK_DI" g1_di

# ---------------------------------------------------------------------------
# 5. The table + the 判据. Δ verify is =1 minus =0 for the SAME prompt.
# ---------------------------------------------------------------------------
echo
python3 - "$MIN_GAIN_MS" -- "${T_TAG[@]}" -- "${T_ARM[@]}" -- "${T_PROMPT[@]}" -- \
        "${T_VERIFY[@]}" -- "${T_DRAFT[@]}" -- "${T_MEANK[@]}" -- \
        "${T_TOKSTEP[@]}" -- "${T_STEPS[@]}" -- "${T_ENGAGED[@]}" -- "${T_NOTE[@]}" <<'PY' \
    | tee "$LOGDIR/table.txt"
import sys
argv = sys.argv[1:]
min_gain = float(argv[0])
groups = [[]]
for a in argv[1:]:
    if a == '--': groups.append([])
    else: groups[-1].append(a)
tag, arm, prm, ver, dra, mek, tok, ste, eng, note = groups

def f(x):
    try: return float(x)
    except ValueError: return None

key = lambda p: 'digits' if '数到' in p or '\u6570\u5230' in p else '\u51fa\u5e08\u8868'
rows = {}
print('%-7s %-4s %-9s %9s %8s %8s %8s %7s %-8s %5s %s' %
      ('CASE', 'ARM', 'PROMPT', 'verify_ms', 'draft_ms', 'mean-k', 'tok/step', 'steps',
       'ENGAGED', 'dbl', 'TEXT'))
for i in range(len(tag)):
    v = f(ver[i])
    rows[(arm[i], key(prm[i]))] = v
    print('%-7s %-4s %-9s %9s %8s %8s %8s %7s %-8s %5s %s' %
          (tag[i], arm[i], key(prm[i]),
           ver[i] if v is None else '%.2f' % v,
           dra[i] if f(dra[i]) is None else '%.2f' % float(dra[i]),
           mek[i] if f(mek[i]) is None else '%.3f' % float(mek[i]),
           tok[i] if f(tok[i]) is None else '%.3f' % float(tok[i]),
           ste[i], eng[i], '-', note[i] if note[i] else 'ok'))

print()
print('\u5224\u636e (per prompt, graph=1 minus graph=0):')
rc = 0
for p in ('\u51fa\u5e08\u8868', 'digits'):
    a, b = rows.get(('0', p)), rows.get(('1', p))
    if a is None or b is None:
        print('  %-9s SKIP (a case produced no number)' % p); rc = 2; continue
    dv = b - a
    gain = -dv
    ok = gain >= min_gain
    print('  %-9s verify %.2f -> %.2f ms (\u0394%+.2f, gain %+.2f ms, need >= %+.1f)  %s'
          % (p, a, b, dv, gain, min_gain, 'PASS' if ok else 'FAIL'))
    if not ok: rc = 1
print()
print('engage rule: the =1 arm MUST print "[verify_graph] captured …" (see the ENGAGED column).')
sys.exit(rc)
PY
verdict=$?

text_fail=0
# A TEXT FAIL is any non-empty note in the table's TEXT column.
for n in "${T_NOTE[@]}"; do case "$n" in ""|ok) ;; *) text_fail=1 ;; esac; done

echo
if [ "$unmeasured" = 1 ]; then
    echo "verify_graph_ab: at least one case produced NO number — the table is not evidence about the graph."
    exit 2
fi
if [ "$engaged_missing" = 1 ]; then
    echo "verify_graph_ab: the engagement proof is MISSING/wrong — read the [verify_graph] lines in the logs."
    echo "  (gate closed → no line; capture refused → '[verify_graph] capture FAILED …'; engaged → 'captured'.)"
    exit 2
fi
if [ "$verdict" = 2 ]; then
    echo "verify_graph_ab: a case was unmeasurable — no verdict."
    exit 2
fi
if [ "$text_fail" = 1 ]; then
    echo "verify_graph_ab: a TEXT check FAILED — the speed numbers are not usable for a verdict."
    exit 1
fi
if [ "$verdict" = 1 ]; then
    echo "verify_graph_ab: the graph engaged and the text is clean, but the \u2265${MIN_GAIN_MS}ms verify_ms gain did NOT materialise."
    exit 1
fi
echo "verify_graph_ab: engaged, text-consistent, and verify_ms dropped by >= ${MIN_GAIN_MS}ms on both prompts."
echo "Logs: $LOGDIR/<case>.{log,dspark,resp.json}   table: $LOGDIR/table.txt"
exit 0
