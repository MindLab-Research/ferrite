#!/usr/bin/env bash
# lazy_l45_ab.sh — the LAZY (m=1 per-row verify) A/B for the four L4/L5 items that
# are already in tree and have NEVER been measured on the lazy path:
#
#   arm base : the 91.1 clean stack (l49_ab.sh:184-189 BASE_ENV, verbatim)
#   arm 1b   : + DSV41_MROWS_ACT_CPASYNC=1          (gemm_fp8_mrows activation cp.async16)
#   arm b6   : + DSV41_VERIFY_WOB_MROWS_F32=1       (verify wo_b: m quant + 1 proj -> 1 launch)
#   arm b5   : + DSV41_GATE_MROWS_ROUTE=1           (MoE gate GEMV + route -> 1 launch)
#   arm b4   : + DSV41_RMSNORM_ROPE_MROWS=1         (kv norm + rope -> 1 launch; needs FORK)
#   arm l48  : + DSV41_HC_DL_KCHUNK=1               (hc dots+LATE K-chunk pipeline)
#   arm front0: DSV41_HC_FRONT_ROWS=0               (the L4-7 REMOVAL arm, base minus L4-7)
#
# WHY THESE ARMS AND NOT OTHERS — see docs/agent/lazy-l45-next-ab-design.md:
#   * L4-7 (hc side stream) is NOT a pending item: it is fully in tree
#     (`dsv41_hc_front_split(..., side_dl)` + `DSV41_HC_DL_SIDE` default ON) AND
#     `DSV41_HC_FRONT_ROWS=1` is already in the 91.1 BASE_ENV below. The only
#     remaining move is the gate DEFAULT (OFF in code), and that is `front0`'s job
#     — the arm that measures what L4-7 is actually worth before anyone flips it.
#   * 1a (`DSV41_MROWS_FOLD_R`) is a PROVABLE no-op on this path: the lazy block is
#     m = 1, and `dsv41_mrows_fold_r_for(n, 1)` clamps into [1, 1] at EVERY gate
#     setting => fold_r = 1, ng = 1, grid = nt, i.e. byte-identical to gate OFF.
#     Measured on real hardware (tests_dsv41_gemm_mrows.cu, `mr_fold_r_contract`).
#     It is deliberately NOT an arm here.
#
# THE ECONOMICS THAT MAKE LAZY DIFFERENT (do not import the batched ms numbers):
#   * a lazy round runs k_emit = k_acc + 1 rows, each a separate
#     `step_rows_sync(1)` -> 40 x `layer_rows(m=1)`; at the observed k_emit ~2.214
#     that is ~88.6 layer-rows/step (vs the batched block's 40/step);
#   * `DSV41_VERIFY_GRAPH=1` + `VERIFY_GRAPH_SLOTS = 3` gives the m = 1 row its own
#     graph slot (23885a7: ~9.5 ms/row bare -> ~6.15 ms/row graphed), so inside the
#     graph there is NO submit cost per launch;
#   * => items whose ONLY saving is "one fewer launch" (b5/b4) keep just one graph
#     node's execution; items that remove WORK (1b: instructions inside the kernel;
#     b6: the whole quant_fp8 kernel + the activation LUT decode) keep all of it.
#     That is why the expected order on lazy is 1b >= b6 > b5 ~ b4, and why the
#     plan's -0.13/-0.66/-0.79 ms (batched, 40 launches/step) do NOT transfer.
#
# 判据 (per arm; all four legs, none of them optional)
#   ① 活性  — the nsys kernel-count relation named in the design's §4.2 table, or
#             (1b) the GridX distribution + the in-tree bit-parity suite. `1b` has
#             no symbol of its own, so its env read-back proves the variable
#             REACHED the process, never that the kernel branch was taken.
#   ② 正确性— bit-exact arms (1b / b5 / b4) must not move a single token; b6 is NOT
#             bit-exact (it skips the fp8 round trip and is strictly MORE accurate),
#             so it is judged by the red lines: the first 61 counting lines, the
#             出师表 must contain 先帝创业未半, zero Latin, faults=0, ar5-hang=0,
#             plus DSV41_DIFF_EAGER mismatch not increasing.
#   ③ 收益  — same-session back-to-back `steady_median` (STEADY_SKIP rounds
#             dropped) vs arm base. nsys is NEVER read for ms (the v5 publish spin
#             is amplified ~300x under the profiler).
#   ④ 生效  — `tr '\0' '\n' < /proc/<pid>/environ | grep DSV41_ | sort` per arm
#             (the R6 trap: a gate that was set but never took effect).
#
# 口径 (the project rules this script encodes)
#   1. SERIAL, ONE SERVE AT A TIME, torn down between arms.
#   2. ONE PROMPT PER SERVE (`[dspark]` accumulators are process-level).
#   3. SAME BINARY + SAME .so ON EVERY ARM; the symbol precheck below refuses to
#      run a Rust-gated arm whose symbol is missing (it would be SILENTLY inert:
#      `Ok(false)`, the chain falls back to the old pair, and the arm reads "no
#      effect"). 1b has no symbol — its evidence is the parity suite.
#   4. `.cu` changed => `bash kernels/cuda/build.sh 103a` BEFORE `cargo build`, and
#      `kernels/cuda/.build_id` must match the id embedded in the binary.
#
# USAGE
#   bash scripts/lazy_l45_ab.sh                     # build + the 7 serial serves
#   ARMS="base 1b b6" bash scripts/lazy_l45_ab.sh   # a subset
#   MAXTOK=1000 STEADY_SKIP=20 bash scripts/lazy_l45_ab.sh
#   bash scripts/lazy_l45_ab.sh --dry-run           # print the plan, launch nothing
#   bash scripts/lazy_l45_ab.sh --no-build          # reuse the deployed pair
#
# Logs: $LOGDIR/<arm>.{log,dspark,resp.json,metrics,txt,env}  (LOGDIR=/tmp/lazy_l45_ab)
# Exit: 0 = every arm ran and every arm's text/red-line leg held; 1 = at least one
#       arm failed a leg; 2 = the run is not usable (harness error / missing pair).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

NODE="${NODE:-ubuntu@43.202.208.136}"
ARCH="${ARCH:-103a}"
RROOT="${RROOT:-ferrite}"
LOGDIR="${LGA_LOGDIR:-/tmp/lazy_l45_ab}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)

PORT_BASE="${PORT_BASE:-8230}"
MAXTOK="${MAXTOK:-1000}"                        # 出师表, full answer, no truncation
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"              # x5s
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"
STEADY_SKIP="${STEADY_SKIP:-20}"
# The acceptance gate. A launch-fusion arm on the GRAPHED lazy path can land
# inside the noise, so the floor is the noise band, not the batched plan's ms.
MIN_GAIN_MS="${MIN_GAIN_MS:-0.8}"

PROMPT="${PROMPT:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"

# ---------------------------------------------------------------------------
# The FIXED base env — the 91.1 clean stack (l49_ab.sh:184-189, verbatim; the
# design §2 of nsys-clean-stack-91-design.md and batch-reverification-plan §4 both
# name this as authoritative). NOT set on purpose: DSV41_MROWS_FOLD_R (1a — a
# proven no-op at m=1, see the header), DSV41_SWAPAB (it would disable the mrows
# dispatch the whole batch rides on), DSV41_NO_GEMV_FP8 (same).
# DSV41_HC_FRONT_ROWS=1 IS in here: that is L4-7, already in the stack.
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

ARMS="${ARMS:-base 1b b6 b5 b4 l48 front0}"

# The arm's EXTRA env — exactly one variable for the four measured items, so each
# delta is attributable to one gate (house rule: one gate, one commit, one A/B).
arm_extra() {
    case "$1" in
        base)   echo "" ;;
        1b)     echo "DSV41_MROWS_ACT_CPASYNC=1" ;;
        b6)     echo "DSV41_VERIFY_WOB_MROWS_F32=1" ;;
        b5)     echo "DSV41_GATE_MROWS_ROUTE=1" ;;
        b4)     echo "DSV41_RMSNORM_ROPE_MROWS=1" ;;
        l48)    echo "DSV41_HC_DL_KCHUNK=1" ;;
        # front0 SUBTRACTS: it is base minus L4-7 (the removal arm). Its delta is
        # what flipping the gate DEFAULT would have to beat.
        front0) echo "DSV41_HC_FRONT_ROWS=0" ;;
        *)      echo "ERROR" ;;
    esac
}

# What a Rust-gated arm needs in the .so, and the activity evidence to look for.
arm_symbol() {
    case "$1" in
        1b)  echo "" ;;                                               # no symbol: a .cu getenv
        b6)  echo "dsv41_gemm_fp8_mrows_f32" ;;
        b5)  echo "ferrite_gemv_bf16_v2_mrows_route" ;;
        b4)  echo "dsv41_rmsnorm_rope_mrows" ;;
        l48) echo "" ;;                                               # .cu getenv (g_hc_dl_kchunk)
        *)   echo "" ;;
    esac
}

arm_note() {
    case "$1" in
        base)   echo "the 91.1 clean stack (reference)" ;;
        1b)     echo "1b  gemm_fp8_mrows activation staging -> cp.async16  [bit-exact; parity MEASURED incl. m=1]" ;;
        b6)     echo "B6  verify wo_b -> ONE f32 mrows launch          [NOT bit-exact: red lines only]" ;;
        b5)     echo "B5  MoE gate GEMV + route -> ONE launch            [bit-exact kernel]" ;;
        b4)     echo "B4  kv norm + rope -> ONE launch (rides FORK side stream) [bit-exact kernel]" ;;
        l48)    echo "L4-8 hc dots+LATE K-chunk pipeline                 [bit-exact by construction]" ;;
        front0) echo "L4-7 REMOVAL (HC_FRONT_ROWS=0): the only remaining L4-7 move is the DEFAULT" ;;
    esac
}

BUILD=1; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,110p' "$0"; exit 0 ;;
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
    echo "FATAL: another lazy_l45_ab.sh holds $LOCK — arms must never overlap."
    exit 2
fi

echo "== LAZY L4/L5 A/B (m=1 per-row verify, the 91.1 stack as the fixed base) =="
echo "-- node $NODE   arch $ARCH   ports $PORT_BASE..$(($PORT_BASE + $(echo $ARMS | wc -w) - 1))   tp $TP"
echo "-- prompt: 出师表 max_tokens=$MAXTOK   steady-skip=$STEADY_SKIP   gate >= ${MIN_GAIN_MS}ms"
echo "-- arms: $ARMS"
for a in $ARMS; do
    [ "$(arm_extra "$a")" = "ERROR" ] && { echo "FATAL: unknown arm '$a'"; exit 2; }
    printf -- "--   %-8s %s\n" "$a" "$(arm_note "$a")"
done

# ---------------------------------------------------------------------------
# 0. The pair must exist, and the arms' symbols must be in it.
# ---------------------------------------------------------------------------
SO_REL="kernels/cuda/libferrite_kernels.so"
BIN_REL="target/release/ferrite-serve"
HAS_SO="$(rssh "[ -f ~/$RROOT/$SO_REL ] && echo yes || echo no")"
HAS_BIN="$(rssh "[ -f ~/$RROOT/$BIN_REL ] && echo yes || echo no")"
if [ "$HAS_SO" != yes ] || [ "$HAS_BIN" != yes ]; then
    echo "FATAL: the deployed pair is missing on $NODE (so=$HAS_SO bin=$HAS_BIN)."
    echo "       Run the build (or drop --no-build) before the arms — a stale pair makes every arm meaningless."
    exit 2
fi
echo "-- symbol precheck (a missing symbol makes a Rust-gated arm SILENTLY inert):"
for a in $ARMS; do
    sym="$(arm_symbol "$a")"
    [ -z "$sym" ] && continue
    n="$(rssh "nm -D ~/$RROOT/$SO_REL 2>/dev/null | grep -c '$sym'")"
    if [ "${n:-0}" -lt 1 ]; then
        echo "   FATAL: $a needs '$sym' in the .so and it is ABSENT (count=$n)."
        echo "          Rebuild (build.sh $ARCH) — otherwise this arm reads 'no effect' for the wrong reason."
        exit 2
    fi
    echo "   $a: $sym present (count=$n)"
done
echo "-- DRY RUN: the plan above; a real run would start $(echo $ARMS | wc -w) serial serves, one arm at a time."
if [ "$DRY" = 1 ]; then
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Build (unless --no-build). .cu first, then the Rust binary (AGENTS.md order).
# ---------------------------------------------------------------------------
if [ "$BUILD" = 1 ]; then
    echo "-- building on $NODE (kernels/cuda/build.sh $ARCH, then cargo build --release)"
    rssh "cd ~/$RROOT && bash kernels/cuda/build.sh $ARCH && cargo build --release -p ferrite-dsv41 2>&1 | tail -5" \
        || { echo "FATAL: build failed"; exit 2; }
    # The id must match, or the loader refuses the pair at startup.
    rssh "cd ~/$RROOT && echo -n 'build_id: '; cat kernels/cuda/.build_id 2>/dev/null || echo '(none)'"
fi

# ---------------------------------------------------------------------------
# 2. Teardown by EXACT process name (`pkill -f` would match this script itself).
# ---------------------------------------------------------------------------
kill_serves() {
    rssh "pkill -9 -x ferrite-serve 2>/dev/null; sleep $TEARDOWN_SLEEP; pgrep -x ferrite-serve | wc -l"
}
left="$(kill_serves)"
[ "$left" != 0 ] && { echo "FATAL: $left ferrite-serve process(es) survived teardown — arms must never overlap."; exit 2; }

# ---------------------------------------------------------------------------
# 3. One arm: start, wait healthy, one request, stop, save everything.
# ---------------------------------------------------------------------------
run_arm() {
    local arm="$1" port="$2"
    local extra; extra="$(arm_extra "$arm")"
    echo "-- arm $arm (port $port)  extra=[${extra:-none}]"

    rssh "cd ~/$RROOT && rm -f $LOGDIR/$arm.log $LOGDIR/$arm.dspark $LOGDIR/$arm.env && \
          env $BASE_ENV $extra CUDA_VISIBLE_DEVICES=$GPU_LIST NCCL_NVLS_ENABLE=0 \
          DSV41_MODEL_DIR=$MODEL_DIR DSV41_KERNELS=\$PWD/kernels/cuda/libferrite_kernels.so \
          nohup ./$BIN_REL --model $MODEL_NAME --port $port --tp $TP \
              > $LOGDIR/$arm.log 2>&1 & echo started" >/dev/null

    local i=0
    while [ $i -lt "$HEALTH_TRIES" ]; do
        sleep 5
        if rssh "$CURL http://127.0.0.1:$port/health 2>/dev/null" >/dev/null 2>&1; then break; fi
        i=$((i + 1))
    done
    if [ $i -ge "$HEALTH_TRIES" ]; then
        echo "   FATAL: $arm never became healthy — see $LOGDIR/$arm.log"
        return 2
    fi

    # ④ the env read-back (the R6 trap). 1b/1a live in the process env either way;
    # the _kernel_ branch is what the parity suite and the nsys GridX have to prove.
    pid="$(rssh "pgrep -x ferrite-serve | head -1")"
    rssh "tr '\0' '\n' < /proc/$pid/environ | grep -E '^DSV41_|^NCCL_' | sort" > "$LOGDIR/$arm.env" 2>/dev/null

    rssh "cd ~/$RROOT && $CURL -s --noproxy '*' -X POST http://127.0.0.1:$port/v1/chat/completions \
          -H 'Content-Type: application/json' \
          -d '{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAXTOK}' \
          > $LOGDIR/$arm.resp.json" >/dev/null 2>&1

    rssh "pkill -9 -x ferrite-serve 2>/dev/null; sleep $TEARDOWN_SLEEP; pgrep -x ferrite-serve | wc -l" >/dev/null
    rssh "grep -a '\[dspark\]' $LOGDIR/$arm.log > $LOGDIR/$arm.dspark 2>/dev/null; true"
    return 0
}

# ---------------------------------------------------------------------------
# 4. Parse one arm: the steady step wall, the k_acc histogram, the red lines.
# ---------------------------------------------------------------------------
parse_arm() {
    python3 - "$LOGDIR/$1.dspark" "$LOGDIR/$1.resp.json" "$LOGDIR/$1.log" \
               "$LOGDIR/$1.txt" "$STEADY_SKIP" "$1" <<'PY'
import hashlib, json, re, sys

dspark_p, resp_p, log_p, txt_p, arm = sys.argv[1:6]
skip = int(sys.argv[6])


def read(p):
    try:
        return open(p, "r", errors="ignore").read()
    except OSError:
        return ""


def field(line, key):
    i = line.find(key)
    if i < 0:
        return None
    m = re.match(r"[-+0-9.eE]+", line[i + len(key):])
    return float(m.group(0)) if m else None


log = read(log_p)
dspark_txt = read(dspark_p)

# ---- the process-level `[dspark] steps=… mean-k=… verify=…` line: corroboration
#      only (its accumulators live outside the request loop, so it is a SERVE
#      average, never the per-step wall).
dspark_line = ""
for line in dspark_txt.splitlines():
    if "[dspark] steps=" in line and "mean-k=" in line:
        dspark_line = line.rstrip("\n")
mean_k = field(dspark_line, "mean-k=")
verify_ms = field(dspark_line, "verify=")
steps = field(dspark_line, "steps=")

# ---- the per-round wall: `[dsv41] step pos=<p>: <X>ms`
seq = [(int(a), float(b)) for a, b in re.findall(r"\[dsv41\] step pos=(\d+): ([\d.]+)ms", log)]
wall = [ms for _, ms in seq]
steady = wall[skip:] if len(wall) > skip + 1 else wall[:]
if len(steady) > 1:
    steady = steady[:-1]          # the stop round can cut a block short


def stats(xs):
    if not xs:
        return 0, None, None, None
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return n, sum(s) / n, med, s[0]


sn, smean, smed, smin = stats(steady)

# ---- k_acc histogram from the position deltas: k_emit = k_acc + 1 per round.
#      NOT optional: a broken transport shows up here before it shows up in ms.
deltas = [seq[i + 1][0] - seq[i][0] for i in range(len(seq) - 1)]
kacc = [d - 1 for d in deltas if 1 <= d <= 7]
kacc_mean = (sum(kacc) / len(kacc)) if kacc else None

# ---- which verify shape the graph captured: m=1 IS the lazy arm. If the log
#      shows m=5/m=6 the route stayed on the batched arm and the run is not a
#      lazy measurement at all.
found = sorted({int(x) for x in re.findall(r"\[verify_graph\] captured verify_graph_m(\d+)", log)})
shapes = ",".join("m=%d" % x for x in found) if found else ""

# ---- the answer text (saved for the md5 / red-line compare across arms).
content = ""
try:
    j = json.loads(read(resp_p))
    content = j["choices"][0]["message"]["content"]
except Exception:
    content = ""
open(txt_p, "w").write(content)
md5 = hashlib.md5(content.encode()).hexdigest() if content else "NA"
latin = sum(1 for ch in content if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
adj = 0
for i in range(2, len(content)):
    if content[i] == content[i - 1] == content[i - 2] and not content[i].isspace():
        adj += 1
faults = len(re.findall(r"faults=[1-9]", log))
hang = len(re.findall(r"ar5-hang", log))

print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
    arm,
    "%.2f" % smed if smed is not None else "NA",
    "%.2f" % smean if smean is not None else "NA",
    "%.2f" % smin if smin is not None else "NA",
    str(sn),
    "%.3f" % kacc_mean if kacc_mean is not None else "NA",
    "%.3f" % mean_k if mean_k is not None else "NA",
    "%.2f" % verify_ms if verify_ms is not None else "NA",
    shapes or "-",
    str(latin),
    str(adj),
    "OK" if ("先帝创业未半" in content and latin == 0) else "FAIL",
))
PY
}

echo "== P1/P2 per arm: one serial serve each =="
printf "%-8s %-9s %-9s %-8s %-5s %-8s %-8s %-8s %-6s %-6s %-4s %s\n" \
       arm steady_med steady_mean steady_min n kacc_mean mean-k verify shapes latin adj text
port="$PORT_BASE"
RAN=""
for a in $ARMS; do
    parse_out="$(parse_arm "$a" 2>/dev/null)"
    if run_arm "$a" "$port"; then
        RAN="$RAN $a"
    else
        echo "-- arm $a FAILED to run"
    fi
    parse_out="$(parse_arm "$a")"
    echo "$parse_out" | sed 's/\t/ /g'
    echo "$parse_out" > "$LOGDIR/$a.row"
    port=$((port + 1))
done

echo
echo "== Δ table (vs arm base) and the verdict =="
python3 - "$LOGDIR" "$ARMS" "$MIN_GAIN_MS" "$STEADY_SKIP" <<'PY'
import os, sys

logdir, arms, min_gain = sys.argv[1], sys.argv[2].split(), float(sys.argv[3])


def load(arm):
    p = os.path.join(logdir, "%s.row" % arm)
    try:
        f = open(p).read().rstrip("\n").split("\t")
    except OSError:
        return None
    if len(f) < 12:
        return None
    return {
        "med": float(f[1]) if f[1] != "NA" else None,
        "n": int(f[4]) if f[4] != "NA" else 0,
        "kacc": float(f[5]) if f[5] != "NA" else None,
        "shapes": f[8],
        "latin": int(f[9]),
        "adj": int(f[10]),
        "text": f[11],
    }


base = load("base")
print("%-8s %-10s %-10s %-8s %-7s %-6s %s" % ("arm", "steady_med", "Δvs base", "n", "kacc", "shapes", "verdict"))
bad = []
for a in arms:
    m = load(a)
    if m is None or m["med"] is None or m["n"] < 5:
        print("%-8s %-10s %-10s %-8s %-7s %-6s %s" % (a, "UNMEASURED", "-", m["n"] if m else 0,
                                                      "-", "-", "NOT USABLE"))
        bad.append(a)
        continue
    if base and base["med"] is not None and a != "base":
        d = m["med"] - base["med"]
        # front0 SUBTRACTS: a NEGATIVE delta there means removing L4-7 was FASTER,
        # i.e. L4-7 is not paying for itself on this stack.
        tag = "faster" if d < 0 else "slower"
        ok = (m["text"] == "OK")
        print("%-8s %-10.2f %+-10.2f %-8d %-7s %-6s %s%s" % (
            a, m["med"], d, m["n"],
            "%.3f" % m["kacc"] if m["kacc"] is not None else "-",
            m["shapes"] or "-",
            "" if ok else "TEXT/RED-LINE FAIL ",
            ("(%s; %s gate %.2fms)" % (tag, "meets" if abs(d) >= min_gain else "inside noise", min_gain)) if a != "front0"
            else ("(%s; L4-7 %s paying off)" % (tag, "is" if d >= min_gain else "is NOT"))))
        if not ok:
            bad.append(a)
    else:
        print("%-8s %-10.2f %-10s %-8d %-7s %-6s %s" % (
            a, m["med"], "-", m["n"],
            "%.3f" % m["kacc"] if m["kacc"] is not None else "-",
            m["shapes"] or "-", m["text"]))

print()
print("READ THIS WITH THE DESIGN OPEN (docs/agent/lazy-l45-next-ab-design.md §5.2):")
print(" * ③ 收益 is `steady_median` ONLY. Do not read nsys ms.")
print(" * 1b / b5 / b4 are bit-exact: the TEXT column must be OK and the answer md5")
print("   must equal base's. b6 is NOT: it only has to hold the red lines.")
print(" * shapes must be m=1 (that is the lazy arm). m=5/m=6 means the route never")
print("   took the lazy branch and this run is not evidence about lazy at all.")
if bad:
    print("VERDICT: NOT all arms clean (%s) — fix before reading any delta." % ",".join(bad))
    sys.exit(1)
print("VERDICT: all arms ran and held their text/red-line leg.")
PY
