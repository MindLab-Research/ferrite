#!/usr/bin/env bash
# =============================================================================
# tcgen05_smoke.sh — SMOKE TEST (correctness-only) for the tcgen05 e4m3 GROUPED
#                    routed gate/up arm (`dsv41_expert_gemm_e4m3_grouped`).
# =============================================================================
#
# WHAT IT DECIDES
#   The grouped tcgen05 e4m3 tile (`tc5::e4x`, `m_grouped_gemm_nt_masked`) has
#   NEVER executed on any GPU. Its source carries TWO undecided `[OPEN]` decode
#   guesses (kernels/cuda/dsv41_experts_mxf4.cu, header + :5334 / :5482):
#     (1) the DENSE non-block-scaled `idesc` format code (a_format=E4M3=0,
#         b_format=E2M1=5, both taken "from the only enum this repo has
#         decoded"); and
#     (2) whether the fp4 operand of the f8f6f4 family is the UNPACKED
#         ("unpacksmem", 1 element/byte) form the kernel stages, or a PACKED
#         (16 B/row) form that would contradict the whole B-side layout.
#   Either one wrong => SILENT WRONG VALUES (the 拉丁/乱码 red line) or an
#   illegal instruction. So the first GPU contact must be a CHEAP smoke, not a
#   full 1000-token bench: this script is that first contact.
#
#   docs/agent/tcgen05-e4m3-grouped-expectation.md §7/§8 is the basis:
#     * §7-第0步  four `dsv41_route_*` symbols missing => the arm CANNOT run
#                 (declines), so DO NOT spend a GPU on it. => STAGE 1.
#     * §7-第3步  "如果时间很紧，可以在第 1 步之前插入一次「短 prompt 冒烟」
#                 (MAXTOK=100~200)(...)，用来在 2 分钟内抓「非法指令 / serve
#                 起不来 / 拉丁红线」".                       => STAGE 2.
#     * §4.2/§8.2 别用「与基线逐字节相同」当判据 (the MMA + fold f32 order
#                 differs from the SIMT fma chain) — use the red lines, and
#                 compare only the LEADING tokens.            => STAGE 3.
#
# GATE CHAIN (the ONLY combination under which the grouped arm dispatches;
# tcgen05-unblock + post-swallow-performance-plan confirmed). In `moe_rows`
# (the SPEC/verify multi-row path) `moe_experts_grouped_gate_up`
# (chain_dev.rs:10606) runs the kernel iff ALL of:
#   DSV41_EXPERT_ACT_E4M3=1     e4m3 activation bytes (the kernel eats
#                               kind::f8f6f4). ALREADY in the production matrix.
#   DSV41_EXPERT_TCGEN05_E4M3=1 the e4m3 tcgen05 family's runtime gate.
#   DSV41_EXPERT_GROUPED=1      builds the permuted layout the kernel indexes.
#   DSV41_GATEUP_FUSE=0         the e4x epilogue only CLAMPS, it never fuses
#                               swiglu, so the fused shape declines the arm.
#   DSV41_EXPERT_ILV=0          plain (non-interleaved) w1/w3 planes; the
#                               interleaved layout is unreadable by this arm.
# ⚠️ DSV41_EXPERT_ILV=0 is REDUNDANT (not an independent variable): `ilv_ok()`
#    (load.rs:767-778) conjoins `gateup_fuse()`, so GATEUP_FUSE=0 alone already
#    forces the plain layout at load time. It is kept for belt-and-braces.
# ⚠️ DSV41_SPEC=1 DSV41_DSPARK=1 are REQUIRED in BOTH arms: the grouped kernel
#    lives in `moe_rows` (verify), and without spec decoding the step takes the
#    single-row `moe()` path — which arms a DIFFERENT unverified kernel (the
#    swapAB `dsv41_expert_tcgen05_gate_up_e4m3`, expectation §1.4 / R7) and
#    would confound the attribution.
#
# STAGES
#   1  SYMBOL PRECHECK (no GPU) — `nm -D` the .so for the FIVE symbols. Any miss
#      => abort with exit 2 and rebuild instructions; the arm would decline
#      silently and the run would measure the OLD path.
#   2  SHORT SMOKE (GPU) — arm serve + "你好" max_tokens=20. Catches: serve does
#      not come up / crashes, the arm did not actually dispatch (the four
#      one-shot decline notices), and a 拉丁/乱码 answer. Keeps the serve up.
#   3  NUMERIC CHECK (GPU) — on the SAME arm serve: 出师表 max_tokens=100.
#      Tear down, then run the BASELINE (the identical env MINUS the four
#      arm gates) and diff the two answers' LEADING tokens.
#
#   PASS = no crash + readable text (no 拉丁 / no adjacent doubles / 非空)
#          + the first TOKEN_MATCH_CHARS chars agree with the baseline
#          (a late divergence is reported; a divergence at the VERY FIRST char
#           is the数值问题 verdict: do NOT ship).
#
# USAGE
#   bash scripts/tcgen05_smoke.sh                 # 3 stages, default node
#   bash scripts/tcgen05_smoke.sh --dry-run       # stage 0+1 only (no GPU)
#   bash scripts/tcgen05_smoke.sh --stages 1      # symbols only
#   NODE=local bash scripts/tcgen05_smoke.sh      # run ON the GPU box
#   LAYOUT_CTRL=1 bash scripts/tcgen05_smoke.sh   # + the extra layout-control
#                                                 # arm (GATEUP_FUSE=0 ILV=0,
#                                                 # no tcgen05) for a tighter
#                                                 # numerical parity (opt-in)
#
# EXIT CODES  0 = PASS · 1 = the kernel FAILED the smoke (crash / 乱码 / 数值)
#             2 = SETUP failure (unreachable node / missing symbol / build)
#
# LOGS  $LOGDIR = /tmp/tcgen05_smoke
#         <tag>.log  <tag>.json  <tag>.txt  <tag>.env  <tag>.metrics
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# ---- knobs -----------------------------------------------------------------
NODE="${NODE:-ubuntu@43.202.208.136}"
RROOT="${RROOT:-ferrite}"
ARCH="${ARCH:-103a}"                                  # B300 = sm_103a (必带)
PORT="${PORT:-8699}"                                  # 8691 belongs to batched_400_v2
TP="${TP:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MODEL_DIR="${DSV41_MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${DSV41_MODEL_NAME:-deepseek-v4.1-flash}"
LOGDIR="${LOGDIR:-/tmp/tcgen05_smoke}"
HEALTH_TRIES="${HEALTH_TRIES:-60}"                    # x5s = 5 min
REQ_TIMEOUT="${REQ_TIMEOUT:-300}"
TEARDOWN_SLEEP="${TEARDOWN_SLEEP:-8}"
TOKEN_MATCH_CHARS="${TOKEN_MATCH_CHARS:-10}"          # ≈ the first 10 tokens (CJK:
                                                      # ~1-2 chars/token; a char
                                                      # proxy, we hold no tokenizer)

PROMPT_SMOKE="${PROMPT_SMOKE:-你好}"
MAXTOK_SMOKE="${MAXTOK_SMOKE:-20}"
PROMPT_NUM="${PROMPT_NUM:-请背诵《出师表》开头。}"
MAXTOK_NUM="${MAXTOK_NUM:-100}"

# The fixed base every arm shares. `DSV41_EXPERT_ACT_E4M3` is part of the
# documented matrix (expectation §7-第1步), NOT an arm delta — so the only
# difference between the baseline and the arm is the four tcgen05/layout gates.
BASE_GATES="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 DSV41_TIMING=1"
ARM_GATES="DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 \
DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0"
CTRL_GATES="DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0"
# The five symbols STAGE 1 gates on. The four `dsv41_route_*`/gemm names must be
# present together (`supports_route_group()` probes the route set as a SET —
# device.rs:3180); `act_e4m3_cap` gates the e4m3 activation itself (device.rs:4814).
SYMS=(dsv41_expert_act_e4m3_cap
      dsv41_expert_gemm_e4m3_grouped
      dsv41_route_group
      dsv41_route_gather_rows
      dsv41_route_scatter_rows)

DRY=0
STAGES="${STAGES:-1 2 3}"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --stages)  shift; STAGES="$1" ;;
        -h|--help) sed -n '2,95p' "$0"; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# ---- transport: remote by default (the GPU box owns the tree + .so); NODE=local
#      runs everything in this shell (= on the GPU box). -----------------------
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
if [ "$NODE" = "local" ] || [ "$NODE" = "-" ] || [ -z "$NODE" ]; then
    REMOTE=0
else
    REMOTE=1
fi
rsh() { # run a shell command on the GPU box (stdin is forwarded: the request
        # body goes over the same channel via curl --data-binary @-)
    if [ "$REMOTE" = 1 ]; then
        ssh "${SSH_OPTS[@]}" "$NODE" "$1"
    else
        bash -c "$1"
    fi
}
SO="$HOME/$RROOT/kernels/cuda/libferrite_kernels.so"
BIN="$HOME/$RROOT/target/release/ferrite-serve"

mkdir -p "$LOGDIR"
pass=0; fail=0; warn_n=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
warn() { echo "  WARN  $*"; warn_n=$((warn_n+1)); }
head1(){ echo; echo "=============================================================="; echo "== $*"; echo "=============================================================="; }
CURL="curl -s --noproxy '*'"

# ---- teardown (只 teardown 自己起的 serve；preflight 已保证起前无人占用) ----
SERVE_PORT=""
cleanup() {
    local p="${SERVE_PORT:-$PORT}"
    rsh "$CURL -m 5 -X POST http://localhost:$p/shutdown >/dev/null 2>&1" || true
    for _ in $(seq 1 10); do
        [ "$(rsh 'pgrep -x ferrite-serve | wc -l' 2>/dev/null)" = 0 ] && break
        rsh "sleep 3" || true
    done
    rsh "pkill -9 -x ferrite-serve 2>/dev/null; sleep $TEARDOWN_SLEEP; true" || true
    SERVE_PORT=""
}
trap cleanup EXIT

echo "== tcgen05 e4m3 GROUPED — smoke test (correctness only) =="
echo "   tree   : $ROOT"
echo "   node   : $([ "$REMOTE" = 1 ] && echo "$NODE" || echo '<local>')   repo: ~/$RROOT"
echo "   model  : $MODEL_DIR   tp=$TP   gpus=$GPU_LIST   port=$PORT"
echo "   stages : $STAGES $([ "$DRY" = 1 ] && echo '(dry-run: stage 0+1 only)')"
echo "   base   : $BASE_GATES"
echo "   arm    : + $ARM_GATES"
[ "${LAYOUT_CTRL:-0}" = 1 ] && echo "   ctrl   : + $CTRL_GATES (opt-in layout control, no tcgen05)"
echo "   prompts: smoke='$PROMPT_SMOKE'($MAXTOK_SMOKE tok)  numeric='$PROMPT_NUM'($MAXTOK_NUM tok)"
echo "   match  : first ${TOKEN_MATCH_CHARS} chars must agree with the baseline"

# ---------------------------------------------------------------- stage 0 ---
head1 "STAGE 0 — preflight (reachable node, pair present, GPU free)"

rsh "true" >/dev/null 2>&1 || { echo "FATAL: cannot reach node '$NODE'" >&2; exit 2; }
ok "node reachable"

for f in "$SO" "$BIN"; do
    if rsh "[ -e $f ]"; then ok "present: $f"; else
        echo "FATAL: missing $f on the node — build first:" >&2
        echo "       ssh $NODE 'cd ~/$RROOT/kernels/cuda && bash build.sh $ARCH' \\" >&2
        echo "         && ssh $NODE 'cd ~/$RROOT && touch crates/ferrite-kernel/build.rs && cargo build --release'" >&2
        exit 2
    fi
done

alive="$(rsh 'pgrep -x ferrite-serve | wc -l')"
if [ "$alive" != 0 ]; then
    warn "$alive ferrite-serve process(es) already running — killing them (runs must never overlap)"
    rsh "pkill -9 -x ferrite-serve 2>/dev/null; sleep $TEARDOWN_SLEEP; true"
    alive="$(rsh 'pgrep -x ferrite-serve | wc -l')"
    [ "$alive" = 0 ] || { echo "FATAL: could not free the box ($alive survivors)" >&2; exit 2; }
fi
ok "no serve running"

if rsh "command -v nvidia-smi >/dev/null 2>&1"; then
    used="$(rsh "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END {print s+0}'")"
    if [ "${used:-0}" -le 2048 ]; then
        ok "GPU memory in use: ${used} MiB (idle)"
    else
        warn "GPU memory in use: ${used} MiB — another job may be resident (numbers could be polluted)"
    fi
else
    warn "nvidia-smi not found on the node — the idle check was skipped"
fi

if [ -n "$(rsh "cd ~/$RROOT && git status --porcelain 2>/dev/null | head -1")" ]; then
    warn "the node's working tree is DIRTY — the .so is only valid for that (uncommitted) source"
fi
rsh "[ -f $HOME/$RROOT/kernels/cuda/.build_id ]" \
    && echo "   build_id: $(rsh "cat $HOME/$RROOT/kernels/cuda/.build_id")"

# ---------------------------------------------------------------- stage 1 ---
head1 "STAGE 1 — symbol precheck (no GPU): the 5 symbols the arm needs"

missing=()
for s in "${SYMS[@]}"; do
    if rsh "nm -D --defined-only $SO 2>/dev/null | grep -qw $s"; then
        ok "$s"
    else
        bad "$s ABSENT"
        missing+=("$s")
    fi
done

if [ "${#missing[@]}" != 0 ]; then
    echo
    echo "SETUP FAIL — ${#missing[@]} symbol(s) missing: ${missing[*]}" >&2
    echo "  The grouped arm CANNOT dispatch without them (it would decline and the run" >&2
    echo "  would measure the OLD per-(row,slot) path — expectation §7-第0步)." >&2
    echo "  Rebuild BOTH products (AGENTS.md 双产物纪律 — the .so alone leaves the pair stale):" >&2
    echo "    ssh $NODE 'cd ~/$RROOT/kernels/cuda && bash build.sh $ARCH'" >&2
    echo "    ssh $NODE 'cd ~/$RROOT && touch crates/ferrite-kernel/build.rs && cargo build --release'" >&2
    echo "  build.sh compiles the e4m3 skeleton in BY DEFAULT" >&2
    echo "  (DSV41_TCGEN05_GATEUP_E4M3_SKELETON; opt out with DSV41_BUILD_TCGEN05_E4M3=0)." >&2
    echo "  No GPU was spent. Re-run this script afterwards." >&2
    exit 2
fi
ok "all 5 symbols present — the arm can dispatch"

if [ "$DRY" = 1 ]; then
    echo
    echo "dry-run: stages remaining = $(echo "$STAGES" | tr ' ' ',' | sed 's/1,\?//') ; nothing was launched (no GPU spent)."
    exit 0
fi

# ---- serve helpers ---------------------------------------------------------
start_serve() { # $1 = tag, $2 = gates string (may be empty)
    local tag="$1" gates="$2" rlog="~/tcgen05_smoke_${tag}.log"
    rsh "pkill -9 -x ferrite-serve 2>/dev/null; sleep $TEARDOWN_SLEEP; true" || true
    rsh "cd ~/$RROOT && nohup env CUDA_VISIBLE_DEVICES=$GPU_LIST \
        LD_LIBRARY_PATH=\$HOME/$RROOT/kernels/cuda \
        DSV41_KERNELS=\$HOME/$RROOT/kernels/cuda/libferrite_kernels.so \
        $gates ./target/release/ferrite-serve --model dsv41 --serve --tp $TP \
        --model-dir $MODEL_DIR --port $PORT > $rlog 2>&1 &" \
        || { echo "FATAL: could not spawn the serve for $tag" >&2; return 2; }
    SERVE_PORT="$PORT"
    if ! rsh "for i in \$(seq 1 $HEALTH_TRIES); do $CURL -m 2 http://localhost:$PORT/health >/dev/null 2>&1 && exit 0; sleep 5; done; exit 1"; then
        echo "FATAL: serve '$tag' never became healthy; last log lines:" >&2
        rsh "tail -20 $rlog" | sed 's/^/    | /' >&2
        rsh "cat $rlog" > "$LOGDIR/$tag.log" 2>/dev/null
        return 2
    fi
    # the env that ACTUALLY runs, straight off /proc — the cheapest proof the
    # gate chain survived the shell (and a leak guard).
    rsh "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort" \
        >"$LOGDIR/$tag.env" 2>/dev/null
    echo "   [$tag] effective DSV41_ env:"; sed 's/^/       /' "$LOGDIR/$tag.env"
    return 0
}

req() { # $1 = tag, $2 = prompt, $3 = max_tokens ; writes $LOGDIR/<tag>.json
    local tag="$1" prompt="$2" mtok="$3" body="$LOGDIR/$tag.body.json"
    python3 - "$MODEL_NAME" "$prompt" "$mtok" >"$body" <<'PY'
import json, sys
model, prompt, mtok = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": mtok, "stream": False, "temperature": 0}))
PY
    rsh "$CURL -m $REQ_TIMEOUT -X POST http://localhost:$PORT/v1/chat/completions \
         -H 'Content-Type: application/json' --data-binary @-" <"$body" >"$LOGDIR/$tag.json" 2>/dev/null
}

fetch_log() { rsh "cat ~/tcgen05_smoke_$1.log" >"$LOGDIR/$1.log" 2>/dev/null; }

# The four one-shot decline notices. ANY of them = the arm did NOT dispatch and
# the step answered on the OLD path (this project's #1 measurement-bias trap).
# They are exact substrings of the eprintln! texts (chain_dev.rs:772/789/865/920).
decline_notes() {
    grep -cE 'DSV41_EXPERT_GROUPED is set, but the routed MoE still runs the proven|DSV41_EXPERT_TCGEN05_E4M3 is set, but the routed MoE still dispatches|DSV41_EXPERT_TCGEN05_E4M3 is set, but the routed gate/up stays on the batched|DSV41_EXPERT_ACT_E4M3 is set, but the routed experts still run the' "$1" 2>/dev/null
}

# text-level report (project red lines: 不能重复、不能乱码 — same level) --------
text_metrics() { # $1 = tag ; writes <tag>.txt + <tag>.metrics
    python3 - "$LOGDIR/$1.json" "$LOGDIR/$1.txt" "$LOGDIR/$1.metrics" <<'PY'
import json, re, sys
resp_p, txt_p, m_p = sys.argv[1:4]
try:
    content = json.load(open(resp_p))["choices"][0]["message"]["content"]
    err = ""
except Exception as exc:  # noqa: BLE001
    content, err = "", str(exc)
open(txt_p, "w").write(content)
ch = list(content)
dbl = sum(1 for i in range(1, len(ch)) if ch[i] == ch[i - 1] and not ch[i].isspace())
latin_idx = [i for i, c in enumerate(ch) if ("a" <= c <= "z" or "A" <= c <= "Z")]
latin = len(latin_idx)
latin_samples = ",".join("".join(ch[max(0, i - 1):i + 2]) for i in latin_idx[:5])
full_of = {"illegal": [], "fault": [], "cuda error": [], "panic": [], "abort": []}
with open(m_p, "w") as fh:
    fh.write("chars=%d\n" % len(content))
    fh.write("dbl=%d\n" % dbl)
    fh.write("latin=%d\n" % latin)
    fh.write("latin_samples=%s\n" % latin_samples)
    fh.write("has_kaishen=%s\n" % ("yes" if "先帝创业未半" in content else "no"))
    fh.write("resp_err=%s\n" % err)
PY
}

metric() { awk -F= -v k="$2" '$1==k{print $2}' "$LOGDIR/$1.metrics" 2>/dev/null; }

# ---------------------------------------------------------------- stage 2 ---
if [[ " $STAGES " == *" 2 "* ]]; then
    head1 "STAGE 2 — short smoke (GPU): arm serve + '$PROMPT_SMOKE' ($MAXTOK_SMOKE tok)"

    start_serve arm "$BASE_GATES $ARM_GATES" || { echo "RESULT: FAIL (stage 2: the serve did not come up)" >&2; exit 2; }
    ok "serve healthy with the arm armed"

    # The arm's own gates MUST be in the running env (read back off /proc).
    for g in DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0; do
        if grep -qx "$g" "$LOGDIR/arm.env"; then ok "env: $g"; else bad "env: $g NOT in the running serve"; fi
    done

    req arm_smoke "$PROMPT_SMOKE" "$MAXTOK_SMOKE"
    fetch_log arm
    text_metrics arm_smoke

    # (a) crash?  NB: `\bfault\b`, NOT the bare substring — the log's own
    #     "default" would otherwise be counted as a fault (false positive).
    crash="$(grep -ciE '\billegal\b|\bfault\b|CUDA error|\bpanic\b|\babort(ed)?\b|out of memory|segmentation' "$LOGDIR/arm.log" 2>/dev/null)"
    if rsh "$CURL -m 5 http://localhost:$PORT/health >/dev/null 2>&1"; then
        ok "serve survived the request (health OK)"
    else
        bad "serve is GONE after the request — crash"
    fi
    [ "${crash:-0}" = 0 ] && ok "no illegal/fault/panic markers in the log" \
                          || bad "$crash crash marker line(s) in the log (see $LOGDIR/arm.log)"

    # (b) did the ARM actually dispatch? any decline notice = it did NOT.
    dn="$(decline_notes "$LOGDIR/arm.log")"
    if [ "${dn:-0}" = 0 ]; then
        ok "no 'armed but declined' notice — the grouped arm dispatched"
    else
        bad "$dn decline notice(s) — the arm did NOT dispatch (the OLD path answered):"
        grep -E '^warning: DSV41_EXPERT_(GROUPED|TCGEN05_E4M3|ACT_E4M3)' "$LOGDIR/arm.log" | sed 's/^/        | /'
    fi

    # (c) readable?
    mchars="$(metric arm_smoke chars)"; mlatin="$(metric arm_smoke latin)"; mdbl="$(metric arm_smoke dbl)"
    echo "   arm smoke text: '$(head -c 80 "$LOGDIR/arm_smoke.txt")'"
    [ "${mchars:-0}" -gt 0 ] && ok "answer non-empty (${mchars} chars)" || bad "EMPTY answer (resp_err=$(metric arm_smoke resp_err))"
    [ "${mlatin:-0}" = 0 ] && ok "no latin char (零拉丁红线)" || bad "$mlatin latin char(s): $(metric arm_smoke latin_samples)"
    [ "${mdbl:-0}" = 0 ]  && ok "no adjacent double-char (重复红线)" || bad "$mdbl adjacent double-char(s)"

    # informational: [gmo]/[phs] tell whether the single-row `moe()` ran in the
    # window (expectation §8.3 — R7: `moe()` would arm the OTHER unverified
    # swapAB e4x kernel under the same DSV41_EXPERT_TCGEN05_E4M3 gate).
    if grep -qE '\[gmo\]|\[phs\]' "$LOGDIR/arm.log"; then
        warn "the log carries [gmo]/[phs] lines — the single-row moe() path also ran in this window (R7:";
        warn "      the swapAB dsv41_expert_tcgen05_gate_up_e4m3 shares this gate; attribute with care)"
    fi
    echo "   (log: $LOGDIR/arm.log)"
fi

# ---------------------------------------------------------------- stage 3 ---
if [[ " $STAGES " == *" 3 "* ]]; then
    head1 "STAGE 3 — numeric check (GPU): '$PROMPT_NUM' ($MAXTOK_NUM tok), arm vs baseline"

    if [[ " $STAGES " != *" 2 "* ]]; then
        start_serve arm "$BASE_GATES $ARM_GATES" || { echo "RESULT: FAIL (stage 3: the arm serve did not come up)" >&2; exit 2; }
    fi
    # arm text on the arm serve (same process as the stage-2 smoke; text is what
    # we read, never the process-level timing accumulators — s2_ab_matrix's
    # one-prompt-per-serve rule is a TIMING rule).
    req arm_num "$PROMPT_NUM" "$MAXTOK_NUM"
    fetch_log arm
    text_metrics arm_num
    cleanup                      # tear the arm down BEFORE the baseline serve

    start_serve base "$BASE_GATES" || { echo "RESULT: FAIL (stage 3: the BASELINE serve did not come up)" >&2; exit 2; }
    # leak guard: the four arm gates must NOT be in the baseline's running env.
    leak="$(grep -cE '^DSV41_(EXPERT_TCGEN05_E4M3|EXPERT_GROUPED)=' "$LOGDIR/base.env")"
    if [ "${leak:-0}" = 0 ]; then ok "baseline env free of the arm gates"; else bad "the ARM gates leaked into the baseline serve"; fi
    grep -E '^DSV41_(GATEUP_FUSE|EXPERT_ILV)=' "$LOGDIR/base.env" && warn "baseline carries GATEUP_FUSE/EXPERT_ILV — it is NOT the production baseline" || true

    req base_num "$PROMPT_NUM" "$MAXTOK_NUM"
    fetch_log base
    text_metrics base_num
    cleanup

    # optional third arm: the layout-control (GATEUP_FUSE=0 + ILV=0, NO tcgen05).
    # It is the TIGHTEST parity partner for the arm — base vs arm differ by BOTH
    # the kernel and the layout tax (expectation §7-第2步), so when the arm
    # diverges early this arm separates "kernel wrong" from "layout tax".
    if [ "${LAYOUT_CTRL:-0}" = 1 ]; then
        start_serve ctrl "$BASE_GATES $CTRL_GATES" || { echo "RESULT: FAIL (stage 3: the layout-control serve did not come up)" >&2; exit 2; }
        req ctrl_num "$PROMPT_NUM" "$MAXTOK_NUM"
        fetch_log ctrl
        text_metrics ctrl_num
        cleanup
    fi

    echo
    echo "   baseline : '$(head -c 80 "$LOGDIR/base_num.txt")'"
    echo "   arm      : '$(head -c 80 "$LOGDIR/arm_num.txt")'"
    for t in base_num arm_num ctrl_num; do
        [ -f "$LOGDIR/$t.metrics" ] || continue
        printf "   %-9s chars=%s dbl=%s latin=%s has_kaishen=%s\n" "$t" \
            "$(metric "$t" chars)" "$(metric "$t" dbl)" "$(metric "$t" latin)" "$(metric "$t" has_kaishen)"
    done

    if [ ! -s "$LOGDIR/base_num.txt" ] || [ ! -s "$LOGDIR/arm_num.txt" ]; then
        bad "one of the two answers is EMPTY — cannot compare (resp_err: base=$(metric base_num resp_err) arm=$(metric arm_num resp_err))"
    else
        python3 - "$LOGDIR/base_num.txt" "$LOGDIR/arm_num.txt" "$TOKEN_MATCH_CHARS" \
                 "$LOGDIR/arm_num.metrics" <<'PY'
import sys
base = open(sys.argv[1]).read()
arm = open(sys.argv[2]).read()
need = int(sys.argv[3])
m_p = sys.argv[4]
n = min(len(base), len(arm))
i = 0
while i < n and base[i] == arm[i]:
    i += 1
same = base == arm
print("   leading-char agreement: %d/%d  (identical whole answer: %s)" % (i, n, same))
print("   base[%d]:%r" % (i, base[max(0, i - 6):i + 8]))
print("   arm [%d]:%r" % (i, arm[max(0, i - 6):i + 8]))
open(m_p, "a").write("prefix_chars=%d\n" % i)
open(m_p, "a").write("compare=%s\n" % ("identical" if same else "differ"))
if i >= need:
    print("   VERDICT: PASS — the first %d chars agree (>= %d). A later divergence is the" % (i, need))
    print("            expected MMA-vs-SIMT f32 rounding (expectation §4.2-1); not a red line.")
elif i > 0:
    print("   VERDICT: WARN — the answers agree on %d chars then diverge (< %d). Inspect the" % (i, need))
    print("            two texts above by eye before believing the kernel.")
else:
    print("   VERDICT: FAIL — the FIRST char already differs: the kernel produces a DIFFERENT")
    print("            answer from the very first token. Do NOT ship (expectation §4.2-2/-3:")
    print("            the dense idesc format code / the fp4 packed-vs-unpacked guess).")
PY
        pfx="$(metric arm_num prefix_chars)"
        if [ "${pfx:-0}" -ge "$TOKEN_MATCH_CHARS" ]; then
            ok "numerics: the first ${pfx} chars match the baseline"
        elif [ "${pfx:-0}" -gt 0 ]; then
            warn "numerics: only the first ${pfx} chars match the baseline (< ${TOKEN_MATCH_CHARS})"
        else
            bad "numerics: the first char ALREADY differs — wrong values, do NOT ship"
        fi
    fi
fi

# ------------------------------------------------------------------ verdict ---
head1 "RESULT"
echo "   PASS=$pass  FAIL=$fail  WARN=$warn_n   logs: $LOGDIR"
if [ "$fail" = 0 ]; then
    echo "   RESULT: PASS — no crash, readable output, the arm dispatched, leading"
    echo "           tokens agree with the baseline. This clears the SMOKE gate only:"
    echo "           it is NOT a performance verdict (expectation §5: the kernel is a"
    echo "           correctness-first skeleton, M=128 fixed => ~120x tensor-core"
    echo "           overcompute). Next: the batched_400_v2 A/B for ms."
    exit 0
else
    echo "   RESULT: FAIL — see the FAIL lines above. Binary-search in THIS order"
    echo "           (expectation §7-第4步): DSV41_EXPERT_GROUPED=0 (keeps the e4m3"
    echo "           activation) then DSV41_EXPERT_TCGEN05_E4M3=0 — do NOT turn off"
    echo "           DSV41_EXPERT_ACT_E4M3 first (it is the activation format; a second"
    echo "           variable). Logs under $LOGDIR."
    exit 1
fi
