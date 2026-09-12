#!/usr/bin/env bash
# =============================================================================
# triple_ab.sh — the THREE-IN-ONE A/B for the e4m3 two-pass fix.
#
# WHY (the three questions this one command answers)
#   Three runtime gates, all default OFF, are verified together because they land
#   on the SAME measurement (the DSpark verify block of DeepSeek-V4.1):
#
#     * DSV41_EXPERT_ACT_E4M3  the e2m1x2 two-pass expert activation (the
#         "e4m3-precision" arm). Its KERNEL half was just re-bound to the caller's
#         `out_slot_stride` (the layout-contract fix: the kernel's `fuse` can no
#         longer be derived independently of the Rust `two` flag), and its RUST
#         mirror gained the missing `dim%512==0` condition. The .so therefore
#         MUST be rebuilt (the last bug was exactly "gate ON, old .so, no effect").
#         Q1: does the 出师表 text come back NORMAL (not the "6.6.6.6" counting
#             degeneration) and does `opa` stay eliminated?
#
#     * DSV41_SH_EXP_MROWS     the shared expert as ONE multi-row pass.
#         Q2: does verify_ms drop (plan expectation -8.3 ms)?
#
#     * DSV41_VERIFY_GRAPH     the DSpark verify block captured into a CUDA graph.
#         Q3: does verify_ms drop again (a graph removes the launch-submit half)?
#
# THE FOUR ARMS (strictly cumulative — each arm adds exactly ONE gate)
#   1  base            all three OFF            (the yardstick)
#   2  +E4M3           EXPERT_ACT_E4M3=1
#   3  +E4M3+SH        + SH_EXP_MROWS=1
#   4  +E4M3+SH+GRAPH  + VERIFY_GRAPH=1
#   Cumulative ON PURPOSE: Δ(A2→A3) is the mrows gain and Δ(A3→A4) is the graph
#   gain, each read against the arm the plan expects. The doc's "A before B"
#   ordering matters here: the graph only removes what mrows left, so measuring
#   the graph against the mrows arm (not against base) is the only fair number.
#
# WHY A BUILD STEP, AND WHY IT IS CALLED EVERY ARM
#   All three gates are RUNTIME env (a `OnceLock` read once per process), so ONE
#   pair (kernel .so + binary) is valid for all four arms; the kernel fix is
#   common to all of them. `ensure_pair` is nonetheless called before EVERY arm
#   (the single-source discipline) and rebuilds ONLY when something is actually
#   stale: the .so is missing, a *.cu is newer than the .so, the pair's build-id
#   does not embed, or TRIPLE_REBUILD_EACH is set. So the loop pays for nvcc at
#   most once and never launches a serve onto a mismatched pair.
#
# WHY EACH ARM RUNS TWO SERVES (eager + spec)
#   `DSV41_SPEC` is a READ-ONCE PROCESS gate (crates/ferrite-dsv41/src/serve.rs
#   spec_mode(), a OnceLock), so EAGER (plain single-row decode) and SPEC
#   (draft + verify commit) CANNOT share one serve. Each arm therefore runs:
#
#     eager leg — plain decode, 出师表 200 tok -> the TEXT judgment (no dspark)
#     spec  leg — DSV41_SPEC=1, 出师表 300 tok -> k_acc + verify_ms + captured + text
#
#   The "single serve discipline" (never two ferrite-serve at once) is enforced
#   by teardown() between every leg. LEGS="eager,spec" (default) runs both;
#   LEGS=spec runs only the speed legs.
#
# THE JUDGED FIELDS (exactly the task's list)
#   opa        — does the decoded text contain the literal "opa"? (True/False)
#   first      — the head of the decoded text; 《 / 开篇 / degen are its flags
#   verify=    — the LAST `[dspark] steps=…` line's per-step verify_ms
#   captured   — did the spec log print `[verify_graph] captured …`?
#
# USAGE
#   bash scripts/triple_ab.sh                       # all four arms, both legs
#   nohup bash scripts/triple_ab.sh > /tmp/triple_ab/run.log 2>&1 &
#   LEGS=spec        bash scripts/triple_ab.sh      # speed legs only
#   TRIPLE_REBUILD_EACH=1 bash scripts/triple_ab.sh # force nvcc before each arm
#   PORT_BASE=8300   bash scripts/triple_ab.sh      # move the ports (8 serves)
#   TRIPLE_SKIP_BUILD=1 bash scripts/triple_ab.sh   # trust a prebuilt pair
#
# OUTPUT
#   one row per (arm, leg), then the 判据 block. Artifacts in $LOGDIR (default
#   /tmp/triple_ab): <arm>_<leg>.{log,log.kept,dspark,vg,resp.json,env} plus
#   report.py and table.txt (the tee'd report).
#   Exit: 0 = all three gates hold, 1 = a gate FAILED, 2 = a case was
#   UNMEASURED / harness error (NOT evidence about the gates).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
K="$ROOT/kernels/cuda"
SO="$K/libferrite_kernels.so"
BIN="$ROOT/target/release/ferrite-serve"
ARCH="${ARCH:-103a}"                       # B300 = sm_103a (build.sh default 100a is stale)

REPO="${REPO:-$ROOT}"
LOGDIR="${TRIPLE_LOGDIR:-/tmp/triple_ab}"
MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4.1-flash}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
PORT_BASE="${PORT_BASE:-8195}"
READY_TRIES="${READY_TRIES:-90}"           # x 6 s = up to 9 min to become healthy
LEGS="${LEGS:-eager,spec}"
SKIP_BUILD="${TRIPLE_SKIP_BUILD:-}"
CURL="curl -s --noproxy '*'"
GATE_MS="${GATE_MS:-2.0}"                  # the task's per-step gate for each Δ
EXP_MROWS_MS="${EXP_MROWS_MS:-8.3}"        # the plan's expectation for Δ(A2→A3)

# The project yardstick prompt: it pins the canonical opening, so the first-token
# judgment has a fixed target (先帝创业未半 / 臣亮言).
P_SH="${P_SH:-请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。}"
MAXTOK_EAGER="${MAXTOK_EAGER:-200}"
MAXTOK_SPEC="${MAXTOK_SPEC:-300}"

mkdir -p "$LOGDIR"

ARM_TAGS=(A1_base A2_e4m3 A3_shexp A4_graph)
ARM_GATES=(
    ""
    "DSV41_EXPERT_ACT_E4M3=1"
    "DSV41_EXPERT_ACT_E4M3=1 DSV41_SH_EXP_MROWS=1"
    "DSV41_EXPERT_ACT_E4M3=1 DSV41_SH_EXP_MROWS=1 DSV41_VERIFY_GRAPH=1"
)

echo "== triple_ab (e4m3 fix + SH_EXP_MROWS + VERIFY_GRAPH) =="
echo "-- repo $ROOT  rev $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "-- arms ${ARM_TAGS[*]}   legs $LEGS   ports $PORT_BASE..   tp $TP"
echo "-- eager: max_tokens=$MAXTOK_EAGER   spec: max_tokens=$MAXTOK_SPEC"
echo "-- logs $LOGDIR   (gate ${GATE_MS}ms/leg; mrows expectation -${EXP_MROWS_MS}ms)"

# ---------------------------------------------------------------------------
# Build / pair management. The one working order (see scripts/dsv41_serve_ab.sh):
#   build.sh (rewrites .so + .build_id) -> touch build.rs (force the stamp rerun)
#   -> cargo build --release (bakes the fresh id into the binary)
# `cargo build` ALONE can never heal a mismatch: cargo is incremental and will
# happily relink the OLD FERRITE_BUILD_ID, so the touch is mandatory.
# ---------------------------------------------------------------------------
pair_ok() {
    [ -f "$SO" ] && [ -f "$K/.build_id" ] && [ -x "$BIN" ] || return 1
    command -v strings >/dev/null 2>&1 || return 0   # cannot verify -> assume OK
    # grep -c (not -q): -q exits early -> SIGPIPE on strings -> pipefail misfires.
    strings "$BIN" | grep -cF -- "$(cat "$K/.build_id")" >/dev/null
}

ensure_pair() {
    local need=0
    [ -f "$SO" ] || need=1
    [ -f "$K/.build_id" ] || need=1
    [ -x "$BIN" ] || need=1
    if [ -f "$SO" ] && [ -n "$(find "$K" -maxdepth 1 -name '*.cu' -newer "$SO" -print -quit 2>/dev/null)" ]; then
        need=1
    fi
    [ -n "${TRIPLE_REBUILD_EACH:-}" ] && need=1

    if [ "$need" = 1 ]; then
        echo "-- build: kernels/cuda/build.sh $ARCH  (kernel .cu changed / .so stale / pair missing${TRIPLE_REBUILD_EACH:+ / forced})"
        command -v nvcc >/dev/null 2>&1 || { echo "FATAL: nvcc not found — the .so must be rebuilt (toolkit needed, no GPU)"; return 2; }
        ( cd "$K" && bash build.sh "$ARCH" ) || { echo "FATAL: kernels/cuda/build.sh $ARCH failed"; return 2; }
        touch "$ROOT/crates/ferrite-kernel/build.rs"
        ( cd "$ROOT" && cargo build --release ) || { echo "FATAL: cargo build --release failed"; return 2; }
    else
        echo "-- build: pair fresh (build_id $(cut -c1-24 "$K/.build_id" 2>/dev/null)…), nvcc skipped"
        # Cheap incremental; keeps a hand-edited Rust change from being missed.
        ( cd "$ROOT" && cargo build --release ) >/dev/null 2>&1 || { echo "FATAL: cargo build --release failed"; return 2; }
    fi

    if ! pair_ok; then
        echo "FATAL: .so and binary are NOT same-source."
        echo "       .build_id: $(cat "$K/.build_id" 2>/dev/null || echo '<missing>')"
        echo "       rebuild: (cd kernels/cuda && bash build.sh $ARCH) && touch crates/ferrite-kernel/build.rs && cargo build --release"
        return 2
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Single-serve discipline.
# ---------------------------------------------------------------------------
CURRENT_PORT=""
teardown() {  # port
    local port="${1:-}"
    [ -n "$port" ] && $CURL -m 5 -X POST "http://localhost:$port/shutdown" >/dev/null 2>&1
    local _
    for _ in $(seq 1 10); do
        [ "$(pgrep -x ferrite-serve | wc -l)" = 0 ] && break
        sleep 3
    done
    pkill -9 -x ferrite-serve 2>/dev/null
    CURRENT_PORT=""
    sleep 8
}
trap 'teardown "$CURRENT_PORT"' INT TERM

launch_and_wait() {  # log port "EXTRA_ENV ..."
    local log="$1" port="$2" envs="$3"
    pkill -9 -x ferrite-serve 2>/dev/null
    sleep 6
    # shellcheck disable=SC2086 # $envs is a deliberate space-separated VAR=VALUE list
    ( cd "$REPO" && nohup env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
        LD_LIBRARY_PATH="$K" \
        DSV41_KERNELS="$SO" \
        DSV41_TIMING=1 \
        $envs \
        timeout 3600 "$BIN" --model dsv41 --serve --tp "$TP" \
        --model-dir "$MODEL_DIR" --port "$port" > "$log" 2>&1 & )
    CURRENT_PORT="$port"

    local i
    for i in $(seq 1 "$READY_TRIES"); do
        if $CURL -m 2 "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "   ready after ~$((i * 6))s"
            return 0
        fi
        if grep -qiE 'build.?id mismatch' "$log" 2>/dev/null; then
            echo "   FATAL: build-id mismatch — refusing to measure a mixed pair"
            tail -5 "$log" | sed 's/^/   | /'
            return 2
        fi
        pgrep -x ferrite-serve >/dev/null 2>&1 || {
            echo "   FATAL: serve died before it was ready"
            tail -20 "$log" | sed 's/^/   | /'
            return 2
        }
        sleep 6
    done
    echo "   FATAL: serve not healthy after $((READY_TRIES * 6))s"
    tail -8 "$log" | sed 's/^/   | /'
    return 1
}

bench() {  # tag port prompt max_tok
    local tag="$1" port="$2" prompt="$3" max_tok="$4"
    local body t0 t1
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' \
            "$MODEL_NAME" "$prompt" "$max_tok")"
    t0=$(date +%s)
    $CURL -m 900 "http://localhost:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" > "$LOGDIR/$tag.resp.json" 2>/dev/null
    t1=$(date +%s)
    echo "   e2e $((t1 - t0))s -> $LOGDIR/$tag.resp.json"
}

# ---------------------------------------------------------------------------
# The arms, strictly serial.
# ---------------------------------------------------------------------------
port="$PORT_BASE"
for ai in "${!ARM_TAGS[@]}"; do
    arm_tag="${ARM_TAGS[$ai]}"
    gates="${ARM_GATES[$ai]}"

    if [ -z "$SKIP_BUILD" ]; then
        ensure_pair || { echo "FATAL: build/pair step failed before arm $arm_tag"; exit 2; }
    fi

    for leg in ${LEGS//,/ }; do
        case "$leg" in
            eager) extra="";                          max_tok="$MAXTOK_EAGER" ;;
            spec)  extra="DSV41_SPEC=1 DSV41_DSPARK=1"; max_tok="$MAXTOK_SPEC" ;;
            *) echo "FATAL: unknown leg '$leg' (LEGS=eager,spec)"; exit 2 ;;
        esac

        tag="${arm_tag}_${leg}"
        log="$LOGDIR/$tag.log"
        echo
        echo "==== $tag  port=$port  gates=[${gates:-<none>}]  extra=[${extra:-<none>}]  max_tokens=$max_tok"

        if ! launch_and_wait "$log" "$port" "$gates $extra"; then
            teardown "$port"
            echo "   arm $tag ABORTED (see $log)"
            port=$((port + 1))
            continue
        fi

        {
            echo "tag=$tag"
            echo "arm=$arm_tag"
            echo "leg=$leg"
            echo "gates=$gates"
            echo "extra=$extra"
            echo "port=$port"
            echo "max_tokens=$max_tok"
        } > "$LOGDIR/$tag.env"

        bench "$tag" "$port" "$P_SH" "$max_tok"
        grep 'dspark] steps' "$log" | tail -1 > "$LOGDIR/$tag.dspark" 2>/dev/null || true
        grep -m3 'verify_graph' "$log" > "$LOGDIR/$tag.vg" 2>/dev/null || true
        cp "$log" "$LOGDIR/$tag.log.kept" 2>/dev/null || true

        [ -s "$LOGDIR/$tag.dspark" ] && sed 's/^/   | /' "$LOGDIR/$tag.dspark"
        [ -s "$LOGDIR/$tag.vg" ] && sed 's/^/   | /' "$LOGDIR/$tag.vg"
        echo "   faults: $(grep -cE 'illegal|fault' "$log" 2>/dev/null || echo 0)   log: $log"

        teardown "$port"
        port=$((port + 1))
    done
done

# ---------------------------------------------------------------------------
# Report (embedded — the script is a single deliverable).
# ---------------------------------------------------------------------------
cat > "$LOGDIR/report.py" <<'PY'
#!/usr/bin/env python3
"""triple_ab report — the per-case table + the three 判据."""
import hashlib
import json
import os
import re
import sys

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/triple_ab"
GATE_MS = float(os.environ.get("GATE_MS", "2.0"))
EXP_MROWS = float(os.environ.get("EXP_MROWS_MS", "8.3"))

ARMS = [("A1_base", "base"), ("A2_e4m3", "+E4M3"),
        ("A3_shexp", "+E4M3+SH"), ("A4_graph", "+E4M3+SH+GRAPH")]
LEGS = ["eager", "spec"]


def rd(path, binary=False):
    try:
        return open(path, "rb" if binary else "r",
                    errors=None if binary else "ignore").read()
    except OSError:
        return b"" if binary else ""


def field(line, key):
    i = line.find(key)
    if i < 0:
        return None
    m = re.match(r"[-+0-9.eE]+", line[i + len(key):])
    return float(m.group(0)) if m else None


def analyse(tag):
    dspark = rd(os.path.join(LOGDIR, tag + ".dspark")).strip()
    log = (rd(os.path.join(LOGDIR, tag + ".log"))
           or rd(os.path.join(LOGDIR, tag + ".log.kept")))
    try:
        content = json.loads(rd(os.path.join(LOGDIR, tag + ".resp.json")))["choices"][0]["message"]["content"]
        resp_ok = True
    except Exception:  # noqa: BLE001
        content, resp_ok = "", False
    ch = list(content)
    dbl = sum(1 for i in range(1, len(ch)) if ch[i] == ch[i - 1] and not ch[i].isspace())
    return dict(
        dspark=dspark, content=content, resp_ok=resp_ok,
        v=field(dspark, "verify="), draft=field(dspark, "draft="),
        k=field(dspark, "mean-k="), t=field(dspark, "tok/step="), s=field(dspark, "steps="),
        captured="[verify_graph] captured" in log,
        vgfail="[verify_graph] capture FAILED" in log,
        faults=len(re.findall(r"illegal|fault", log)),
        opa=("opa" in content),
        has_lqb=("《" in content[:16]),
        opening=("先帝创业未半" in content[:48]) or ("臣亮言" in content[:24]),
        degen=bool(content) and (content[0].isdigit() or "6.6.6.6" in content),
        first=content[:12], dbl=dbl, chars=len(content),
        md5=hashlib.md5(content.encode()).hexdigest()[:8] if content else "NA",
    )


rows = {(a, l): analyse("%s_%s" % (a, l)) for a, _ in ARMS for l in LEGS}

# ---- per-case table --------------------------------------------------------
print("== per-case ==")
print("%-8s %-16s %9s %8s %8s %8s %6s %-9s %-5s %-14s %-5s %5s %6s %s" % (
    "LEG", "ARM", "verify_ms", "draft_ms", "mean-k", "tok/step", "steps",
    "CAPTURED", "opa", "first", "《", "dbl", "chars", "md5"))
for leg in LEGS:
    for atag, alabel in ARMS:
        r = rows[(atag, leg)]
        cap = "captured" if r["captured"] else "failed" if r["vgfail"] else "no"
        print("%-8s %-16s %9s %8s %8s %8s %6s %-9s %-5s %-14s %-5s %5d %6d %s" % (
            leg, alabel,
            "NA" if r["v"] is None else "%.2f" % r["v"],
            "NA" if r["draft"] is None else "%.2f" % r["draft"],
            "NA" if r["k"] is None else "%.3f" % r["k"],
            "NA" if r["t"] is None else "%.3f" % r["t"],
            "NA" if r["s"] is None else "%d" % int(r["s"]),
            cap, str(r["opa"]), repr(r["first"]), str(r["has_lqb"]),
            r["dbl"], r["chars"], r["md5"]))

print()
print("== 判据 ==")
rc = 0

# (1) TEXT — the eager leg is the text gate: an e4m3 arm must NOT degenerate and
#     must have dropped `opa`; A1 (base) is the known-opa yardstick.
bad = []
for atag, alabel in ARMS:
    r = rows[(atag, "eager")]
    if not r["resp_ok"]:
        print("  TEXT   %-16s eager UNMEASURABLE (no/failed response)" % alabel)
        rc = 2
        continue
    e4 = atag != "A1_base"
    mark = " <-- e4m3 arm" if e4 else " (base yardstick: opa expected)"
    print("  TEXT   %-16s eager opa=%-5s degen=%-5s 《=%-5s 开篇=%-5s dbl=%-3d chars=%-4d md5=%s%s"
          % (alabel, r["opa"], r["degen"], r["has_lqb"], r["opening"],
             r["dbl"], r["chars"], r["md5"], mark))
    if e4 and (r["degen"] or r["opa"]):
        bad.append(alabel)
if bad:
    print("  TEXT   FAIL: e4m3 arm(s) still degenerate / still carry opa: %s" % ", ".join(bad))
    rc = max(rc, 1)
else:
    print("  TEXT   OK: every e4m3 arm's eager 出师表 is normal (no 6.6.6.6, opa cleared)")

# (2) verify_ms chain — each Δ read against the arm the plan names.
print("  -- verify_ms 链 (spec leg) --")
for a_from, a_to, label, exp in (
        ("A2_e4m3", "A3_shexp", "SH_EXP_MROWS", "-%.1f ms" % EXP_MROWS),
        ("A3_shexp", "A4_graph", "VERIFY_GRAPH", "the launch-submit half")):
    rf, rt = rows[(a_from, "spec")], rows[(a_to, "spec")]
    if rf["v"] is None or rt["v"] is None:
        miss = [n for n, r in ((a_from, rf), (a_to, rt)) if r["v"] is None]
        print("  SPEED  %-14s UNMEASURED (%s printed no '[dspark] steps=' line — the 50-step"
              " print gate was not reached; no verdict from this case)"
              % (label, ", ".join(miss)))
        rc = 2
        continue
    dv = rt["v"] - rf["v"]
    ok = -dv >= GATE_MS
    print("  SPEED  %-14s verify %.2f -> %.2f ms  (Δ%+.2f, gain %+.2f ms; 任务 gate >= %.1f ms %s; 预期 %s)"
          % (label, rf["v"], rt["v"], dv, -dv, GATE_MS, "PASS" if ok else "FAIL", exp))
    if not ok:
        rc = max(rc, 1)

# (3) CAPTURE engagement — the graph arm must print `captured`, the rest must not.
cap_ok = True
for atag, alabel in ARMS:
    r = rows[(atag, "spec")]
    want = (atag == "A4_graph")
    got = r["captured"]
    if r["vgfail"]:
        print("  ENGAGE %-16s captured=FAILED (capture_end refused — see the log)" % alabel)
        cap_ok = False
    elif want and not got:
        print("  ENGAGE %-16s captured=no   -> this arm is NOT evidence about the graph" % alabel)
        cap_ok = False
    elif not want and got:
        print("  ENGAGE %-16s captured=yes  -> env not honoured (VERIFY_GRAPH leaked ON)" % alabel)
        cap_ok = False
if cap_ok:
    print("  ENGAGE OK: A4_graph printed 'captured'; A1..A3 did not.")

print()
print("verdict: rc=%d  (0 = all three gates hold, 1 = a gate failed, 2 = a case was unmeasurable)" % rc)
sys.exit(rc)
PY

python3 "$LOGDIR/report.py" "$LOGDIR" | tee "$LOGDIR/table.txt"
rc="${PIPESTATUS[0]}"
echo
echo "triple_ab exit=$rc   logs: $LOGDIR/<arm>_<leg>.{log,log.kept,dspark,vg,resp.json,env}   table: $LOGDIR/table.txt"
exit "$rc"
