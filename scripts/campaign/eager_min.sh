#!/bin/bash
# The minimal EAGER arm — the user's own suggestion, and the fastest way to place the 44/77/1010 defect.
#
# What we know: the reply was `1,2,3,44,5,6,77,8,9,1010` and the DIFF_EAGER probe showed
# first_mismatch=none, i.e. the m-row verify and the single-row engine agree with EACH OTHER while both
# are wrong. AGENTS.md records the reference: "数字任务…EAGER 对照完美 1..100". So either the shared
# forward has regressed, or one of the flags the arms carry is responsible.
#
# This arm strips everything the arms normally add and keeps only what the model needs to run:
#   * NO spec (DSV41_SPEC unset => the plain single-row decode path),
#   * the hand-written BS arm off (it destroys the model),
#   * no graphs (GRAPH_OFF is the runner default here, which is what the reference used too),
#   * and the runner's COMMON is overridden down to the minimum.
# If the text is perfect, a flag is the culprit; if it is still doubled, the shared forward is broken.
set -uo pipefail
cd "$HOME/ferrite"
bash "$HOME/ensure_built.sh" || { echo "[runner] build gate FAILED"; exit 1; }
NAME=EAGER_MIN
bash "$HOME/arm_run.sh" "$NAME" \
  DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 \
  DSV41_SPEC=0 DSV41_DSPARK=0 \
  DSV41_EXPERT_ACT_E4M3=0 DSV41_BF16_TRUNCATE=0 DSV41_EXPERT_ILV=0 \
  DSV41_GRAPH_STEP=0 DSV41_VERIFY_GRAPH=0 \
  2>&1 | tee "$HOME/armrun_${NAME}.txt" | grep -aE "OUT:|WATCHDOG" | head -3
echo "=== the byte-exact reply ==="
[ -f "$HOME/num100_last.txt" ] && cat -A "$HOME/num100_last.txt" | head -20
echo "=== step p50 (this is the DECODE reference; ~6.3 ms is the documented eager figure) ==="
python3 - "$HOME/armrun_${NAME}.log" <<'PY'
import re, statistics, sys
t = open(sys.argv[1], errors='ignore').read()
v = [float(x) for x in re.findall(r"\[dsv41\] step pos=\d+: ([0-9.]+)ms", t)]
if v:
    s = sorted(v[-200:])
    print(f"n={len(s)} p50={statistics.median(s):.2f}ms p10={s[len(s)//10]:.2f}")
PY
