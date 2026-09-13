#!/bin/bash
# The verify-value locator. Per AGENTS.md, DSV41_DIFF_EAGER=1 replays emitted.len() single-row forwards
# with the same prefix KV and reports the FIRST mismatching index and its absolute position — the
# project's own "定位利器" for exactly the fault the sids_writeback() comment calls the blocker
# ("the write-back is mathematically right; the blocker is verify's values"). Run under DSV41_SPEC=1
# so the measured path is the MTP one (eager and verify are separate implementations).
#
# Judged on: the first [diff] mismatch index (if any) plus the 1..100 text's first-N-correct count.
set -uo pipefail
cd "$HOME/ferrite"
bash "$HOME/ensure_built.sh" || { echo "[runner] build gate FAILED"; exit 1; }
NAME=VDIFF
bash "$HOME/arm_run.sh" "$NAME" \
  DSV41_SPEC=1 DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 \
  DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 \
  DSV41_DIFF_EAGER=1 DSV41_TOKTRACE=1 \
  2>&1 | tee "$HOME/armrun_${NAME}.txt" | grep -aE "OUT:|WATCHDOG|SERVE_FAILED" | head -3
echo "=== the [diff] lines (the locator's output) ==="
grep -a "\[diff\]" "$HOME/armrun_${NAME}.log" | head -12
echo "=== the emitted token ids (DSV41_TOKTRACE) ==="
grep -a "\[toktr\]" "$HOME/armrun_${NAME}.log" | head -12
echo "=== text ==="
python3 - "$HOME/armrun_${NAME}.log" <<'PY'
import re, sys
txt = open(sys.argv[1], errors='ignore').read()
m = re.findall(r"\] OUT: (.*)", txt)
if m:
    toks = [x for x in m[-1].strip("'\"").replace("\\n", "\n").split("\n") if x.strip()]
    ok = 0
    for i, tk in enumerate(toks[:61]):
        if tk.strip() == str(i+1): ok += 1
        else: break
    print(f"{len(toks)} lines; first {ok} are exactly 1..N; head={toks[:20]}")
else:
    print("no OUT line")
PY
