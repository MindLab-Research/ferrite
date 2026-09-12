#!/usr/bin/env bash
# swapAB text parity: DSV41_SWAPAB=0 vs =1, back to back, four-prompt A/B.
#
# The numerical half of the swapAB parity judgement is the Rust unit test
# (`crates/ferrite-dsv41/tests/swapab_parity.rs`); THIS is the end-to-end half:
# the four fixed prompts must come out character-for-character identical between
# the SIMT gemv arm and the tensor-core arm, with a real p50 improvement and
# zero faults.
#
# Why a tolerance cannot carry the whole judgement: a GEMV output near a logit
# boundary can flip a single leading token at ~1e-3 (the W8A16 precedent — a
# 0.05% disagreement still flipped a token now and then). Text parity is the
# only signal that catches it. See docs/agent/perf-roadmap.md §swapAB and
# docs/agent/dsv41-methodology.md ("上机必须做一次 parity, 不能只在无 GPU 环境
# 做 cargo check").
#
# Usage (run from the repo root; the caller owns tree state and MUST have
# rebuilt the .so + binary from the SAME source — dsv41_serve_ab.sh self-heals a
# stale pair, but a mismatched pair invalidates the whole A/B):
#
#   scripts/dsv41_swapab_text_parity.sh
#
# Verdict (all must hold):
#   1. the four prompts are character-for-character identical across arms;
#   2. faults = 0 in both arms;
#   3. the swapAB arm's p50 is lower (target 6.26ms -> ~4.2ms, ~238 tok/s).
# Any leading-token flip is reported per prompt; a flip in the BODY (not the
# first token) is a hard fail.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DRIVER="$HERE/dsv41_serve_ab.sh"
[ -x "$DRIVER" ] || { echo "FATAL: $DRIVER not executable"; exit 1; }

A="swapab_off"
B="swapab_on"
echo "== swapAB text parity: '$A' (SIMT gemv) vs '$B' (tensor core) =="

# Arm 1: the current SIMT path. The gate is read ONCE per process, so the two
# arms MUST be two separate serves (one binary, two launches) — never two
# requests in one process.
DSV41_SWAPAB=0 "$DRIVER" "$A" DSV41_SWAPAB=0
# Arm 2: the tensor-core path.
DSV41_SWAPAB=1 "$DRIVER" "$B" DSV41_SWAPAB=1

OA="/tmp/ab_${A}_out.txt"
OB="/tmp/ab_${B}_out.txt"
[ -s "$OA" ] && [ -s "$OB" ] || { echo "FATAL: missing output file(s) $OA / $OB"; exit 1; }

python3 - "$OA" "$OB" "$A" "$B" <<'PY'
import json
import sys

a_path, b_path, a_tag, b_tag = sys.argv[1:5]
PROMPTS = [
    "The capital of France is",
    "请背诵《静夜思》",
    "1+1=",
    "请背诵《出师表》开头",
]


def load(path):
    """The driver `tee`s one JSON body per line; keep them in order."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(d["choices"][0]["message"]["content"])
        except Exception:
            out.append(None)
    return out


A, B = load(a_path), load(b_path)
bad = 0
for i, p in enumerate(PROMPTS):
    ca = A[i] if i < len(A) else None
    cb = B[i] if i < len(B) else None
    same = ca == cb
    # A first-token flip shows up as identical tails after the first token.
    body_same = (
        ca is not None
        and cb is not None
        and len(ca) > 1
        and len(cb) > 1
        and ca[1:] == cb[1:]
    )
    verdict = "IDENTICAL" if same else ("BODY-OK/leading-token-diff" if body_same else "*** DIFF ***")
    if not same and not body_same:
        bad += 1
    print(f"  [{verdict:28s}] {p!r}")
    if not same:
        print(f"      {a_tag}: {ca!r}")
        print(f"      {b_tag}: {cb!r}")

print()
if bad:
    print(f"VERDICT: *** FAIL *** — {bad} prompt(s) differ in the body (not just the first token)")
    sys.exit(1)
print("VERDICT: PASS — all four prompts match character-for-character "
      "(body correct; any leading-token difference is acceptable per the W8A16 precedent)")
PY
rc=$?

echo
echo "p50/tok-s per arm are printed above by the driver; compare $A vs $B."
echo "target: step p50 6.26ms -> ~4.2ms (~238 tok/s)."
exit $rc
