#!/bin/bash
# add_precision_envs.sh [--apply] — add the five promoted precision gates to the shipping
# scripts (push400_hw_test.sh, verify_correct.sh). Idempotent: skips a gate already present.
# Default is DRY-RUN (prints the edit); pass --apply to write, with a timestamped backup.
#
# Run this ONLY after each gate has passed its own promotion (DBG readback matching the host
# reference, red lines intact, no throughput regression) — see docs §64/§87. Promoting all five
# at once would violate the one-variable-at-a-time rule, so this is deliberately a manual,
# reviewed step: it lists each gate with its env pair so you can add them incrementally.
set -euo pipefail
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
GATES=(
 "DSV41_ROUTED_DOWN_QUANT=1"     # routing weight timing + routed-down input quantisation
 "DSV41_WINDOW_KV_QUANT=1"       # A2: window KV fp8/block32/e8m0 in-place roundtrip
 "DSV41_COMPRESS_LATENT_QUANT=1" # A3: compressed latent fp4/block16 with the e4m3 (non-pow2) scale
 "DSV41_INDEXER_FP4_RT=1"        # A4: indexer q/k fp4/block32 pow2 roundtrip
 "DSV41_ATTN_P_BF16=1"           # I3: attention PV probability operand rounded to bf16
)
LINE=$(printf '%s ' "${GATES[@]}")
echo "== precision envs to add =="
printf '  %s\n' "${GATES[@]}"
for f in "$HOME/push400_hw_test.sh" "$HOME/verify_correct.sh"; do
  [ -f "$f" ] || { echo "-- $f missing, skip"; continue; }
  if grep -q "DSV41_ROUTED_DOWN_QUANT=1" "$f"; then
    echo "-- $f already carries DSV41_ROUTED_DOWN_QUANT=1; checking the rest individually"
  fi
  missing=()
  for g in "${GATES[@]}"; do grep -q "${g%%=*}=1" "$f" || missing+=("$g"); done
  echo "-- $f: missing ${#missing[@]} of ${#GATES[@]} gate(s)"
  [ "${#missing[@]}" -eq 0 ] && continue
  if [ "$APPLY" = "1" ]; then
    cp "$f" "$f.bak.$(date +%Y%m%d%H%M%S)"
    # insert right after the line that sets DSV41_EXPERT_ACT_E4M3 (a stable anchor), else append
    if grep -q "DSV41_EXPERT_ACT_E4M3=1" "$f"; then
      python3 - "$f" "${missing[*]}" <<'PY'
import sys, re
f, ins = sys.argv[1], sys.argv[2]
s = open(f).read()
out = []
for line in s.split("\n"):
    out.append(line)
    if "DSV41_EXPERT_ACT_E4M3=1" in line and "DSV41_ROUTED_DOWN_QUANT" not in line:
        out.append("  " + ins.replace(" ", " \\\n  ") + " \\")
open(f, "w").write("\n".join(out))
PY
    else
      printf '\n# precision gates (promoted individually, see docs §64/§87)\nexport %s\n' "$LINE" >> "$f"
    fi
    echo "   APPLIED to $f (backup alongside)"
  else
    echo "   (dry-run) would add: ${missing[*]}"
  fi
done
echo "== done (dry-run=$((1-APPLY))) =="
