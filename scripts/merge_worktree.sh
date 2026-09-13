#!/bin/bash
# merge_worktree.sh <worktree-dir> <new-symbol> [<must-still-exist-symbol> ...]
# Automates the merge discipline that AGENTS.md now requires (distilled from the §83 incident):
#   1. export the worktree's UNCOMMITTED code diff (docs excluded -> no conflicts with mine)
#   2. pre-check, then git apply; on conflict fall back to patch -p1 -F3
#   3. confirm landing THREE ways: new-symbol count > 0, diff --stat printed, and every
#      "must still exist" symbol still present (my earlier fixes must not be clobbered)
#   4. refuse to continue if any .rej/.orig residue appeared
#   5. sweep for duplicate function definitions in the touched .cu files (§83) and compile the
#      touched .cu with build.sh's REAL flags (--use_fast_math), because a single-file check
#      without them missed that bug
# It does NOT commit: inspect the output first.
set -euo pipefail
WT=${1:?usage: merge_worktree.sh <worktree-dir> <new-symbol> [must-keep-symbol ...]}
NEWSYM=${2:?need the new symbol to confirm landing}
shift 2
KEEP=("$@")
REPO=$HOME/src/ferrite
cd "$WT"
# A worktree may carry its change as an UNCOMMITTED diff or as a local COMMIT (the cp.async
# line committed 29edc66). Detect which, so the patch is never silently empty.
if [ -n "$(git status --porcelain -- crates kernels)" ]; then
  echo "== worktree has uncommitted changes: diffing HEAD =="
  git diff HEAD -- crates kernels > /tmp/mw.patch
else
  echo "== worktree is clean: taking its last commit's diff (HEAD~1..HEAD) =="
  git diff HEAD~1 HEAD -- crates kernels > /tmp/mw.patch
fi
[ -s /tmp/mw.patch ] || { echo "!!! empty patch — refusing to continue"; exit 1; }
echo "== patch size: $(wc -l < /tmp/mw.patch) lines =="
git diff HEAD --stat -- crates kernels | tail -6

cd "$REPO"
if git apply --check /tmp/mw.patch 2>/dev/null; then
  git apply /tmp/mw.patch && echo "== applied cleanly (git apply) =="
else
  echo "== git apply would conflict; falling back to patch -p1 -F3 =="
  patch -p1 -F3 --no-backup-if-mismatch < /tmp/mw.patch | tail -6
fi

echo "== residue check (must be empty) =="
find . -path ./target -prune -o \( -name '*.rej' -o -name '*.orig' \) -print | head -5

echo "== landing confirmation =="
printf '  new symbol %-28s occurrences: ' "$NEWSYM"; grep -rc "$NEWSYM" crates kernels 2>/dev/null | grep -v ':0$' | head -3 || echo 0
for s in "${KEEP[@]}"; do
  printf '  must-still-exist %-24s occurrences: ' "$s"; grep -rc "$s" crates kernels 2>/dev/null | grep -v ':0$' | head -2 || echo "MISSING!"
done

echo "== duplicate-definition sweep on touched .cu files =="
python3 - <<'PY'
import re, collections, subprocess
files = subprocess.run(["git","diff","--name-only"], capture_output=True, text=True).stdout.split()
cu = [f for f in files if f.endswith(".cu")] or ["kernels/cuda/dsv41_glue.cu","kernels/cuda/dsv41_kernels.cu"]
pat = re.compile(r'^\s*(?:__device__\s+|__global__\s+|static\s+)*[\w:<>\*&\s]+?(\w+)\s*\([^;]*\)\s*\{')
for f in cu:
    try: lines = open(f).read().split('\n')
    except Exception: continue
    d = collections.defaultdict(list)
    for i,l in enumerate(lines,1):
        m = pat.match(l)
        if m and ('__device__' in l or '__global__' in l or l.strip().startswith('static')):
            d[m.group(1)].append(i)
    dup = {k:v for k,v in d.items() if len(v)>1}
    print(f"  {f}: defs={len(d)} duplicate-names={len(dup)}" + (f"  !!! {dup}" if dup else ""))
PY
echo "== REAL-FLAGS compile gate: compile the touched .cu in a TEMP DIR on the remote =="
# Deliberately NOT via a temporary commit + push: that churns origin/main and once left a
# stray 'TEMP' commit behind when the script was interrupted after pushing but before its
# local rollback. Copying the files to a scratch directory and compiling there leaves history
# untouched and still uses build.sh's real per-file flags.
cd "$REPO"
CU=($(git diff --cached --name-only 2>/dev/null | grep '\.cu$' || true))
[ ${#CU[@]} -eq 0 ] && CU=($(git diff --name-only | grep '\.cu$'))
[ ${#CU[@]} -eq 0 ] && CU=("kernels/cuda/dsv41_glue.cu")
FAILED=0
for f in "${CU[@]}"; do
  base=$(basename "$f")
  # §22 trap: moe_bs_handwritten.cu is #included by moe_bs_shim.cu and must NOT be compiled
  # standalone (bogus 'expected a ";"'). Check it through its shim instead — compiling it
  # directly is what produced 4 false errors and wrongly rejected a good patch.
  [ "$base" = "moe_bs_handwritten.cu" ] && base="moe_bs_shim.cu"
  case "$f" in
    */tilelang_gen/*) FLAGS="-O2" ; INC="-I. -I../tilelang_inc" ;;
    *)                FLAGS="-O3 --use_fast_math" ; INC="-I." ;;
  esac
  # ship the whole tilelang_gen dir for shims (the shim includes its sibling kernel)
  case "$f" in
    */tilelang_gen/*) SRC="kernels/cuda/tilelang_gen" ; DEST="tilelang_gen" ;;
    *)                SRC="kernels/cuda" ; DEST="cuda" ;;
  esac
  ssh -o BatchMode=yes ubuntu@43.202.208.136 "rm -rf /tmp/mgate && mkdir -p /tmp/mgate" >/dev/null
  scp -q -o BatchMode=yes -r "$SRC" ubuntu@43.202.208.136:/tmp/mgate/ 2>/dev/null
  n=$(ssh -o BatchMode=yes ubuntu@43.202.208.136 "cd /tmp/mgate/$DEST && /usr/local/cuda-13.2/bin/nvcc -c $base -o /tmp/mgate.o -gencode arch=compute_103a,code=sm_103a $FLAGS -std=c++17 $INC 2>&1 | grep -c 'error'" || echo 999)
  printf '   %-40s errors=%s\n' "$base" "$n"
  [ "$n" = "0" ] || FAILED=1
done
if [ "$FAILED" != "0" ]; then
  echo "!!!! REAL-FLAGS COMPILE FAILED — undoing the working-tree merge (no commit was made)"
  git checkout -- crates kernels
  exit 2
fi
echo "== DONE (temporary commit left in place: amend its message, or reset and re-commit) =="
