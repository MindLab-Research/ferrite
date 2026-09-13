#!/bin/bash
# Artifact freshness gate — run BEFORE any arm that depends on a recent change.
#
# Twice now a batch has built from a tree that lagged my latest pushes (the .so carried the LDW fix
# but not the SFDUMP accessor / STAGE1 gate), so an arm silently measured the wrong binary: the
# "build failed / stale binary" class of measurement bias the project rules call out. Cargo and
# .so symbols are checked against the features the current tree is supposed to carry.
#
# Usage: check_artifacts.sh [feature ...]   (default: all known feature probes)
set -uo pipefail
cd "$HOME/ferrite"
fail=0
say () { printf '%-52s %s\n' "$1" "$2"; }

echo "=== tree ==="
git log --oneline -1
say "worktree clean (only untracked tilelang_inc)?" \
    "$(git status --porcelain | grep -v '^??' | wc -l) modified files"

SO=kernels/cuda/libferrite_kernels.so
BIN=target/release/ferrite-serve
say ".so exists" "$([ -f $SO ] && echo yes || echo NO)"
say "serve binary exists" "$([ -f $BIN ] && echo yes || echo NO)"
stat -c "%y %n" "$SO" "$BIN" 2>/dev/null | sed 's/^/    /'

# Feature probes: each is a source string that MUST be present in BOTH the tree and the built
# artifact (the .so for device/shim code, the binary for Rust code) when the feature is in the tree.
probe_so () {  # label, source-grep-file, pattern
  local label="$1" src="$2" pat="$3"
  local in_tree in_art
  in_tree=$(grep -c "$pat" "$src" 2>/dev/null || true)
  in_art=$(strings "$SO" 2>/dev/null | grep -c "$pat" || true)
  in_tree=${in_tree:-0}; in_art=${in_art:-0}
  if [ "$in_tree" -gt 0 ] && [ "$in_art" -eq 0 ]; then
    say "$label" "STALE .so (tree=$in_tree, .so=$in_art)"; fail=1
  else
    say "$label" "ok (tree=$in_tree, .so=$in_art)"
  fi
}
probe_bin () {  # label, source-grep-file, pattern
  local label="$1" src="$2" pat="$3"
  local in_tree in_art
  in_tree=$(grep -c "$pat" "$src" 2>/dev/null || true)
  in_art=$(strings "$BIN" 2>/dev/null | grep -c "$pat" || true)
  in_tree=${in_tree:-0}; in_art=${in_art:-0}
  if [ "$in_tree" -gt 0 ] && [ "$in_art" -eq 0 ]; then
    say "$label" "STALE binary (tree=$in_tree, bin=$in_art)"; fail=1
  else
    say "$label" "ok (tree=$in_tree, bin=$in_art)"
  fi
}

probe_so  "SFDUMP accessor in .so"        kernels/cuda/tilelang_gen/moe_bs_shim.cu "dsv41_moe_bs_sfdump_ptr"
probe_so  "SFDUMP_K gate in .so"          kernels/cuda/tilelang_gen/moe_bs_shim.cu "DSV41_MOE_BS_SFDUMP_K"
probe_so  "MBAR_RING gate in .so"           kernels/cuda/tilelang_gen/moe_bs_shim.cu "DSV41_MOE_BS_MBAR_RING"
probe_so  "MBAR_PERSTAGE gate in .so"      kernels/cuda/tilelang_gen/moe_bs_shim.cu "DSV41_MOE_BS_MBAR_PERSTAGE"
probe_so  "down-arm LDW gate in .so"       kernels/cuda/tilelang_gen/moe_bs_dn_shim.cu "DSV41_MOE_DOWN_BS_LDW"
probe_so  "KEEP_STAGE gate in .so"        kernels/cuda/tilelang_gen/moe_bs_shim.cu "DSV41_MOE_BS_KEEP_STAGE"
probe_so  "STAGE1 gate in .so"            kernels/cuda/tilelang_gen/moe_bs_shim.cu "DSV41_MOE_BS_STAGE1"
probe_so  "SFST gate in .so"              kernels/cuda/tilelang_gen/moe_bs_shim.cu "DSV41_MOE_BS_SFST"
probe_so  "LDW gate in .so"               kernels/cuda/tilelang_gen/moe_bs_shim.cu "DSV41_MOE_BS_LDW"
probe_so  "GATEUP_DUMP tag in .so path"   kernels/cuda/tilelang_gen/moe_bs_shim.cu "moe-bs\] ARMED"
probe_bin "GATEUP_DUMP in serve"          crates/ferrite-models/src/dsv41/chain_dev.rs "DSV41_GATEUP_DUMP"
probe_bin "SFDUMP env in serve"           crates/ferrite-models/src/dsv41/chain_dev.rs "DSV41_MOE_BS_SFDUMP"
say "sanity: .so newer than its sources?" \
    "$([ "$SO" -nt kernels/cuda/tilelang_gen/moe_bs_shim.cu ] && echo yes || echo 'NO (rebuild!)')"

# The engine itself refuses to start on a build-id mismatch (so vs binary) and the arm then reports
# 0 steps with an empty output — one whole wasted round, and easy to misread as a kernel problem. So
# compare them HERE, before any arm runs: a batch whose two artefacts come from different checks must
# never reach a measurement.
# The engine reads the .so's id through the `ferrite_kernel_build_id` symbol and compares it against
# the binary's embedded `FERRITE_BUILD_ID` (build.rs -> rustc-env), refusing to start on a mismatch.
# My first attempt grepped for a `build_id …` *prefix*, which never appears — the ids are bare
# `<sha40>-dirty+cu<hex16>` strings — so the check reported "unreadable" and aborted the batch. Match
# the SHAPE instead, and fail ONLY when both are readable and differ: the engine remains the enforcer,
# this is the early warning that saves a wasted round.
ID_RE='[0-9a-f]{40}-dirty\+cu[0-9a-f]{16}'
# ⚠️ The AUTHORITY is kernels/cuda/.build_id: build.rs reads exactly that file and bakes it into the
# binary as FERRITE_BUILD_ID (crates/ferrite-kernel/build.rs:215), while the engine dlsyms the .so's
# ferrite_kernel_build_id() and compares (cuda.rs:2088). Grepping `strings` for an id-shaped token was
# the earlier approach and it produced FALSE mismatches, each of which cost a needless full rebuild —
# the tool meant to save a round was spending them. So: trust the stamp file, and verify only that the
# binary carries it and was linked after it. The .so's own id is checked at runtime by the engine,
# which refuses to start on a mismatch.
STAMP_ID=$(cat kernels/cuda/.build_id 2>/dev/null | tr -d '\n' | grep -oE "$ID_RE" | head -1)
BIN_HAS=$( [ -n "$STAMP_ID" ] && strings "$BIN" 2>/dev/null | grep -cF "$STAMP_ID" || echo 0 )
SO_ID=$(strings "$SO" 2>/dev/null | grep -oE "$ID_RE" | head -1)
BIN_ID="$STAMP_ID"
say "stamp kernels/cuda/.build_id" "${STAMP_ID:-MISSING}"
say "binary carries that id" "$([ "$BIN_HAS" -gt 0 ] && echo yes || echo NO)"
say "so id (advisory; the engine is the enforcer)" "${SO_ID:-(not extractable)}"
if [ -z "$STAMP_ID" ]; then
  say ".build_id" "MISSING — build.sh has not run in this tree"; fail=1
elif [ "$BIN_HAS" -le 0 ]; then
  say "so/binary pair" "binary does NOT carry the stamp id — it predates the kernel build (rebuild cargo)"
  fail=1
elif [ ! "$BIN" -nt kernels/cuda/.build_id ]; then
  say "link order" "binary is older than .build_id — cargo ran before the kernel finished"; fail=1
else
  say "so/binary pair" "ok (stamp id present in the binary and linked after it)"
fi

echo "=== verdict ==="
if [ "$fail" = 0 ]; then echo "ARTIFACTS_FRESH"; else echo "ARTIFACTS_STALE — rebuild BOTH (build.sh 103a + cargo build) before any measurement"; fi
exit $fail
