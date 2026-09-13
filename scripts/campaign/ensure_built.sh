#!/bin/bash
# Build only what actually changed.
#
# A round used to cost: build.sh (8 TUs, and my own `touch` forced a FULL rebuild of the heavy ones
# — 10+ min sometimes) + cargo build --release (~51 s) + the arm itself only ~2.5 min. The build
# therefore dominated the measurement by several times, which is the user's complaint.
#
# Gates:
#   * kernel side: md5 of every .cu/inc source vs a stamp file. This is STRONGER than build.sh's own
#     staleness logic (which once silently skipped a changed shim — check_artifacts.sh caught it),
#     and it is immune to `touch` (content, not mtime).
#   * Rust side: newest .rs mtime vs the binary.
# check_artifacts.sh still runs afterwards and is the authority on whether the artefacts match the
# tree, so a wrong skip cannot silently measure a stale binary.
set -uo pipefail
cd "$HOME/ferrite"
STAMP="$HOME/.ferrite_cu_stamp"
NEW=$(cat kernels/cuda/*.cu kernels/cuda/tilelang_gen/*.cu kernels/cuda/tilelang_inc/tl_templates/cuda/*.h 2>/dev/null | md5sum | cut -d' ' -f1)
OLD=$(cat "$STAMP" 2>/dev/null || echo none)

# ⚠️ The stamp only proves the SOURCES did not change. It says NOTHING about whether the .so was built
# FROM those sources: a partial `git checkout <rev> -- kernels crates` (what the bisect scripts do)
# leaves the sources at one commit and the .so at another. Measured consequence: this script printed
# "unchanged, skipping", and build.rs then refused the pair with
#   error: failed to run custom build command for ferrite-kernel   (CARGO_RC=101)
# i.e. the gate was right and this decision was wrong. So verify the three build-id sources
# (.so / binary / kernels/cuda/.build_id) BEFORE trusting the stamp, and force the rebuild when they
# disagree — that is the only way "stale pair" cannot be silently measured.
ART_OK=1
bash "$HOME/check_artifacts.sh" >/dev/null 2>&1 || ART_OK=0
if [ "$NEW" != "$OLD" ] || [ ! -f kernels/cuda/libferrite_kernels.so ] || [ "$ART_OK" = 0 ]; then
  echo "=== rebuilding .so (full build.sh) — reason: $( \
        [ "$NEW" != "$OLD" ] && echo 'kernel sources changed' || \
        { [ "$ART_OK" = 0 ] && echo 'artefacts are NOT same-source (check_artifacts says stale)' || echo 'no .so yet'; } ) ==="
  set -o pipefail
  (cd kernels/cuda && bash build.sh 103a) 2>&1 | tail -2
  rc=$?
  echo "KERNEL_RC=$rc"
  if [ "$rc" != 0 ]; then
    echo "KERNEL BUILD FAILED — aborting: a stale .so must never be measured (this exact lapse let a"
    echo "arm run against the previous .so while the freshly changed TU failed to compile)."
    exit 1
  fi
  echo "$NEW" > "$STAMP"
else
  echo "=== kernel sources unchanged ($NEW): skipping build.sh ==="
fi
source "$HOME/.cargo/env"
if [ ! -f target/release/ferrite-serve ] ||
   [ -n "$(find crates -name '*.rs' -newer target/release/ferrite-serve -print -quit 2>/dev/null)" ]; then
  echo "=== Rust sources changed: cargo build --release ==="
  set -o pipefail
  cargo build --release 2>&1 | tail -2
  rc=$?
  echo "CARGO_RC=$rc"
  if [ "$rc" != 0 ]; then echo "CARGO BUILD FAILED — aborting"; exit 1; fi
else
  echo "=== Rust sources unchanged: skipping cargo build ==="
fi
bash "$HOME/check_artifacts.sh" || { echo "ARTIFACTS_STALE — aborting before any measurement"; exit 1; }
