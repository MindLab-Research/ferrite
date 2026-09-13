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
if [ "$NEW" != "$OLD" ] || [ ! -f kernels/cuda/libferrite_kernels.so ]; then
  echo "=== kernel sources changed ($OLD -> $NEW): rebuilding .so (full build.sh) ==="
  (cd kernels/cuda && set -o pipefail; bash build.sh 103a 2>&1 | tail -2; echo KERNEL_RC=${PIPESTATUS[0]})
  echo "$NEW" > "$STAMP"
else
  echo "=== kernel sources unchanged ($NEW): skipping build.sh ==="
fi
source "$HOME/.cargo/env"
if [ ! -f target/release/ferrite-serve ] ||
   [ -n "$(find crates -name '*.rs' -newer target/release/ferrite-serve -print -quit 2>/dev/null)" ]; then
  echo "=== Rust sources changed: cargo build --release ==="
  (set -o pipefail; cargo build --release 2>&1 | tail -2; echo CARGO_RC=${PIPESTATUS[0]})
else
  echo "=== Rust sources unchanged: skipping cargo build ==="
fi
bash "$HOME/check_artifacts.sh" || { echo "ARTIFACTS_STALE — aborting before any measurement"; exit 1; }
