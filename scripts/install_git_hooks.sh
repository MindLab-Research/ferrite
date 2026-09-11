#!/usr/bin/env bash
# Install ferrite's git hooks (scripts/git-hooks/*) into .git/hooks.
#
# The hooks live in the repo so they are reviewable and survive a re-clone; git
# itself only loads them from .git/hooks (or core.hooksPath), so they must be
# installed once per clone:
#
#     bash scripts/install_git_hooks.sh
#
# What they do: drop kernels/cuda/{libferrite_kernels.so,.build_id} on every
# checkout/merge, so a revision switch can never leave a stale .so behind. The
# compile-time gate in crates/ferrite-kernel/build.rs then refuses to produce a
# release binary until both artifacts are rebuilt together.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC="$ROOT/scripts/git-hooks"
DST="$ROOT/.git/hooks"
[ -d "$SRC" ] || { echo "FATAL: $SRC not found (run from the repo)"; exit 1; }
[ -d "$DST" ] || mkdir -p "$DST"

for h in "$SRC"/*; do
    [ -f "$h" ] || continue
    name="$(basename "$h")"
    case "$name" in
        *.sample|*~|_*) continue ;;
    esac
    target="$DST/$name"
    # Preserve any pre-existing, non-symlink hook instead of clobbering it.
    if [ -e "$target" ] && [ ! -L "$target" ]; then
        cp -f "$target" "$target.ferrite-bak"
        echo "backed up existing $name -> $name.ferrite-bak"
    fi
    chmod +x "$h"
    ln -sf "$h" "$target"
    echo "installed $name -> $h"
done

echo "done. hooks are per-clone; re-run this after cloning ferrite elsewhere."
