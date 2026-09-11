#!/usr/bin/env bash
# Bisect helper for the degenerate-serve hunt: check out a revision, rebuild BOTH
# products (the .so and the binary, same tree), then run one A/B arm and report.
# "Which change broke it" has to be answered by rebuilding at each candidate,
# because the symptom (identical text on all four prompts, 12 steps, faults=0)
# has survived three different fixes so far.
#
#   scripts/dsv41_bisect.sh <commit|tag> <tag>       # e.g. 5cddf22 mg1
#
# Leaves the tree detached at <commit|tag>; switch back explicitly afterwards.
set -euo pipefail
REV="${1:?usage: dsv41_bisect.sh <commit> <tag>}"
TAG="${2:?usage: dsv41_bisect.sh <commit> <tag>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git fetch origin main -q
git checkout -q "$REV"
echo "== building $REV ($(git log --oneline -1)) =="
( cd kernels/cuda && bash build.sh 103a >/tmp/bisect_build.log 2>&1 ) || {
    echo "build.sh FAILED"; tail -20 /tmp/bisect_build.log; exit 1; }
source ~/.cargo/env
cargo build --release >/dev/null 2>&1
bash scripts/dsv41_serve_ab.sh "$TAG"
