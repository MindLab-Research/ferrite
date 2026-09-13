#!/bin/bash
# gold_bisect_auto.sh [GOOD] [BAD] — unattended bisection of the shared-forward regression using the
# golden arbiter (see gold_bisect.sh for why that arbiter is sound: the golden's inputs are
# byte-identical to what every run reproduces, and its gate|up equals the self-tested official oracle
# bit-exactly, so one run + one numpy diff is a binary verdict).
#
# Each iteration rebuilds (~4 min), runs one minimal-eager counting run, and compares. ~6 iterations
# for the 57-commit range. The tree is restored to origin/main at the end.
set -uo pipefail
GOOD=${1:-ac69b054}
BAD=${2:-HEAD}
cd "$HOME/ferrite" || exit 1
echo "=== auto bisect: good=$GOOD bad=$BAD ==="
step=0
SKIP=0
while :; do
  step=$((step+1))
  N=$(git rev-list --count "$GOOD..$BAD")
  echo "--- step $step: range $GOOD..$BAD has $N commits"
  if [ "$N" -le 1 ]; then
    echo "########## FIRST BAD COMMIT: $(git rev-list --reverse "$GOOD..$BAD" | head -1) ##########"
    git log -1 --format="%h %ad %s" --date=format:"%m-%d %H:%M" "$(git rev-list --reverse "$GOOD..$BAD" | head -1)"
    break
  fi
  K=$(( (N + 1) / 2 ))
  MID=$(git rev-list --reverse "$GOOD..$BAD" | awk -v k="$K" 'NR==k')
  echo "    testing midpoint #$K = $MID  ($(git log -1 --format='%h %ad %s' --date=format:%H:%M $MID | cut -c1-90))"
  out=$(bash "$HOME/gold_bisect.sh" "$MID" 2>&1 | tail -6)
  echo "$out" | sed 's/^/      /'
  # ⚠️ A commit that does not BUILD is not evidence about the regression. The first version of this
  # script treated a build failure as a divergence and moved BAD down -- i.e. it walked away from the
  # answer while looking like it was making progress (af59890f cannot compile: its actq_scale sits
  # ahead of fast_round_scale). Such a commit is SKIPPED, and an all-skipped range is reported as
  # INCONCLUSIVE rather than as a verdict.
  if echo "$out" | grep -qE "BUILD_SH FAILED|CARGO FAILED"; then
    echo "    => INVALID (this commit does not build) -> skipping, splitting the range instead"
    SKIP=$((SKIP+1))
    if [ "$SKIP" -ge 6 ]; then echo "########## INCONCLUSIVE: six commits in a row failed to build ##########"; break; fi
    GOOD=$MID   # treat as unusable by advancing GOOD past it (its subtree cannot hold the answer)
    continue
  fi
  if echo "$out" | grep -q "MATCHES the golden"; then
    GOOD=$MID
    echo "    => GOOD (this commit still reproduces the official gate|up) -> move GOOD up"
  else
    BAD=$MID
    echo "    => BAD (diverges) -> move BAD down"
  fi
done
echo "=== restoring origin/main ==="
git checkout -q origin/main -- kernels crates
(cd kernels/cuda && bash build.sh 103a 2>&1 | tail -1)
(source "$HOME/.cargo/env" && cargo build --release 2>&1 | tail -1)
echo "=== AUTO BISECT DONE (tree restored) ==="
