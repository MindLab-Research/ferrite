#!/usr/bin/env bash
# a32 / occupancy A/B for the DSV4.1 M=1 fp8 GEMV, at PRODUCTION shapes.
#
# THE QUESTION. The block-wide pre-decoded activation (`s_af`, "a32": k f32 =
# 20 KB at k=5120) was measured as a -6/-8/-13% win on the small-n probes
# (n=256/1024/1664) and has never been re-measured at the production k. At
# k=5120 it is what lifts the block's dynamic shared memory from ~28 KB to
# ~47.4 KB, i.e. from 8 resident blocks/SM to 4. If the occupancy doubling is
# worth more than the LDS latency a32 saves, dropping it is a free -0.74 ms/step.
#
# THE KNOB. NOT DSV41_GEMV_FP8_MODE. READ THIS BEFORE CHANGING THE ARMS:
#   mode 0 = scalar, 1 = vectorised, 3 = staged+ordered, 4 = staged+ordered +
#   block-wide ACTIVATION STAGING (the fp8 `s_a` copy, k bytes).
#   a32 (`s_af`, 4k bytes) is built in BOTH mode 3 and mode 4 - mode 3 just reads
#   the fp8 activation from global memory instead of the staged copy (see ap0/ap
#   in gemm_fp8_gemv_kernel). So "mode 3 vs mode 4" isolates the k-byte activation
#   staging, NOT the 4k-byte a32 table. The a32 table has its own gate:
#       DSV41_GEMV_A32=1  keep a32   (default, current production)
#       DSV41_GEMV_A32=0  drop a32   (frees 20 KB at k=5120; bit-identical)
#
# Both gates are read once per process (static), so each ARM IS A SEPARATE
# PROCESS. Arms:
#   m4_a32   DSV41_GEMV_FP8_MODE=4 DSV41_GEMV_A32=1   <- baseline
#   m4_noa32 DSV41_GEMV_FP8_MODE=4 DSV41_GEMV_A32=0   <- the occupancy bet
#   m3_a32   DSV41_GEMV_FP8_MODE=3 DSV41_GEMV_A32=1   <- activation staging off
#   m3_noa32 DSV41_GEMV_FP8_MODE=3 DSV41_GEMV_A32=0   <- both off
#
# Usage:  scripts/dsv41_a32_bench.sh              # all four arms
#         scripts/dsv41_a32_bench.sh m4_a32 m4_noa32
#         ARMS="m4_a32 m4_noa32" scripts/dsv41_a32_bench.sh
#         SKIP_BUILD=1 scripts/dsv41_a32_bench.sh    # reuse the current .so
#
# Output: one SHAPE line per (arm, n), then a per-n delta table and a
# fingerprint check. A fingerprint mismatch between arms INVALIDATES the arm:
# the two a32 forms compute the same product, so any drift means something else
# moved.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
K="$ROOT/kernels/cuda"
BIN="/tmp/dsv41_a32_bench"
export LD_LIBRARY_PATH="$K${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ARCH="${ARCH:-100a}"
NS="${NS:-}"                       # e.g. NS="256 1024 4096"
REPS="${DSV41_A32_REPS:-400}"

ARMS="${*:-${ARMS:-m4_a32 m4_noa32 m3_a32 m3_noa32}}"

# --- the GPU must be free: a live serve makes every number meaningless -------- #
if pgrep -x dsv41-run >/dev/null 2>&1; then
    echo "refusing to start: a dsv41-run is alive (the GPU must be free)" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "refusing to start: nvidia-smi not found (no GPU on this node?)" >&2
    exit 1
fi

# --- the .so must match this tree (the a32 gate lives in the kernel) ---------- #
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "== building kernels ($ARCH) =="
    ( cd "$K" && bash build.sh "$ARCH" ) || { echo "build.sh FAILED" >&2; exit 1; }
else
    echo "== SKIP_BUILD=1: reusing $K/libferrite_kernels.so =="
fi
[ -f "$K/libferrite_kernels.so" ] || { echo "missing $K/libferrite_kernels.so" >&2; exit 1; }

echo "== compiling the bench harness =="
# nvcc takes neither -Wl,-rpath nor a bare -l path here: link -L + LD_LIBRARY_PATH.
nvcc -O2 -std=c++17 -o "$BIN" "$ROOT/scripts/dsv41_a32_bench.cu" -L"$K" -lferrite_kernels || {
    echo "harness compile FAILED" >&2; exit 1; }

run_arm() {
    local arm="$1" mode a32
    case "$arm" in
        m4_a32)   mode=4; a32=1 ;;
        m4_noa32) mode=4; a32=0 ;;
        m3_a32)   mode=3; a32=1 ;;
        m3_noa32) mode=3; a32=0 ;;
        *) echo "unknown arm '$arm' (m4_a32|m4_noa32|m3_a32|m3_noa32)" >&2; return 2 ;;
    esac
    echo "---- arm $arm (MODE=$mode A32=$a32) ----"
    # shellcheck disable=SC2086
    env DSV41_GEMV_FP8_MODE="$mode" DSV41_GEMV_A32="$a32" DSV41_A32_REPS="$REPS" \
        "$BIN" $NS 2>&1 | tee "/tmp/a32_${arm}.log"
}

for a in $ARMS; do run_arm "$a" || exit 1; done

echo
echo "== per-n comparison =="
python3 - "$ARMS" <<'PY'
import re, sys
arms = sys.argv[1].split()
data, order, fps = {}, [], {}
for arm in arms:
    try:
        lines = open(f"/tmp/a32_{arm}.log").read().splitlines()
    except FileNotFoundError:
        continue
    for ln in lines:
        m = re.search(r"SHAPE n=(\d+)\s+k=(\d+)\s+mode=(\d+)\s+warps=(\d+)\s+smem=(\d+)\s+occ=(-?\d+)\s+median=\s*([\d.]+) us.*fp=(\S+)", ln)
        if not m:
            continue
        n, k, mode, warps, smem, occ, us, fp = m.groups()
        data.setdefault(arm, {})[int(n)] = dict(us=float(us), smem=int(smem), occ=int(occ))
        fps.setdefault(arm, {})[int(n)] = fp
        if int(n) not in order:
            order.append(int(n))

base = "m4_a32" if "m4_a32" in data else (arms[0] if arms else None)
if base is None or base not in data:
    print("  (no SHAPE lines parsed)"); sys.exit(1)

print(f"  base arm: {base}")
hdr = f"  {'n':>6} " + " ".join(f"{a:>22}" for a in arms)
print(hdr)
for n in order:
    cells = []
    b = data[base].get(n)
    for a in arms:
        d = data.get(a, {}).get(n)
        if not d:
            cells.append(f"{'-':>22}"); continue
        tag = f"{d['us']:8.3f}us occ{d['occ']} {d['smem']//1024:>3}KB"
        if a != base and b:
            tag += f" {d['us']-b['us']:+7.3f}"
        cells.append(f"{tag:>22}")
    print(f"  {n:>6} " + " ".join(cells))

# fingerprint gate: a mismatch means the "occupancy" arm changed the arithmetic.
if base in fps:
    bad = 0
    for n, fp in fps[base].items():
        for a in arms:
            if a in fps and n in fps[a] and fps[a][n] != fp:
                print(f"  !! FINGERPRINT MISMATCH n={n} {base} vs {a}: {fp} vs {fps[a][n]}")
                bad += 1
    print("  fingerprint: " + ("OK (identical across arms)" if bad == 0 else f"*** {bad} MISMATCH ***"))

# verdict on the occupancy bet, at the largest n
if base in data and "m4_noa32" in data:
    for n in reversed(order):
        b, c = data[base].get(n), data["m4_noa32"].get(n)
        if b and c:
            print(f"  occupancy bet @ n={n}: {b['us']:.3f} -> {c['us']:.3f} us "
                  f"({c['us']-b['us']:+.3f} us)  occ {b['occ']} -> {c['occ']}  "
                  f"smem {b['smem']//1024}KB -> {c['smem']//1024}KB")
            break
PY

echo
echo "logs: /tmp/a32_m4_a32.log /tmp/a32_m4_noa32.log /tmp/a32_m3_a32.log /tmp/a32_m3_noa32.log"
