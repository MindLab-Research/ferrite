#!/usr/bin/env bash
# Build the ferrite CUDA kernels. Compiling needs nvcc but NO GPU (compile
# only) — safe to run on a busy B300 node.
#
# Usage: ./build.sh [sm_arch]     default: 100a (B300 Blackwell Ultra)
#        FERRITE_OUT=libferrite_kernels.so ./build.sh
set -euo pipefail

ARCH="${1:-100a}"
OUT="${FERRITE_OUT:-libferrite_kernels.so}"
DIR="$(dirname "$0")"
# DeepSeek-V4.1-Flash kernels live in their own translation units; they are
# linked into the same .so so the engine keeps a single dlopen target.
SRCS=("$DIR/ferrite_kernels.cu")
# the tcgen05 MXFP4 expert GEMM is its own TU; tests_*.cu carry a main
# and are deliberately NOT linked into the shared object.
#
# PROJ-MMA (`dsv41_proj_mma_skel.cu`, 2026-09-13): the TENSOR-CORE form of the
# multi-row fp8 projection GEMV. It rides here as its OWN TU on purpose — the
# kernel, its launcher (`dsv41_gemm_fp8_mrows_mma`) and the `DSV41_PROJ_MMA`
# gate are self-contained (their only in-tree neighbours are the swapAB gemv
# they generalise), so linking the file directly keeps the future move into
# `dsv41_kernels.cu` a pure relocation.
# ⚠️ The symbol now being present in EVERY `.so` is NOT a behaviour change: the
# runtime gate is default OFF and the Rust arm is the only way in
# (docs/agent/tensorcore-proj-design.md §3.3/§7). The build-vs-runtime split is
# the same contract the tcgen05 blocks below carry — a build that omitted the
# TU would make a `DSV41_PROJ_MMA=1` A/B silently measure the OLD path.
#
# MOE-ALIGN (`dsv41_moe_align.cu`, 2026-09-13): the DEVICE-side projection of the
# TileLang MoE arm's segment tables out of `dsv41_route_group`'s output
# (`dsv41_moe_align_from_group`). It rides here as its OWN TU because it needs no
# shared memory at all while `dsv41_route.cu` owns the TU-local anonymous
# `extern __shared__` alias — a second dynamic-smem declaration in that TU is a
# redeclaration error. Same build-vs-runtime split as the blocks above: the
# symbol is COMPILED IN unconditionally, and the runtime gate
# (`DSV41_MOE_TILELANG` + `moe_tilelang_ready`'s probe) is the only way in. A
# build that omitted this TU would make the device-table arm silently measure the
# HOST-table path — the project's #1 measurement-bias trap.
for f in "$DIR"/dsv41_kernels.cu "$DIR"/dsv41_experts_mxf4.cu "$DIR"/dsv41_vision.cu "$DIR"/dsv41_glue.cu "$DIR"/dsv41_route.cu "$DIR"/dsv41_moe_align.cu "$DIR"/dsv41_proj_mma_skel.cu; do
    [ -f "$f" ] && SRCS+=("$f")
done

# TILELANG (DSV41_GEMM_TILELANG, 2026-09-13): the FIRST TileLang-generated projection
# kernel, forked from the tilelang-proj-proto prototype into the tree (route (a)
# "source vendoring" of docs/agent/tilelang-integration-design.md §2).
#
# Layout: `tilelang_gen/wkv_shim.cu` is the ONLY TU taken from this subdir. The
# frozen `wkv_*_tl.cu` dumps are #included by it (renamed — TileLang names every
# kernel `main_kernel`, so two dumps in one link are a duplicate symbol) and are
# deliberately NOT standalone SRCS. `tilelang_inc/` is the vendored header set
# (tl_templates + the cutlass/cute transitive closure; see tilelang_gen/PROVENANCE.md).
#
# ⚠️ Same build-vs-runtime split as the other gated blocks here (mxf4 / e4m3 / PROJ-MMA):
# the shim is COMPILED IN unconditionally so the symbol is always present in the .so;
# the runtime gate `DSV41_GEMM_TILELANG` (default OFF, read once in chain_dev.rs) is
# the only way in. A build that omitted the TU would make a `DSV41_GEMM_TILELANG=1`
# A/B silently measure the OLD path (the project's #1 measurement-bias trap).
# ⚠️ The frozen geometry (n=512/k=5120/bN=128/ks=8) is baked into the generated grid
# and index arithmetic, so the C entry DECLINES (returns 2) every other shape and the
# caller keeps its existing kernel. Extending to the second shape is per-shape: see
# tilelang_gen/PROVENANCE.md §5.
for f in "$DIR"/tilelang_gen/*_shim.cu; do
    [ -f "$f" ] && SRCS+=("$f")
done

NVCC="${NVCC:-nvcc}"
"$NVCC" --version >/dev/null 2>&1 || { echo "error: nvcc not found (CUDA toolkit required)"; exit 1; }

# -O3 + fPIC shared object. --use_fast_math is OPT-IN via FERRITE_FAST_MATH=1
# (2026-09-10: enforced-by-default measured a ~2x replay regression —
# 27.99ms vs 13.40ms at full 2032 MHz clock, i.e. not thermal/power).
# --use_fast_math: DEFAULT ON (matching the working config). Set
# FERRITE_NO_FAST_MATH=1 to disable for A/B. WARNING (measured 2026-09-10):
# the .so built WITHOUT it CRASHES the batched capture (faults=2, err 900) —
# fast-math shifts kernel durations and thereby the in-capture pool size-class
# requests (the known pool-size sensitivity). Keep it ON unless investigating.
#
# CONSEQUENCE (2026-09-11, cost a full debug cycle): because it stays ON, the
# compiler MAY REASSOCIATE floating-point expressions. Any rewrite that gives it
# more expression freedom — multi-way unrolling, split accumulators — must pin
# every add/mul with __fadd_rn/__fmul_rn. A four-way unrolled gemv body written
# with plain operators drifted ~1 ULP per layer and degenerated the model after
# 40 layers (all four prompts collapsed to the same output). `fmaf(...)` chains
# are inherently safe (fmaf is explicitly rounded). Same for `extern __shared__`:
# the launcher's THIRD argument must carry the size (a zero made a staging buffer
# point at nothing and faulted: 4 faults, empty outputs).
FAST_MATH_FLAG="--use_fast_math"
if [ -n "${FERRITE_NO_FAST_MATH:-}" ]; then FAST_MATH_FLAG=""; fi

# ---------------------------------------------------------------------------
# GATED in-tree kernel blocks (2026-09-12). Both tcgen05 gate/up arms live in
# dsv41_experts_mxf4.cu behind a COMPILE-TIME #ifdef, so the .so either carries
# their extern "C" entry point or it does not — the Rust side probes the SYMBOL
# (`Device::supports_expert_tcgen05_mxf4`, device.rs:4289), never a version
# string.
#   mxf4 arm      DEFAULT ON — -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1
#                 -> symbol dsv41_expert_tcgen05_gate_up_mxf4
#                 opt OUT with DSV41_BUILD_TCGEN05_MXF4=0|no|false|off|<empty>
#   mxf8f6f4 arm  opt-in    — DSV41_BUILD_TCGEN05_MXF8F6F4=<nonempty>
#                 -> -DDSV41_TCGEN05_GATEUP_SKELETON=1
#   e4m3 arm      DEFAULT ON — -DDSV41_TCGEN05_GATEUP_E4M3_SKELETON=1
#                 -> symbol dsv41_expert_tcgen05_gate_up_e4m3
#                 opt OUT with DSV41_BUILD_TCGEN05_E4M3=0|no|false|off|<empty>
#                 The e4m3 arm is the (b) path's e4m3-activation sibling (gated
#                 at runtime by DSV41_EXPERT_TCGEN05_E4M3, default OFF): same
#                 build-vs-runtime split as the mxf4 arm, and for the same
#                 reason — an opt-in macro makes `supports_expert_tcgen05_e4m3()`
#                 false in a stock .so, so every A/B run with the gate ON would
#                 silently measure the e4m1/GEMV path instead (the project's #1
#                 measurement-bias trap).
# The two macros are independent by design (dsv41_experts_mxf4.cu:3427 — nested
# namespaces `tc5` vs `tc5::mxf4`), so enabling one can never change the other.
#
# WHY THE mxf4 ARM IS ON BY DEFAULT (2026-09-12, the fix for "dispatch 不可达").
# The Rust dispatch test is a TWO-PART gate: this BUILD-TIME symbol probe AND the
# process-static RUNTIME gate `DSV41_EXPERT_TCGEN05[_MXF4]` (default OFF, so the
# proven GEMV/GEMM path stays in force). While the macro was opt-in, a stock .so
# had no symbol at all, so `supports_expert_tcgen05_mxf4()` was false and every
# `DSV41_EXPERT_TCGEN05=1` A/B silently measured the OLD path (one-shot warning
# only). Compiling the block in unconditionally makes the symbol always present
# and leaves the RUNTIME default OFF — exactly the "compiled in, still gated"
# contract already documented at the end of this script. The symbol being
# present is NOT sufficient to change any behaviour.
# ⚠️ The mxf8f6f4 arm stays opt-in: it is the older 1X form (kRing=3) and is NOT
# the routed-expert-residual (b) path.
#
# ⚠️ The flag set is folded into BUILD_ID below. Without that, a flag-less
# rebuild of the same source would rewrite .build_id with the SAME string while
# the .so silently lost the symbol — the same-source gate would call a
# symbol-less .so "matching" and the serve A/B would measure the OLD path on
# both arms (the project's #1 measurement-bias trap). Consequence: ANY build
# after this change gets a new id, so rebuild BOTH products (see
# scripts/dsv41_serve_ab.sh for the one working order).
# ⚠️ scripts/dsv41_serve_ab.sh SELF-HEALS a stale pair by re-running this script.
# That is now SAFE BY DEFAULT: the mxf4 symbol is in the flag set unless
# explicitly opted out, so an UNEXPORTED rebuild keeps it (before this change it
# dropped it — failure mode (a) in scripts/dsv41_tcgen05_mxf4_verify.sh). `=0`
# is the only way to lose it, and then the Rust side sees no symbol and falls
# back loudly.
# ---------------------------------------------------------------------------
SKELETON_FLAGS=()
# mxf4 (the (b) path): DEFAULT ON; an explicit 0/no/false/off — or an EMPTY
# value, which used to mean OFF — opts out. Every other value keeps it ON.
case "${DSV41_BUILD_TCGEN05_MXF4-1}" in
    0|no|false|off|"") ;;
    *) SKELETON_FLAGS+=(-DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1) ;;
esac
if [ -n "${DSV41_BUILD_TCGEN05_MXF8F6F4:-}" ]; then
    SKELETON_FLAGS+=(-DDSV41_TCGEN05_GATEUP_SKELETON=1)
fi
# e4m3 arm (the (b) path's e4m3-activation sibling): DEFAULT ON, same opt-out
# rule as the mxf4 arm (0/no/false/off/empty disables).
# ⚠️ The two arms are INDEPENDENT by design (nested namespaces `tc5::mxf4` vs
# `tc5::e4` in the same TU), so enabling or disabling one can never change the
# other's codegen or symbols.
case "${DSV41_BUILD_TCGEN05_E4M3-1}" in
    0|no|false|off|"") ;;
    *) SKELETON_FLAGS+=(-DDSV41_TCGEN05_GATEUP_E4M3_SKELETON=1) ;;
esac
# Build stamp: the Rust side refuses to load a .so built from another
# revision (user rule: 严禁组合不同版本). Use the git revision of THIS tree.
BUILD_ID="$(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$(dirname "$0")" status --porcelain 2>/dev/null)" ]; then
  BUILD_ID="${BUILD_ID}-dirty"
fi
# Same-source enforcement: fold the .cu content hash in. The Rust side embeds
# whatever this script last wrote to .build_id, so rebuilding only ONE of the
# two artifacts produces a mismatch and the process REFUSES TO START.
CU_HASH="$( { sha256sum "${SRCS[@]}"; echo "flags ${SKELETON_FLAGS[*]-<none>}"; } \
            | sha256sum | cut -c1-16)"
BUILD_ID="${BUILD_ID}+cu${CU_HASH}"
echo "$BUILD_ID" > "$(dirname "$0")/.build_id"

# SAME-FAST-MATH SPLIT (2026-09-14, the fp4-MoE illegal-instruction root cause):
# TileLang's tcgen05 inline-asm kernels MUST NOT be compiled with
# --use_fast_math — the JIT (which works) does not use it, and with it the
# MMA traps with CUDA_ERROR_ILLEGAL_INSTRUCTION (five-way review + JIT A/B
# isolated; the flag lets ptxas reassociate/schedule around the asm in a way
# the tcgen05 unit rejects). The REST of the .so keeps fast math ON (removing
# it globally regressed the batched capture by ~2x and destabilized the pool
# size classes — see the 2026-09-10 note below). So: tilelang_gen shims are
# compiled as their own TUs WITHOUT the flag, everything else WITH it, and the
# .o files are linked into one .so.
TL_OBJS=()
OTHER_SRCS=()
for f in "${SRCS[@]}"; do
    case "$f" in
        */tilelang_gen/*_shim.cu)
            obj="${f%.cu}.no_fm.o"
            "$NVCC" -O3 -c -Xcompiler -fPIC -std=c++20 \
                -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
                -I "$DIR/tilelang_inc" \
                $( [ -d "/opt/dlami/nvme/dsv41_venv/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include" ] && echo "-I /opt/dlami/nvme/dsv41_venv/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include" ) \
                -o "$obj" "$f" || exit 1
            TL_OBJS+=("$obj")
            ;;
        *) OTHER_SRCS+=("$f") ;;
    esac
done

"$NVCC" -O3 -shared -Xcompiler -fPIC $FAST_MATH_FLAG \
    -std=c++20 \
    -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
    -I "$DIR/tilelang_inc" \
    $( [ -d "/opt/dlami/nvme/dsv41_venv/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include" ] && echo "-I /opt/dlami/nvme/dsv41_venv/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include" ) \
    -DFERRITE_KERNEL_BUILD_ID="\"${BUILD_ID}\"" \
    "${SKELETON_FLAGS[@]+"${SKELETON_FLAGS[@]}"}" \
    -o "$OUT" "${OTHER_SRCS[@]}" "${TL_OBJS[@]+"${TL_OBJS[@]}"}"

echo "built ${OUT} for sm_${ARCH} from ${SRCS[*]} (build_id ${BUILD_ID})"
# NOTE: the `|| true` is load-bearing under `set -e`. With the mxf4 flag now
# opt-out-able, SKELETON_FLAGS can be empty again, and a FALSE `A && B` as the
# LAST statement of the script would make an otherwise successful build exit 1
# (scripts/verify_graph_ab.sh:151-161 already had to work around exactly that).
[ ${#SKELETON_FLAGS[@]} -gt 0 ] && \
    echo "  skeleton flags: ${SKELETON_FLAGS[*]} (gated blocks are COMPILED IN; each still needs its runtime env gate: DSV41_EXPERT_TCGEN05[_MXF4] for the e2m1 arm, DSV41_EXPERT_TCGEN05_E4M3 for the e4m3 arm, plus DSV41_MOE_BATCH on and DSV41_EXPERT_ILV=0 for either routed mxf4 arm)" \
    || true
