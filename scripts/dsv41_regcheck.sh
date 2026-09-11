#!/usr/bin/env bash
# Report each DSV4.1 kernel's register footprint as compiled for sm_103a with the
# production flags. This is how the "manual 4-way unroll vs #pragma unroll"
# difference gets diagnosed: manual unrolling forces four live copies of every
# scale/byte/address, and if that pushes a kernel over an occupancy boundary the
# cp.async staging timing changes - the model degenerated even though the
# arithmetic was provably identical (device-side 64/64 bitwise comparison).
#
# Do NOT run this while a serve A/B is in flight: it compiles for ~1 minute and
# competes for the CPU the serve host path needs.
set -euo pipefail
cd "$(dirname "$0")/../kernels/cuda"
echo "== register footprint (production flags, sm_103a) =="
nvcc -O3 -shared -Xcompiler -fPIC --use_fast_math -std=c++17 \
    -gencode arch=compute_103a,code=sm_103a -Xptxas -v \
    -DFERRITE_KERNEL_BUILD_ID='"regcheck"' -o /tmp/regcheck.so dsv41_*.cu 2>&1 \
    | grep -E "Function properties for|Used [0-9]+ registers|spill|stack frame" \
    | sed 's/^ *//' | head -80
