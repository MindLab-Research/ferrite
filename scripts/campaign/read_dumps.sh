#!/bin/bash
# Reading the current dumps: (a) are the two runs' INPUTS identical (to separate "upstream
# non-determinism" from "the BS kernel's own race"), (b) which buffer does the ZERO gate touch
# (the ZERO_A arm's "nonzero output" is only a smoking gun if it zeroes the buffer the device path
# actually gathers from).
set -uo pipefail
echo "=== (a) input identity: the older run vs the fresh one ==="
for f in x.f32 ids.i32 xq4.u8 xsc4.f32; do
  a=/tmp/gu_in_GD4_BS/$f
  b=/tmp/lv_REF/eager/$f
  if [ ! -f "$a" ] || [ ! -f "$b" ]; then echo "$f: missing"; continue; fi
  if cmp -s "$a" "$b"; then
    echo "$f IDENTICAL ($(stat -c %s "$b") B)"
  else
    python3 - "$a" "$b" "$f" <<'PY'
import sys, numpy as np
a, b, name = sys.argv[1], sys.argv[2], sys.argv[3]
dt = "<f4" if name.endswith("f32") else "u1"
A = np.fromfile(a, dtype=dt).astype(np.float64)
B = np.fromfile(b, dtype=dt).astype(np.float64)
n = min(A.size, B.size)
d = np.abs(A[:n] - B[:n])
print(f"{name} DIFFERS: n={n} max|d|={d.max():.6g} nonzero={int((d != 0).sum())} "
      f"first8_A={A[:8]} first8_B={B[:8]}")
PY
  fi
done
echo
echo "=== (b) the ZERO gate's target vs the device path's A buffer ==="
cd "$HOME/ferrite/kernels/cuda/tilelang_gen"
grep -n "g_zero_mode" moe_bs_shim.cu | head -10
echo "--- which pointer does the *device-table* entry gather into? ---"
sed -n '/dsv41_moe_tilelang_gate_up_bs_dev/,/tl_bs_numcheck/p' moe_bs_shim.cu \
  | grep -n "gather_kernel\|g_a\|A_dev\|= A\b" | head -12
