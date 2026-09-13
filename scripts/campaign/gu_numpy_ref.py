#!/usr/bin/env python3
"""CPU-only, formula-level reference for the routed-expert gate/up GEMM.

Mirrors `ref_inference/kernel.py`'s `fp4_gemm_kernel` structure exactly (the official PyTorch
implementation is the only trusted oracle):

    for each 32-K block b:                      # block_K = weight_group_size = 32
        C_local = sum_{k in block b} a_fp8[k] * w_fp4[k]      (f32, the MMA's partial)
        accum  += C_local * scales_a[.., b] * scales_b[.., b] (f32)
    out = bf16(accum)                                          # out_dtype = default dtype

Inputs (all produced by the ferrite serve under test, so the comparison is valid for ANY run):
    <in-dir>/x.f32     [m*dim]  f32   the layer's MoE input
    <in-dir>/ids.i32   [m*topk] i32   the global expert ids
    <in-dir>/xq4.u8    [m*dim]  u8    ferrite's e4m3 activation codes
    <in-dir>/xsc4.f32  [m*dim/32+8] f32  ferrite's per-32 activation scales (powers of two)
    <in-dir>/gateup.f32[m*topk*act_slot] f32  the arm's raw output (for the comparison)

Weights come from the official checkpoint (true fp4: I8 packed 2 codes/byte + per-32 E8M0), via
`--ckpt` (default the HF dir our serve loads) and `--inter-rank` (the FSDP/TP8 slice of `inter`).

Usage:
  gu_numpy_ref.py --in-dir /tmp/gu_in_GD3_BS [--ckpt DIR] [--inter-rank 0] [--out /tmp/gu_numpy.f32]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

DIM, INTER_FULL, TOPK, ACT, LIM = 5120, 2304, 6, 640, 10.0


def e2m1_to_f(c):  # {0,.5,1,1.5,2,3,4,6} by magnitude, sign in bit 3
    mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    v = mag[c & 0x7]
    return np.where(c & 0x8, -v, v).astype(np.float32)


def e4m3_to_f(b):
    b = b.astype(np.uint32)
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    v = np.where(e == 0, np.ldexp(m.astype(np.float32), -9),
                 np.ldexp(1.0 + m.astype(np.float32) / 8.0, e.astype(np.int32) - 7))
    return np.where(s == 1, -v, v).astype(np.float32)


def e8m0_to_f(b):
    return np.ldexp(np.float32(1.0), b.astype(np.int32) - 127).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--ckpt", default="/opt/dlami/nvme/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--inter-rank", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=DIM,
                    help="limit K to this many elements (pair with DSV41_MOE_BS_STAGE1=1)")
    ap.add_argument("--out", default="/tmp/gu_numpy.f32")
    a = ap.parse_args()

    d = a.in_dir
    x = np.fromfile(f"{d}/x.f32", dtype="<f4")
    ids = np.fromfile(f"{d}/ids.i32", dtype="<i4")
    xq4 = np.fromfile(f"{d}/xq4.u8", dtype="u1")
    xsc4 = np.fromfile(f"{d}/xsc4.f32", dtype="<f4")
    print(f"in-dir {d}: x={x.size} ids={ids.size} xq4={xq4.size} xsc4={xsc4.size}")
    rows = x.size // DIM
    ids = ids[: rows * TOPK]
    print(f"rows(m)={rows} ids={ids.tolist()}")

    # ---- (1) our activation quantisation vs the official act_quant rules on the same x --------
    xb = x[: DIM].astype(np.float32)
    xr = xb.reshape(-1, 32)
    amax = np.abs(xr).max(axis=1)
    amax = np.maximum(amax, 1e-4)                       # kernel.py:76
    bits = amax.astype(np.float32).view(np.uint32)
    exp = ((bits >> 23) & 0xFF).astype(np.int32)
    man = bits & 0x7FFFFF
    e = exp - 127 + (man != 0)                          # fast_log2_ceil(amax * (1/448))
    xratio = np.float32(1.0 / 448.0)
    bits2 = (amax.astype(np.float32) * xratio).view(np.uint32)
    exp2 = ((bits2 >> 23) & 0xFF).astype(np.int32)
    man2 = bits2 & 0x7FFFFF
    e2 = exp2 - 127 + (man2 != 0)
    s_off = np.ldexp(np.float32(1.0), e2).astype(np.float32)
    q = np.clip(xr / s_off[:, None], -448.0, 448.0)
    codes_off = np.round(q.astype(np.float64)).astype(np.uint8)  # placeholder, replaced below
    # full e4m3 encode (RNE) via bit-level construction
    def e4m3_encode(v):
        v = np.clip(v.astype(np.float32), -448.0, 448.0)
        sgn = (v < 0).astype(np.uint32)
        av = np.abs(v).astype(np.float32)
        out = np.zeros(av.shape, dtype=np.uint8)
        az = av == 0
        nz = ~az
        if nz.any():
            ex = np.floor(np.log2(av[nz])).astype(np.int32)
            ex = np.clip(ex, -6, 8)
            scale = np.ldexp(np.float32(1.0), -3) * np.ldexp(np.float32(1.0), ex)
            mant = np.round(av[nz] / scale).astype(np.int32)   # 0..15
            carry = mant >= 16
            mant = np.where(carry, mant // 2, mant)
            ex = np.where(carry, ex + 1, ex)
            code = ((ex + 7) << 3) | mant
            code = np.clip(code, 0, 0x7E)
            out[nz] = code.astype(np.uint8)
        return (out | (sgn.astype(np.uint8) << 7)).astype(np.uint8)

    codes_off = np.zeros(xq4[: DIM].shape, dtype=np.uint8)
    qq = np.clip(xr / s_off[:, None], -448.0, 448.0)
    # Use the OFFICIAL conversion, not a hand-rolled encoder: torch's float8_e4m3fn cast is what
    # `kernel.py`'s `Cast(float8_e4m3fn, ..)` lowers to. (A hand-rolled RNE encoder here was wrong
    # in the mantissa field and produced a bogus 51% mismatch — measure with the reference, always.)
    import torch

    codes_off = (
        torch.from_numpy(qq.astype(np.float32))
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .numpy()
        .reshape(xq4[: DIM].shape)
    )
    our_codes = xq4[: DIM]
    our_s = xsc4[: DIM // 32]
    n_diff_code = int((codes_off != our_codes).sum())
    assert np.all(our_s > 0)
    print(f"[act] official-rule codes vs ferrite xq4: differing bytes = {n_diff_code}/{DIM}")
    if n_diff_code:
        idx = np.nonzero(codes_off != our_codes)[0][:8]
        print("[act] first differing (k, official, ferrite, x):",
              [(int(i), int(codes_off[i]), int(our_codes[i]), float(xb[i])) for i in idx])
    print(f"[act] official scales (2^n) vs ferrite xsc4: max rel diff = "
          f"{float(np.abs(s_off - our_s).max() / np.abs(our_s).max()):.3g}")

    # ---- (2) weights from the official checkpoint ---------------------------------------------
    idx_path = os.path.join(a.ckpt, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"] if os.path.exists(idx_path) else {}
    if not wmap:
        print(f"no safetensors index at {idx_path} — run with --ckpt pointing at the HF dir")
        return 2
    try:
        import torch
        from safetensors import safe_open
    except Exception as ex:  # noqa: BLE001
        print(f"torch/safetensors missing ({ex})"); return 2

    L = a.layer
    r0 = a.inter_rank * (INTER_FULL // 8)
    inter_shard = INTER_FULL // 8
    out = np.zeros((rows, TOPK, ACT), dtype=np.float32)
    blocks = DIM // 32
    for slot in range(TOPK):
        e = int(ids[slot])
        payload = {}
        for which in ("w1", "w3"):
            k = f"layers.{L}.ffn.experts.{e}.{which}.weight"
            ks = f"layers.{L}.ffn.experts.{e}.{which}.scale"
            if k not in wmap:
                print(f"missing key {k}"); return 2
            # framework="pt": numpy cannot represent float8_e8m0fnu, and we want the RAW bytes
            # anyway (the packed fp4 codes and the e8m0 exponent bytes) => torch + .view(uint8).
            with safe_open(os.path.join(a.ckpt, wmap[k]), framework="pt") as f:
                wt = f.get_tensor(k).view(torch.uint8).numpy()
                ws = f.get_tensor(ks).view(torch.uint8).numpy()
            payload[which] = (wt, ws)
        for half, which in ((0, "w1"), (1, "w3")):
            wt, ws = payload[which]           # [2304, 2560] packed u8 ; [2304, 160] e8m0
            seg_w = wt[r0:r0 + inter_shard]
            seg_s = ws[r0:r0 + inter_shard]
            if seg_w.shape[0] < 320:
                seg_w = np.concatenate([seg_w, np.zeros((320 - seg_w.shape[0], wt.shape[1]),
                                                        np.uint8)], axis=0)
                seg_s = np.concatenate([seg_s, np.zeros((320 - seg_s.shape[0], ws.shape[1]),
                                                        np.uint8)], axis=0)
            seg_w = seg_w[:320]
            seg_s = seg_s[:320]
            lo = seg_w & 0x0F
            hi = (seg_w >> 4) & 0x0F
            wv = np.empty((seg_w.shape[0], DIM), dtype=np.float32)
            wv[:, 0::2] = e2m1_to_f(lo.astype(np.uint8))
            wv[:, 1::2] = e2m1_to_f(hi.astype(np.uint8))
            av = e4m3_to_f(our_codes[:DIM]).astype(np.float32)
            acc = np.zeros(seg_w.shape[0], dtype=np.float32)
            kmax = max(32, min(a.kmax, DIM))
            for b in range(kmax // 32):
                part = (av[b * 32:(b + 1) * 32] * wv[:, b * 32:(b + 1) * 32]).sum(axis=1)
                acc += part * our_s[b] * e8m0_to_f(seg_s[:, b]).astype(np.float32)
            out[0, slot, half * 320:(half + 1) * 320] = acc.astype(np.float32)

    out_bf16 = out.astype(np.float32)  # the official `Linear` returns the default dtype (bf16);
    out_bf16.view(np.uint32)
    o32 = out.view(np.uint32)
    rounded = ((o32 + 0x7FFF + ((o32 >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    out.tofile(a.out)
    rounded.tofile(a.out + ".bf16.f32")
    print(f"wrote {a.out} (+ .bf16.f32) slots={TOPK} per (rank {a.inter_rank}, layer {L})")
    print("per-slot |gate|max |up|max:", [(round(float(np.abs(out[0, s, :320]).max()), 4),
                                          round(float(np.abs(out[0, s, 320:]).max()), 4))
                                         for s in range(TOPK)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
