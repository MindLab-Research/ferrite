#!/usr/bin/env python3
# =============================================================================
# dn_bs_cpu_ref.py — MoE **down** 方向的纯 CPU 数值参考 + 新旧实现对拍
# =============================================================================
# 目的（任务硬指标）：把 **SIMT 参考**（`expert_gemv_fp4_down_reduce_kernel`，生产 vec==3 臂）
# 与 **新的 blockscaled 臂**（moe_bs_dn_kernel：e4m3(block32) 激活 x e2m1 权重 + ue8m0）
# 放在同一批输入上逐元素比对，并**把差异拆成"算法差"与"fp 序差"**。
#
# 为什么不需要 GPU：两边的语义都在这里逐位建模 ——
#   * SIMT：dsv41_experts_mxf4.cu:2338-2526 的 vec==3 分支（lane 映射、fmaf 链、蝶形
#     shuffle、`__fmul_rn`/`__fadd_rn` 的 ascending-slot epilogue）逐句复刻；
#   * BS  ：e4m3(block32, e8m0) 量化（= 官方 act_quant）+ **精确积**（e4m3/e2m1 的
#     significand 只有 3/1 位 ⇒ 积在 f32 里精确）+ per-32-block 的 2 的幂标度（精确乘法）
#     + f32 累加 + epilogue 的 rounded mul（与 SIMT 同序）。
#   * e2m1/e4m3 的解码表、e8m0 字节语义、量化地板 1e-4、fast_round_scale 全部取自仓库源码。
#
# 只依赖 numpy。运行：PYTHONPATH=<numpy 路径> python3.10 dn_bs_cpu_ref.py
#
# ⚠️ 诚实声明：numpy 没有 fma，本脚本用 float64 计算 a*b+c 再舍入回 float32 模拟 fma
#    （f32 输入下 a*b 在 double 里精确；a*b+c 的双舍入误差 ~1e-16，远低于被测量）。
#    GPU 上 SIMT 核的 mul/add 是否被 ptxas 收缩成 fma 是编译器细节 ⇒ 单条 mul/add 的
#    最后一位可能与这里不同，但那属于"fp 序差"这一档，不影响结论的量级。
# =============================================================================

import numpy as np

# ---------------------------------------------------------------- 常量/几何
DIM = 5120          # down 的输出维（= dim）
INTER = 320         # down 的 contraction（= inter_local）
SLOTS = 6           # topk
KPACK = INTER // 2  # 160 packed bytes / 行
SFPITCH = 16        # w2 e8m0 面的物理行距（= dsv41_sf_pitch(320)=16）
NBLK = INTER // 32  # 10 个 scale block

f32, f64, u8, u32, i32 = np.float32, np.float64, np.uint8, np.uint32, np.int32

# e2m1 解码（dsv41_experts_mxf4.cu:728 dsv41_e2m1_to_f 的逐字表）
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=f32)


def e2m1_decode(codes):
    m = E2M1_MAG[np.asarray(codes) & 7]
    return np.where((np.asarray(codes) & 8) != 0, -m, m).astype(f32)


# e4m3 解码（dsv41_experts_mxf4.cu:742 dsv41_e4m3_to_f）
_E4M3_POS = None


def e4m3_decode(b):
    b = np.asarray(b).astype(np.int32)
    e = (b >> 3) & 0xF
    m = b & 7
    v = np.where(e == 0, m.astype(f32) * f32(1.0 / 512.0),
                 (8 + m).astype(f32) * np.ldexp(f32(1.0), e - 10))
    return np.where((b & 0x80) != 0, -v, v).astype(f32)


for _c in range(128):
    _v = e4m3_decode(np.array([_c]))[0]
    _E4M3_POS = np.append(_E4M3_POS, float(_v)) if _E4M3_POS is not None else np.array([float(_v)])
assert _E4M3_POS.shape == (128,) and np.all(np.diff(_E4M3_POS) >= 0)


def e4m3_rne(x):
    """f32 -> e4m3 字节，round-to-nearest-even + 饱和到 +-448（== __nv_fp8_e4m3）。"""
    x = np.asarray(x, dtype=f32)
    ax = np.clip(np.abs(x).astype(f64), 0.0, 448.0)
    idx = np.searchsorted(_E4M3_POS, ax, side="left")
    lo = np.clip(idx - 1, 0, 127)
    hi = np.clip(idx, 0, 127)
    dlo = ax - _E4M3_POS[lo]
    dhi = _E4M3_POS[hi] - ax
    # 平局取 mantissa 为偶（LSB==0）的一侧
    pick_hi = (dhi < dlo) | ((dhi == dlo) & ((hi & 1) == 0))
    code = np.where(pick_hi, hi, lo).astype(np.int32)
    neg = np.asarray(x) < 0
    return np.where(neg, code | 0x80, code).astype(u8)


def fast_round_scale(amax, max_inv=1.0 / 448.0):
    """glue_fast_round_scale 的逐字版 = kernel.py 的 fast_round_scale（2 的幂）。"""
    r = (np.asarray(amax, dtype=f32) * f32(max_inv)).astype(f32)
    bits = r.view(u32)
    exp = ((bits >> 23) & 0xFF).astype(np.int32)
    man = (bits & 0x7FFFFF) != 0
    e = exp - 127 + man.astype(np.int32)
    return ((e + 127).astype(u32) << 23).view(f32)


def ue8m0_byte(sc):
    e = ((np.asarray(sc, dtype=f32).view(u32) >> 23) & 0xFF).astype(np.int32) - 127
    return np.clip(e + 127, 0, 255).astype(u8)


def sf_val(byte):
    """e8m0 字节 -> f32（= 2^(byte-127)），与 SIMT 核 `__uint_as_float(b<<23)` 同语义。"""
    b = np.asarray(byte).astype(np.int32)
    return np.ldexp(np.ones(b.shape, dtype=f32), b - 127).astype(f32)


def sc_w_of(w2s_s):
    """w2 的 e8m0 面 -> 每 (行,32-block) 的 2 的幂标度（f64，精确）。"""
    e = w2s_s[:, :NBLK].astype(np.int32) - 127
    return np.exp2(e.astype(np.float64))  # [DIM, NBLK]


def fma(a, b, c):
    return (a.astype(f64) * b.astype(f64) + c.astype(f64)).astype(f32)


def fmul(a, b):
    return (a.astype(f32) * b.astype(f32)).astype(f32)


def fadd(a, b):
    return (a.astype(f32) + b.astype(f32)).astype(f32)


# ---------------------------------------------------------------- 数据生成
def make_case(seed=20260913):
    rng = np.random.default_rng(seed)
    n_e = 384
    ids = rng.choice(n_e, size=SLOTS, replace=False).astype(i32)
    # 激活 [SLOTS][pitch=2*inter]，只前 inter 个 float 被读（RAW gate‖up 布局）
    act = (rng.standard_normal((SLOTS, 2 * INTER)) * f32(0.7)).astype(f32)
    # w2 池：每个 slot 的专家一份 [DIM][160] packed fp4 + [DIM][16] e8m0
    # K=320 个 fp4 code/行 → 160 packed 字节（低 nibble = 偶 k）
    codes = rng.integers(0, 16, size=(SLOTS, DIM, INTER)).astype(np.uint8)
    packed = ((codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)) & 0xFF).astype(u8)
    assert packed.shape == (SLOTS, DIM, KPACK)
    w2s = np.zeros((SLOTS, DIM, SFPITCH), dtype=u8)
    w2s[:, :, :NBLK] = rng.integers(118, 136, size=(SLOTS, DIM, NBLK)).astype(u8)
    rw = rng.random(SLOTS).astype(f32)
    rw = (rw / rw.sum()).astype(f32)   # norm_topk_prob
    return ids, act, packed, w2s, rw


def unpack_nibbles(packed_rows):
    """packed [R, K/2] -> 元素值表 [R, K]（fp4 code）"""
    lo = (packed_rows & 0xF).astype(np.uint8)
    hi = ((packed_rows >> 4) & 0xF).astype(np.uint8)
    out = np.empty((packed_rows.shape[0], packed_rows.shape[1] * 2), dtype=np.uint8)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


# ---------------------------------------------------------------- SIMT 参考
def simt_down(act, packed, w2s, rw):
    """expert_gemv_fp4_down_reduce_kernel 的 vec==3 臂 + ascending slot epilogue（f32 逐位）。"""
    R = DIM
    tot = np.zeros(R, dtype=f32)
    lane = np.arange(32, dtype=np.int32)
    nv4 = INTER >> 7  # 2
    for s in range(SLOTS):
        brow = packed[s]      # [R,160]
        srow = w2s[s]         # [R,16]
        a0 = np.zeros((R, 32), dtype=f32)
        a1 = np.zeros((R, 32), dtype=f32)
        for g in range(nv4):
            j = (g << 7) + (lane << 2)           # 元素下标 [32]
            bidx = (g << 6) + (lane << 1)        # packed 字节下标
            sc = sf_val(srow[:, j >> 5])         # [R,32]
            w0 = brow[:, bidx]
            w1 = brow[:, bidx + 1]
            t00 = e2m1_decode(w0 & 0xF)
            t01 = e2m1_decode((w0 >> 4) & 0xF)
            t10 = e2m1_decode(w1 & 0xF)
            t11 = e2m1_decode((w1 >> 4) & 0xF)
            #   j = g*128 + lane*4  ⇒ 该 lane 的 4 个元素是 act[g*128+4l .. +4)
            av0 = act[s][(g << 7) + (lane << 2)].astype(f32)
            av1 = act[s][(g << 7) + (lane << 2) + 1].astype(f32)
            av2 = act[s][(g << 7) + (lane << 2) + 2].astype(f32)
            av3 = act[s][(g << 7) + (lane << 2) + 3].astype(f32)
            av0 = np.broadcast_to(av0, (R, 32)).astype(f32)
            av1 = np.broadcast_to(av1, (R, 32)).astype(f32)
            av2 = np.broadcast_to(av2, (R, 32)).astype(f32)
            av3 = np.broadcast_to(av3, (R, 32)).astype(f32)
            p0 = fmul(av0, t00)
            p0 = fma(av1, t01, p0)
            p1 = fmul(av2, t10)
            p1 = fma(av3, t11, p1)
            a0 = fma(sc, p0, a0)
            a1 = fma(sc, p1, a1)
        acc = fadd(a0, a1)
        # tail: j = (nv4<<7) + lane*2 = 256 + 2*lane（k=320 ⇒ 只一轮）
        jt = (nv4 << 7) + (lane << 1)
        for lane_j in range(32):
            jj = int(jt[lane_j])
            if jj >= INTER:
                continue
            byte = brow[:, (jj >> 1)]
            sc = sf_val(srow[:, (jj >> 5)])
            t = e2m1_decode(byte & 0xF)
            t2 = e2m1_decode((byte >> 4) & 0xF)
            a_j = act[s][jj].astype(f32)
            a_j1 = act[s][jj + 1].astype(f32)
            acc[:, lane_j] = fadd(acc[:, lane_j], fmul(a_j, fmul(t, sc)))
            acc[:, lane_j] = fadd(acc[:, lane_j], fmul(a_j1, fmul(t2, sc)))
        # 蝶形 shuffle（off = 16,8,4,2,1；同一条指令内全体读旧值）
        idxl = np.arange(32, dtype=np.int32)
        for off in (16, 8, 4, 2, 1):
            acc = fadd(acc, acc[:, idxl ^ off])
        tot = fadd(tot, fmul(acc[:, 0], rw[s]))
    return tot


# ---------------------------------------------------------------- BS 臂
def quantize_act_e4m3(act_slice, rw_scalar=None, bf16_boundary=False):
    """官方 act_quant(block=32, ue8m0) 的语义；返回解码后的 f32 操作数（= decode(q)*sc）。"""
    v = np.asarray(act_slice, dtype=f32).copy()
    if rw_scalar is not None:
        v = fmul(v, rw_scalar)
    if bf16_boundary:
        v = v.astype(np.float16).astype(f32)  # 占位（真 bf16 见下）
        v = (v.astype(f64).astype(f32))
    blocks = v.reshape(-1, 32)
    amax = np.maximum(np.max(np.abs(blocks), axis=1), f32(1e-4))
    sc = fast_round_scale(amax)
    q = e4m3_rne(np.clip(blocks.astype(f64) / sc.astype(f64)[:, None], -448.0, 448.0))
    deq = (e4m3_decode(q).astype(f64) * sc.astype(f64)[:, None]).astype(f32)
    return deq.reshape(-1)


def bs_down(act, packed, w2s, rw, rw_in_operand=False, bf16_boundary=False):
    """moe_bs_dn_kernel 的语义：e4m3 激活 x e2m1 权重（精确积）+ per-32-block 2 的幂标度
    + f32 累加（按 32-block 顺序）+ ascending slot + rounded mul（epilogue 或 operand 侧）。"""
    tot = np.zeros(DIM, dtype=f32)
    w_codes = unpack_nibbles(packed[0])  # 只为形状
    for s in range(SLOTS):
        # A 操作数：每 32-block 独立量化
        if rw_in_operand:
            a_deq = quant_operand(act[s][:INTER], rw[s], bf16_boundary)
        else:
            a_deq = quant_operand(act[s][:INTER])
        w_codes = unpack_nibbles(packed[s])            # [DIM, INTER] fp4 code
        w_val = e2m1_decode(w_codes).astype(f64)        # 精确
        a_val = a_deq.astype(f64).reshape(NBLK, 32)
        sc_w = sc_w_of(w2s[s])  # [DIM,NBLK] 2 的幂（精确）
        # per-block：Σ_i a_i * w_i * (2^ea * 2^ew) —— 幂次乘法精确
        acc = np.zeros(DIM, dtype=f32)
        for b in range(NBLK):
            sl = slice(b * 32, (b + 1) * 32)
            term = (w_val[:, sl] * a_val[b][None, :]).sum(axis=1)     # f64 里精确求和
            term = (term * sc_w[:, b]).astype(f32)                    # 2 的幂 ⇒ 精确
            acc = fadd(acc, term)
        if rw_in_operand:
            tot = fadd(tot, acc)
        else:
            tot = fadd(tot, fmul(acc, rw[s]))
    return tot


def bs_down64(act, packed, w2s, rw):
    """同一语义的 float64 金标准（验证 numpy 模型本身没有算错）。"""
    tot = np.zeros(DIM, dtype=f64)
    for s in range(SLOTS):
        blocks = act[s][:INTER].astype(f64).reshape(-1, 32)
        amax = np.maximum(np.max(np.abs(blocks), axis=1), 1e-4)
        e = np.ceil(np.log2(amax / 448.0))
        sc = np.exp2(e)
        q = e4m3_rne(np.clip(blocks / sc[:, None], -448.0, 448.0))
        a_val = e4m3_decode(q).astype(f64) * sc[:, None]
        w_val = e2m1_decode(unpack_nibbles(packed[s])).astype(f64)
        sc_w = sc_w_of(w2s[s])  # [DIM,NBLK] 2 的幂（精确）
        acc = np.zeros(DIM, dtype=f64)
        for b in range(NBLK):
            sl = slice(b * 32, (b + 1) * 32)
            term = (w_val[:, sl] * a_val[b][None, :]).sum(axis=1) * sc_w[:, b]
            acc += term
        tot += acc * float(rw[s])
    return tot


# ---------------------------------------------------------------- 主流程
def report(tag, ref, got):
    ref = ref.astype(f64)
    got = got.astype(f64)
    d = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-30)
    rel = d / denom
    nz = np.abs(ref) > 1e-3 * np.max(np.abs(ref))
    print(f"  {tag:<46} max|d|={d.max():.3e}  rms(d)={np.sqrt((d**2).mean()):.3e}  "
          f"max_rel={rel.max():.3e}  p99_rel={np.quantile(rel[nz], 0.99) if nz.any() else 0:.3e}  "
          f"rel_to_rms_ref={d.max()/np.sqrt((ref**2).mean()):.3e}")


def quant_operand(act_slice, rw_scalar=None, bf16_boundary=False):
    """官方 down 输入的量化操作数：返回 decode(e4m3(q)) * sc 的 f32 值（[INTER]）。
    与 `dsv41_glue.cu::routed_down_prep_kernel` 的 (1)-(5) 步逐项同式。"""
    v = np.asarray(act_slice, dtype=f32).copy()
    if rw_scalar is not None:
        v = fmul(v, rw_scalar)
    if bf16_boundary:
        # x.to(dtype)：bf16 的 RN 舍入（用 f32→bf16→f32 的位精确模拟）
        bits = v.view(u32)
        r = ((bits >> 16) & 1).astype(u32)
        rounded = (bits + np.uint32(0x7FFF) + r) & np.uint32(0xFFFF0000)
        v = rounded.view(f32)
    blocks = v.reshape(-1, 32)
    amax = np.maximum(np.max(np.abs(blocks), axis=1), f32(1e-4))
    sc = fast_round_scale(amax)
    q = e4m3_rne(np.clip(blocks.astype(f64) / sc.astype(f64)[:, None], -448.0, 448.0))
    deq = (e4m3_decode(q).astype(f64) * sc.astype(f64)[:, None]).astype(f32)
    return deq.reshape(-1)


def main():
    ids, act, packed, w2s, rw = make_case()
    print(f"[case] DIM={DIM} INTER={INTER} SLOTS={SLOTS} ids={ids.tolist()}")
    print(f"[case] |act| rms={np.sqrt((act.astype(f64)**2).mean()):.4f}   "
          f"rw={[round(float(v),4) for v in rw]}")

    # ---- 三条基线 ----
    ref = simt_down(act, packed, w2s, rw)                     # SIMT 原样（f32 激活）
    act_q = np.stack([quant_operand(act[s][:INTER]) for s in range(SLOTS)])
    act_q = np.concatenate([act_q, act[:, INTER:]], axis=1)   # 尾部半区不动（SIMT 不读）
    ref_q = simt_down(act_q, packed, w2s, rw)                 # SIMT 求和 + 量化操作数
    bs = bs_down(act, packed, w2s, rw)                        # BS 臂（量化操作数 + MMA 求和）
    bs_gold = bs_down64(act, packed, w2s, rw)

    # ---- 官方顺序（rw 进 operand + bf16 边界）----
    act_qo = np.stack([quant_operand(act[s][:INTER], rw[s], True) for s in range(SLOTS)])
    act_qo = np.concatenate([act_qo, act[:, INTER:]], axis=1)
    one = np.ones(SLOTS, dtype=f32)
    ref_qo = simt_down(act_qo, packed, w2s, one)              # SIMT 求和 + 官方操作数
    bs_o = bs_down(act, packed, w2s, rw, rw_in_operand=True, bf16_boundary=True)

    print("\n[1] 模型自检（BS 语义 f32 vs f64 金标准；量级 = 纯 f32 累加序）")
    report("BS(f32) vs BS(f64 golden)", bs_gold, bs)

    print("\n[2] 差异拆解（同输入逐元素；三档分别可归因）")
    report("(A) 纯量化差: SIMT(量化操作数) vs SIMT(f32)",
           ref, ref_q)
    report("(B) 纯 fp 序差: BS 臂 vs SIMT(量化操作数)   <= 硬指标",
           ref_q, bs)
    report("(C) 合计:      BS 臂 vs SIMT(f32)  [默认 rw 在 epilogue]",
           ref, bs)
    print("\n[3] 与官方顺序（rw 进 operand + bf16 边界）")
    report("(D) 纯序差: BS(rwop) vs SIMT(官方操作数)",
           ref_qo, bs_o)
    report("(E) 合计:   BS(rwop) vs SIMT(f32)",
           ref, bs_o)

    print("\n[4] 分布（默认臂 vs SIMT，元素级）")
    d = np.abs(bs.astype(f64) - ref.astype(f64))
    r = np.abs(ref.astype(f64))
    print(f"  |d| p50={np.quantile(d,0.5):.3e} p90={np.quantile(d,0.9):.3e} "
          f"p99={np.quantile(d,0.99):.3e} max={d.max():.3e}")
    print(f"  |ref| rms={np.sqrt((r**2).mean()):.3e} max={r.max():.3e}")
    print(f"  rms(d)/rms(ref)={np.sqrt((d**2).mean())/np.sqrt((r**2).mean()):.3e}   "
          f"max(d)/max(ref)={d.max()/r.max():.3e}")
    dq = np.abs(bs.astype(f64) - ref_q.astype(f64))
    print(f"  [B 档] max|d|={dq.max():.3e} rms={np.sqrt((dq**2).mean()):.3e} "
          f"max_rel={np.max(dq/np.maximum(np.abs(ref_q.astype(f64)),1e-30)):.3e}")


if __name__ == "__main__":
    main()
