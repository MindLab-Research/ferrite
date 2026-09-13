#!/usr/bin/env python3
# head_bf16_tilelang.py — TileLang bf16 head GEMM 原型（verify/draft 剩余块 ①）。
#
# STATUS: 原型 / 未接线。**不在 build.sh 里、不被 Rust 引用**。
#   见 docs/agent/tilelang-attn-head.md（head 数字 + attention 可行性 + verify 构成表）。
#
# 形状（**以代码事实为准，非任务书**）：
#   ferrite 的 head 是 `head.weight` = [vocab=129280, dim=5120] bf16，
#   Shard::Replicated（weights.rs:90）。verify 的 head **未切片**（每次 1.323 GB），
#   draft 路径切到 vocab/world = 16160 行（每 rank 165 MB）。
#   任务书写 "[16160, 256]" —— 见 head_bf16_tilelang 的 SHAPES 表：16160 是切片行数
#   （对），256 不是 head 的 K（K = dim = 5120）。本文件两种都跑，任务书形状在
#   `csv_literal_16160x256` 行。
#
# 数值路线（**这是本原型最重要的判读点**）：
#   ferrite 的 head 是 `dsv41_head_gemv_bf16_mrows`：w 是 bf16（解码到 f32），
#   x 是 **f32**，逐元素 f32 FMA（gemv_bf16_nt_kernel 的 WPR==1 body 转写）。即
#   真实程序是 **"bf16 权重 × f32 激活" 的 f32 乘加**。
#   tensor-core mma 没有 bf16×f32 形态 —— A 必须降到 bf16。本原型就这样做
#   （host 侧把 x cast 到 bf16），并把由此产生的数值差**量出来**（§numerics）。
#   ⚠️ head 输出喂 argmax，`draft-head-fold-v2-argmax-verdict.md` 已证 ~1e-3 的
#   重结合差就能让 33% 的 near-tie argmax 翻面 —— 所以这个 cast 不是免费的。
#
# 布局：
#   X   [MPAD, K] bf16   （MPAD=16：mma m16n8k32 的 M 约束；不足则 pad）
#   W   [N, K]    bf16   （head.weight，行主序）
#   P   [ks, MPAD, N] f32（K-split 分片）
#   C   [MPAD, N] f32
#
# 依赖：tilelang 0.1.14 / torch / B300(sm_103a)。

import sys

import numpy as np
import torch
import tilelang
import tilelang.language as T

MPAD = 16  # mma m16n8k32 的 M 约束


# ============================================================ kernel（K-split 分片）


@tilelang.jit
def head_bf16_partial(N, K, bN, ks, bK=64, threads=128, ns=3):
    """K-split 分片：grid (N/bN, ks)，每块算一段 K 的 partial，写入 P[kp]。

    head 是权重流（N×K×2 字节）主导的 GEMM：M 只有 1..6，激活 (m×K) 几乎全在 L2。
    ks 规则同投影族（tensorcore-proj-design.md §3.3）：把块数从 N/bN 抬到 (N/bN)*ks
    以填满 148 SM。
    """
    Kc = K // ks
    # bK 必须整除 Kc；小 K（任务书字面的 K=256、ks=8 ⇒ Kc=32）要自动收窄，
    # 否则 `Kc // bK == 0` ⇒ pipelined loop 空转 ⇒ 输出恒 0（静默错值）。
    bK = min(bK, Kc)
    assert Kc % bK == 0, f"bK={bK} must divide Kc={Kc}"

    @T.prim_func
    def main(X: T.Tensor((MPAD, K), "bfloat16"),
             W: T.Tensor((N, K), "bfloat16"),
             P: T.Tensor((ks, MPAD, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), ks, threads=threads) as (bx, kp):
            X_sh = T.alloc_shared((MPAD, bK), "bfloat16")
            W_sh = T.alloc_shared((bN, bK), "bfloat16")
            C_l = T.alloc_fragment((MPAD, bN), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(Kc // bK, num_stages=ns):
                T.copy(X[0, kp * Kc + ko * bK], X_sh)
                T.copy(W[bx * bN, kp * Kc + ko * bK], W_sh)
                T.gemm(X_sh, W_sh, C_l, transpose_B=True)
            T.copy(C_l, P[kp, 0, bx * bN])

    return main


@tilelang.jit
def head_bf16_reduce(ks, N, bN=256):
    """确定性归约：按 kp 升序求和 ks 个 partial。"""

    IDX = tuple(range(ks))

    @T.prim_func
    def main(P: T.Tensor((ks, MPAD, N), "float32"),
             C: T.Tensor((MPAD, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), threads=256) as bx:
            for i, j in T.Parallel(MPAD, bN):
                C[i, bx * bN + j] = sum([P[kq, i, bx * bN + j] for kq in IDX])

    return main


# ============================================================ 形状表

# (name, N, K, note)
SHAPES = [
    # 生产 draft 切片（129280/8）：v1_mrows 的 1.36ms/发 ÷ 8 ≈ 生产口径。
    ("head_slice_16160x5120", 16160, 5120, "生产切片 (vocab/world)"),
    # verify 的未切片 head（1.323 GB/发，ARSAFE 的 1.36ms 就是它）
    ("head_full_129280x5120", 129280, 5120, "verify 未切片（ARSAFE）"),
    # 任务书字面形状（16160 对，256 不是 head 的 K）
    ("csv_literal_16160x256", 16160, 256, "任务书字面（K=256）"),
]


# ============================================================ 参考 & 数值


def _mk(N, K, m, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    X32 = torch.zeros(MPAD, K, device="cuda")
    X32[:m] = torch.randn(m, K, device="cuda", generator=g) * 0.1
    W = (torch.randn(N, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    X = torch.zeros(MPAD, K, device="cuda", dtype=torch.bfloat16)
    X[:m] = X32[:m].to(torch.bfloat16)
    return X32, X, W


def _stats(name, out, truth, mask_frac=0.05):
    d = (out.double() - truth.double()).abs()
    rel = d / truth.double().abs().clamp_min(1e-30)
    mm = truth.double().abs() > (mask_frac * truth.double().abs().max())
    r = rel[mm]
    print(f"    {name:24} max_abs={d.max().item():.3e}  p50_rel={r.median().item():.3e}  "
          f"p99_rel={r.quantile(0.99).item():.3e}  max_rel={r.max().item():.3e}  "
          f"mean_rel={r.mean().item():.3e}")
    return dict(max_abs=d.max().item(), max_rel=r.max().item(), mean_rel=r.mean().item())


def selftest(bN=128, ks=8, ns=3, thr=128):
    """数值形态：TileLang bf16 mma vs (a) f64 真值 (b) ferrite f32 口径参考。"""
    print(f"== head 数值形态（bf16 mma，bN={bN} ks={ks} ns={ns} thr={thr}）==")
    for name, N, K, note in SHAPES:
        m = 6
        X32, X, W = _mk(N, K, m)
        kp = head_bf16_partial(N, K, bN, ks, threads=thr, ns=ns)
        kr = head_bf16_reduce(ks, N)
        P = torch.zeros(ks, MPAD, N, device="cuda")
        C = torch.zeros(MPAD, N, device="cuda")
        kp(X, W, P)
        kr(P, C)
        # ferrite 口径：w(f32) · x(f32)，即 head 真实程序的 f32 参考
        ref = (X32.float() @ W.float().T)[:m]
        truth = (X32.double() @ W.double().T)[:m]
        print(f"  {name} n={N} k={K}  [{note}]")
        _stats("vs f32 ferrite 口径", C[:m], ref)
        _stats("vs f64 真值", C[:m], truth)
        # 纯 bf16 激活 cast 引入的差（不换 program，只换 x 的精度）
        _stats("x-cast bf16 后 f32 参考", (X.float()[:m] @ W.float().T), ref)
    return True


# ============================================================ benchmark


def _bench(fn, iters=1000, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


@tilelang.jit
def empty_kernel():
    @T.prim_func
    def main(X: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32) as bx:
            X[bx] = X[bx] + 0.0

    return main


def benchmark(bN=128, ks=8, ns=3, thr=128, bK=64, iters=1000):
    print(f"== head benchmark TileLang bf16（bN={bN} ks={ks} ns={ns} thr={thr} bK={bK} iters={iters}）==")
    ek = empty_kernel()
    xe = torch.zeros(16, device="cuda")
    floor = _bench(lambda: ek(xe), iters)
    print(f"  launch floor (empty kernel) = {floor:.2f} us")
    print(f"  {'shape':24} {'n':>6} {'k':>5} | {'p1':>8} {'p6':>8} {'p6/p1':>6} "
          f"| {'r1':>8} {'r6':>8} {'r6/r1':>6} | {'wMB':>7} {'TB/s(M6)':>9}")
    out = {}
    for name, N, K, note in SHAPES:
        res = {}
        for m in (1, 6):
            X32, X, W = _mk(N, K, m)
            kp = head_bf16_partial(N, K, bN, ks, threads=thr, ns=ns, bK=bK)
            kr = head_bf16_reduce(ks, N)
            P = torch.zeros(ks, MPAD, N, device="cuda")
            C = torch.zeros(MPAD, N, device="cuda")
            res[m] = (_bench(lambda: kp(X, W, P), iters),
                      _bench(lambda: (kp(X, W, P), kr(P, C)), iters))
        p1, r1 = res[1]
        p6, r6 = res[6]
        wmb = N * K * 2 / 1048576.0
        tbs = wmb / (r6 - floor) if r6 > floor else float("nan")  # MB/us == TB/s
        out[name] = dict(p1=p1, p6=p6, r1=r1, r6=r6, wmb=wmb, tbs=tbs)
        print(f"  {name:24} {N:>6} {K:>5} | {p1:8.2f} {p6:8.2f} {p6/p1:6.2f} "
              f"| {r1:8.2f} {r6:8.2f} {r6/r1:6.2f} | {wmb:7.1f} {tbs:9.2f}")
    return floor, out


def sweep(iters=300):
    """config 搜索：在 slice 形状上找最快 partial（M=6），并报 full 形状的最佳 config。"""
    ek = empty_kernel()
    xe = torch.zeros(16, device="cuda")
    floor = _bench(lambda: ek(xe), iters)
    print(f"== config sweep（M=6 partial-only，iters={iters}，floor={floor:.2f}us）==")
    cfgs = []
    for bN in (64, 128, 256):
        for ks in (4, 8, 16):
            for bK in (32, 64, 128):
                for ns in (2, 3):
                    for thr in (128, 256):
                        cfgs.append((bN, ks, bK, ns, thr))
    for name, N, K, note in SHAPES[:2]:
        X32, X, W = _mk(N, K, 6)
        wmb = N * K * 2 / 1048576.0
        rows = []
        for (bN, ks, bK, ns, thr) in cfgs:
            Kc = K // ks
            if Kc % bK != 0:
                continue
            try:
                kp = head_bf16_partial(N, K, bN, ks, threads=thr, ns=ns, bK=bK)
            except Exception:
                continue
            P = torch.zeros(ks, MPAD, N, device="cuda")
            t = _bench(lambda: kp(X, W, P), iters, warmup=10)
            rows.append((t, bN, ks, bK, ns, thr))
        rows.sort()
        print(f"  {name} n={N} k={K}  ({len(rows)} cfgs tried)")
        for (t, bN, ks, bK, ns, thr) in rows[:6]:
            print(f"    {t:8.2f} us  bN={bN:3d} ks={ks:2d} bK={bK:3d} ns={ns} thr={thr}  "
                  f"({wmb / (t - floor):.2f} TB/s)")
    return floor


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        selftest()
    elif what == "bench":
        benchmark()
    elif what == "sweep":
        sweep()
    elif what == "all":
        selftest()
        benchmark()
    else:
        print(f"usage: {sys.argv[0]} [selftest|bench|all]")
