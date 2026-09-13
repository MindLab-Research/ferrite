#!/usr/bin/env python3
# proj_fp8_tilelang_phase2.py — TileLang fp8 投影族 GEMM 原型（第二阶段）。
#
# STATUS: 原型 / 未接线。**不在 build.sh 里、不被 Rust 引用**。
#   见 docs/agent/tilelang-proj-phase2.md（五形状 benchmark + wo_a 分组形态 + 数值形态）。
#
# 第一阶段（`proj_fp8_tilelang.py` / docs/agent/tilelang-proj-proto.md）交付了 route A：
#   原生 fp8 mma + per-32-K-block scale **在 epilogue 里直读全局**，四形状 M6/M1 = 0.98–1.03。
# 本阶段把它扩到**完整投影族**：
#   ① 五形状：wkv / wq_a / wq_b / wo_b（稠密）+ wo_a（分组）  ← 本文件
#   ② wo_a 分组形态：group 维进 grid（blockIdx.y），激活按 a_stride 跨组分段，
#      输出按 out_stride 跨组写列 —— 复现 `dsv41_wo_a_grouped_fp8` 的 ABI。
#   ③ per-32 epilogue 直读的通用性：跨 num_stages / 重复运行的**确定性**扫描。
#   ④ benchmark（M1/M6 + M6/M1）与数值形态（per 形状 maxrel/meanrel）。
#
# 布局（**以 ferrite `gemm_fp8_mrows_kernel` 的 ABI 为准，NOT per-128**）：
#   a        [m, k]        fp8 e4m3     （本原型 pad 到 MPAD=16 —— mma m16n8k32 的 M 约束）
#   a_scale  [m, k/32]     f32
#   w        [n, k]        fp8 e4m3
#   w_scale  [n/32, k/32]  ue8m0（块 32x32）
# wo_a 分组（`dsv41_wo_a_grouped_fp8` 的 ABI，见 kernels/cuda/dsv41_kernels.cu:8165）：
#   a        [m, a_stride] fp8，第 g 组段在 +g*k 字节处（block-diagonal）
#   a_scale  [m, a_stride/32] f32，第 g 组段在 +g*k/32
#   w        [G, n, k]      fp8，第 g 组在 +g*n*k
#   w_scale  [G, n/32, k/32] ue8m0
#   out      [m, G*n]       f32，第 g 组列在 +g*n（= `out[(r*out_stride) + g*n + row]`）
#
# 数值路线：**route A**（原生 fp8 mma + per-32-K-block scale 作用在 mma 输出上），
#   镜 `gemm_fp8_swapab_kernel:800-813`。实测比 f32 SIMT 参考更接近真值（proto.md §4）。
#   route B（dequant bf16）数值不可用（mean_rel ~1.7e-2），本文件不提供。
#
# wo_a 的**量化**在 shim 侧（host/前一 kernel）完成 —— 见 phase2.md 的形态决策：
#   本文件只吃 (fp8 激活 + f32 per-32 scale)，与 ferrite `quant_fp8` 的产物逐位一致
#   （`quant_ref()` 复刻 `quant_kernel` 的 round_scale=true 算术，用于数值对照）。
#
# 依赖：tilelang 0.1.14 / torch / B300(sm_103a)。

import struct
import sys

import torch
import tilelang
import tilelang.language as T

FP8 = "float8_e4m3fn"
E8M0 = "float8_e8m0fnu"
MPAD = 16  # mma m16n8k32 要求 M 能被 16 整除；激活不足则 pad

# ============================================================ 稠密四形状
# wkv(512,5120) / wq_a(1280,5120) / wq_b(4096,1280) / wo_b(5120,1024)


@tilelang.jit
def proj_fp8_partial(N, K, bN, ks, threads=128, ns=3):
    """K-split 分片：grid (N/bN, ks)，每块算一段 K 的 partial，写入 P[kp]。

    ks 规则（mirror tensorcore-proj-design.md §3.3）：把块数从 N/bN 抬到 (N/bN)*ks 以填满 SM。
    """
    Kc = K // ks

    @T.prim_func
    def main(A: T.Tensor((MPAD, K), FP8),
             ASC: T.Tensor((MPAD, K // 32), "float32"),
             W: T.Tensor((N, K), FP8),
             WSC: T.Tensor((N // 32, K // 32), E8M0),
             P: T.Tensor((ks, MPAD, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), ks, threads=threads) as (bx, kp):
            A_sh = T.alloc_shared((MPAD, 32), FP8)
            W_sh = T.alloc_shared((bN, 32), FP8)
            C_l = T.alloc_fragment((MPAD, bN), "float32")
            C_p = T.alloc_fragment((MPAD, bN), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(Kc // 32, num_stages=ns):
                gko = kp * (Kc // 32) + ko
                T.copy(A[0, gko * 32], A_sh)
                T.copy(W[bx * bN, gko * 32], W_sh)
                # raw fp8 乘积和（mma），再按 32-K 块的 scale 累加。
                # ⚠️ scale 直读全局（不要自建共享缓冲——pipelined loop 不给它多缓冲，
                #    会产生跨 stage 覆盖的静默错值；见 proto.md §3.1 踩坑）。
                T.gemm(A_sh, W_sh, C_p, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(MPAD, bN):
                    C_l[i, j] += C_p[i, j] * ASC[i, gko] * T.cast(
                        WSC[bx * (bN // 32) + j // 32, gko], "float32")
            T.copy(C_l, P[kp, 0, bx * bN])

    return main


@tilelang.jit
def proj_fp8_reduce(ks, N, bN=256):
    """确定性归约：按 kp 升序求和 ks 个 partial。"""

    IDX = tuple(range(ks))

    @T.prim_func
    def main(P: T.Tensor((ks, MPAD, N), "float32"),
             C: T.Tensor((MPAD, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), threads=256) as bx:
            for i, j in T.Parallel(MPAD, bN):
                C[i, bx * bN + j] = sum([P[kq, i, bx * bN + j] for kq in IDX])

    return main


# ============================================================ wo_a 分组
# ferrite `dsv41_wo_a_grouped_fp8`：block-diagonal 分组低秩输出投影。
#   G = nlg = o_groups/world（verify@TP8 时 =1；draft@TP1 时 =8）
#   n = o_lora_rank（每组输出宽），k = hpg*head_dim（每组 K）
#   a_stride = nlh*head_dim = G*k（本 rank 的 attention 输出行距）


@tilelang.jit
def wo_a_grouped_partial(G, N, K, ASTRIDE, bN, ks, threads=128, ns=3):
    """分组 K-split 分片：grid (N/bN, G, ks)。

    group 维走 **blockIdx.y**（不是循环）—— 每个 block 只服务一个组，
    组与组之间零共享（与老 kernel 的 `blockIdx.y = g` 同形）。
    """
    Kc = K // ks
    KELEM = K // 32

    @T.prim_func
    def main(A: T.Tensor((MPAD, ASTRIDE), FP8),
             ASC: T.Tensor((MPAD, ASTRIDE // 32), "float32"),
             W: T.Tensor((G, N, K), FP8),
             WSC: T.Tensor((G, N // 32, K // 32), E8M0),
             P: T.Tensor((ks, G, MPAD, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), G, ks, threads=threads) as (bx, g, kp):
            A_sh = T.alloc_shared((MPAD, 32), FP8)
            W_sh = T.alloc_shared((bN, 32), FP8)
            C_l = T.alloc_fragment((MPAD, bN), "float32")
            C_p = T.alloc_fragment((MPAD, bN), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(Kc // 32, num_stages=ns):
                gko = kp * (Kc // 32) + ko
                # 第 g 组的激活段在 +g*K 字节（block-diagonal）；行距是 ASTRIDE（≠ K）。
                T.copy(A[0, g * K + gko * 32], A_sh)
                T.copy(W[g, bx * bN, gko * 32], W_sh)
                T.gemm(A_sh, W_sh, C_p, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(MPAD, bN):
                    C_l[i, j] += C_p[i, j] * ASC[i, g * KELEM + gko] * T.cast(
                        WSC[g, bx * (bN // 32) + j // 32, gko], "float32")
            T.copy(C_l, P[kp, g, 0, bx * bN])

    return main


@tilelang.jit
def wo_a_grouped_reduce(ks, G, N, bN=256):
    """分组确定性归约：组 g 的 ks 个 partial 归到 OUT[:, g*N + ...]。"""

    IDX = tuple(range(ks))

    @T.prim_func
    def main(P: T.Tensor((ks, G, MPAD, N), "float32"),
             OUT: T.Tensor((MPAD, G * N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), G, threads=256) as (bx, g):
            for i, j in T.Parallel(MPAD, bN):
                OUT[i, g * N + bx * bN + j] = sum(
                    [P[kq, g, i, bx * bN + j] for kq in IDX])

    return main


# ============================================================ 参考 & 数值


def _reference(A, ASC, W, WSC):
    """ferrite 表达式 `(a·as) @ (w·ws)^T` 的 f32 torch 实现（非真值，是"ferrite 口径"）。"""
    af = A.float()
    wf = W.float()
    asp = ASC.repeat_interleave(32, dim=1)
    wsp = WSC.float().repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    return (af * asp) @ (wf * wsp).T


def _reference_grouped(A, ASC, W, WSC, G, K):
    """分组参考：第 g 组 = A[:, g*K:(g+1)*K] × W[g]^T。"""
    cols = []
    for g in range(G):
        ag = A[:, g * K:(g + 1) * K]
        asg = ASC[:, g * (K // 32):(g + 1) * (K // 32)]
        cols.append(_reference(ag, asg, W[g], WSC[g]))
    return torch.cat(cols, dim=1)


def _fast_round_scale(amax, max_inv):
    """复刻 dsv41_kernels.cu:113 的 fast_round_scale（e4m3: max_inv = 1/448）。"""
    bits = struct.unpack("<I", struct.pack("<f", amax * max_inv))[0]
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    e = exp - 127 + (1 if man != 0 else 0)
    return struct.unpack("<f", struct.pack("<I", ((e + 127) << 23) & 0xFFFFFFFF))[0]


def quant_ref(x, block=32):
    """复刻 ferrite `quant_kernel<0>` 的 round_scale=1 路径（maxv=448, e4m3）。

    这是 **shim 侧量化**的参照实现：wo_a 的 TileLang 版只吃它的产物（fp8 + per-32 f32 scale）。
    """
    rows, cols = x.shape
    nb = cols // block
    xb = x.view(rows, nb, block)
    amax = xb.abs().amax(dim=2)                       # per-block amax（fmaxf 结合律无关）
    sc = torch.tensor([[max(_fast_round_scale(float(a), 1.0 / 448.0), 1e-30)
                        for a in row] for row in amax.tolist()],
                      device=x.device, dtype=torch.float32)
    v = (xb / sc.unsqueeze(-1)).clamp(-448.0, 448.0)
    return v.reshape(rows, cols).to(torch.float8_e4m3fn), sc


def _stats(name, out, truth, mask_frac=0.05):
    """与 proto.md §4 同口径：|truth| > frac·max 的元素上统计相对误差。"""
    d = (out.double() - truth.double()).abs()
    rel = d / truth.double().abs().clamp_min(1e-30)
    m = truth.double().abs() > (mask_frac * truth.double().abs().max())
    r = rel[m]
    print(f"    {name:10} max_abs={d.max().item():.3e}  p50_rel={r.median().item():.3e}  "
          f"p99_rel={r.quantile(0.99).item():.3e}  max_rel={r.max().item():.3e}  "
          f"mean_rel={r.mean().item():.3e}")
    return dict(max_abs=d.max().item(), max_rel=r.max().item(), mean_rel=r.mean().item())


# ============================================================ 形状表

# 稠密四形状：(name, N, K)
DENSE = [("wkv", 512, 5120), ("wq_a", 1280, 5120),
         ("wq_b", 4096, 1280), ("wo_b", 5120, 1024)]

# wo_a 分组（**代码事实**，见 kernels/cuda/dsv41_kernels.cu:8165 与 chain_dev.rs:13270）：
#   G=nlg(o_groups/world) / N=o_lora_rank=1024 / K=hpg*head_dim=4096 / ASTRIDE=nlh*head_dim
#   verify@TP8(world=8): G=1, ASTRIDE=4096 (== K)
#   draft @TP1(world=1): G=8, ASTRIDE=32768
WOA = [("wo_a_nlg1", 1, 1024, 4096, 4096),
       ("wo_a_nlg8", 8, 1024, 4096, 32768)]


def _mk_dense_inputs(N, K, m, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.zeros(MPAD, K, device="cuda")
    A[:m] = torch.rand(m, K, device="cuda", generator=g) * 2 - 1
    A = A.to(torch.float8_e4m3fn)
    W = (torch.rand(N, K, device="cuda", generator=g) * 2 - 1).to(torch.float8_e4m3fn)
    ASC = torch.zeros(MPAD, K // 32, device="cuda")
    ASC[:m] = torch.rand(m, K // 32, device="cuda", generator=g) * 0.5 + 0.75
    WSC = (torch.randint(123, 131, (N // 32, K // 32), device="cuda", generator=g)
           .to(torch.uint8).view(torch.float8_e8m0fnu))
    return A, ASC, W, WSC


def _mk_woa_inputs(G, N, K, ASTRIDE, m, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # f32 激活行 → shim 侧量化的真输入（每行 ASTRIDE 元素）
    A32 = torch.zeros(MPAD, ASTRIDE, device="cuda")
    A32[:m] = torch.rand(m, ASTRIDE, device="cuda", generator=g) * 2 - 1
    aq, asc_block = quant_ref(A32[:m])          # (m, ASTRIDE) fp8 + (m, ASTRIDE/32) f32
    A = torch.zeros(MPAD, ASTRIDE, device="cuda").to(torch.float8_e4m3fn)
    A[:m] = aq
    ASC = torch.zeros(MPAD, ASTRIDE // 32, device="cuda")
    ASC[:m] = asc_block
    W = (torch.rand(G, N, K, device="cuda", generator=g) * 2 - 1).to(torch.float8_e4m3fn)
    WSC = (torch.randint(123, 131, (G, N // 32, K // 32), device="cuda", generator=g)
           .to(torch.uint8).view(torch.float8_e8m0fnu))
    return A, ASC, W, WSC


# ============================================================ 自测（数值形态）


def selftest(bN=128, ks=8, ns=3, thr=128, verbose=True):
    """五形状正确性 + route A 的数值形态（vs f64 真值 & vs f32 SIMT 参考）。"""
    print("== 五形状正确性 / 数值形态（route A，bN=%d ks=%d ns=%d thr=%d）==" % (bN, ks, ns, thr))
    out = {}
    for name, N, K in DENSE:
        m = 6
        A, ASC, W, WSC = _mk_dense_inputs(N, K, m)
        kp = proj_fp8_partial(N, K, bN, ks, thr, ns)
        kr = proj_fp8_reduce(ks, N)
        P = torch.zeros(ks, MPAD, N, device="cuda")
        C = torch.zeros(MPAD, N, device="cuda")
        kp(A, ASC, W, WSC, P)
        kr(P, C)
        ref = _reference(A, ASC, W, WSC)[:m]
        truth = _reference(
            A.float(), ASC, W.float(), WSC)[:m].double()
        err = (C[:m] - ref).abs().max().item()
        print(f"  {name:6} n={N:<5} k={K:<5} |C-ref|max={err:.3e}")
        out[name] = _stats(name, C[:m], truth)
    for name, G, N, K, ASTRIDE in WOA:
        m = 6
        A, ASC, W, WSC = _mk_woa_inputs(G, N, K, ASTRIDE, m)
        kp = wo_a_grouped_partial(G, N, K, ASTRIDE, bN, ks, thr, ns)
        kr = wo_a_grouped_reduce(ks, G, N)
        P = torch.zeros(ks, G, MPAD, N, device="cuda")
        C = torch.zeros(MPAD, G * N, device="cuda")
        kp(A, ASC, W, WSC, P)
        kr(P, C)
        ref = _reference_grouped(A, ASC, W, WSC, G, K)[:m]
        truth = _reference_grouped(A.float(), ASC, W.float(), WSC, G, K)[:m].double()
        err = (C[:m] - ref).abs().max().item()
        print(f"  {name:10} G={G} n={N} k={K} a_stride={ASTRIDE} |C-ref|max={err:.3e}")
        out[name] = _stats(name, C[:m], truth)
    return out


def determinism(bN=128, ks=8, thr=128):
    """③ per-32 epilogue 直读的通用性：跨 num_stages / 重复运行的**逐位确定性**。

    proto.md §3.1 的静默错值（自建 scale 共享缓冲）在 ns>1 时才暴露 —— 这里对
    **五形状 × ns∈{1,2,3,4}** 各跑 3 次，要求所有结果与 ns=1 的第一次逐位相同。
    """
    print("== ③ epilogue 直读：跨 num_stages / rep 的逐位确定性 ==")
    allok = True
    for name, N, K in DENSE:
        m = 6
        A, ASC, W, WSC = _mk_dense_inputs(N, K, m)
        baseline = None
        row = []
        for ns in (1, 2, 3, 4):
            kp = proj_fp8_partial(N, K, bN, ks, thr, ns)
            kr = proj_fp8_reduce(ks, N)
            P = torch.zeros(ks, MPAD, N, device="cuda")
            C = torch.zeros(MPAD, N, device="cuda")
            sigs = []
            for _ in range(3):
                P.zero_(); C.zero_()
                kp(A, ASC, W, WSC, P)
                kr(P, C)
                sigs.append(C[:m].clone())
            same = all(torch.equal(sigs[0], s) for s in sigs[1:])
            if baseline is None:
                baseline = sigs[0]
                refok = True
            else:
                refok = torch.equal(baseline, sigs[0])
            row.append(f"ns={ns}:{'ok' if same and refok else 'MISMATCH'}")
            allok &= same and refok
        print(f"  {name:6} " + "  ".join(row))
    for name, G, N, K, ASTRIDE in WOA:
        m = 6
        A, ASC, W, WSC = _mk_woa_inputs(G, N, K, ASTRIDE, m)
        baseline = None
        row = []
        for ns in (1, 2, 3, 4):
            kp = wo_a_grouped_partial(G, N, K, ASTRIDE, bN, ks, thr, ns)
            kr = wo_a_grouped_reduce(ks, G, N)
            P = torch.zeros(ks, G, MPAD, N, device="cuda")
            C = torch.zeros(MPAD, G * N, device="cuda")
            sigs = []
            for _ in range(3):
                P.zero_(); C.zero_()
                kp(A, ASC, W, WSC, P)
                kr(P, C)
                sigs.append(C[:m].clone())
            same = all(torch.equal(sigs[0], s) for s in sigs[1:])
            if baseline is None:
                baseline = sigs[0]
                refok = True
            else:
                refok = torch.equal(baseline, sigs[0])
            row.append(f"ns={ns}:{'ok' if same and refok else 'MISMATCH'}")
            allok &= same and refok
        print(f"  {name:10} " + "  ".join(row))
    print("  => " + ("ALL BIT-IDENTICAL ✅" if allok else "DETERMINISM FAIL ❌"))
    return allok


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
    """launch 底噪校准：空 kernel（同 event 口径）。"""

    @T.prim_func
    def main(X: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32) as bx:
            X[bx] = X[bx] + 0.0

    return main


def benchmark(bN=128, ks=8, ns=3, thr=128, iters=1000):
    """五形状 × M=1/6 × (partial-only / partial+reduce)。"""
    print(f"== benchmark TileLang route A（bN={bN} ks={ks} ns={ns} thr={thr} iters={iters}）==")
    # launch 底噪
    ek = empty_kernel()
    xe = torch.zeros(16, device="cuda")
    floor = _bench(lambda: ek(xe), iters)
    print(f"  launch floor (empty kernel) = {floor:.2f} us")
    rows = []
    for name, N, K in DENSE:
        res = {}
        for m in (1, 6):
            A, ASC, W, WSC = _mk_dense_inputs(N, K, m)
            kp = proj_fp8_partial(N, K, bN, ks, thr, ns)
            kr = proj_fp8_reduce(ks, N)
            P = torch.zeros(ks, MPAD, N, device="cuda")
            C = torch.zeros(MPAD, N, device="cuda")
            res[m] = (_bench(lambda: kp(A, ASC, W, WSC, P), iters),
                      _bench(lambda: (kp(A, ASC, W, WSC, P), kr(P, C)), iters))
        rows.append((name, N, K, res))
    for name, G, N, K, ASTRIDE in WOA:
        res = {}
        for m in (1, 6):
            A, ASC, W, WSC = _mk_woa_inputs(G, N, K, ASTRIDE, m)
            kp = wo_a_grouped_partial(G, N, K, ASTRIDE, bN, ks, thr, ns)
            kr = wo_a_grouped_reduce(ks, G, N)
            P = torch.zeros(ks, G, MPAD, N, device="cuda")
            C = torch.zeros(MPAD, G * N, device="cuda")
            res[m] = (_bench(lambda: kp(A, ASC, W, WSC, P), iters),
                      _bench(lambda: (kp(A, ASC, W, WSC, P), kr(P, C)), iters))
        rows.append((name, G * N, K, res))
    print(f"  {'shape':10} {'n':>5} {'k':>5} | {'p1':>7} {'p6':>7} {'p6/p1':>6} "
          f"| {'r1':>7} {'r6':>7} {'r6/r1':>6} | {'p1n':>6} {'p6n':>6}")
    for name, N, K, res in rows:
        p1, r1 = res[1]
        p6, r6 = res[6]
        print(f"  {name:10} {N:>5} {K:>5} | {p1:7.2f} {p6:7.2f} {p6/p1:6.2f} "
              f"| {r1:7.2f} {r6:7.2f} {r6/r1:6.2f} | {p1-floor:6.2f} {p6-floor:6.2f}")
    return floor


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        selftest()
    elif what == "det":
        selftest()
        determinism()
    elif what == "bench":
        benchmark()
    elif what == "all":
        selftest()
        determinism()
        benchmark()
    else:
        print(f"usage: {sys.argv[0]} [selftest|det|bench|all]")
