#!/usr/bin/env python3
# apply_tilelang_patch.py — 把 vendored 的 TileLang 源码补丁打到**已安装的 tilelang
# 包**上（site-packages / venv 都行），幂等、可审计、可 `--check-only`。
#
# 为什么需要它：`T.tcgen05_gemm_blockscaled` 在 0.1.14（以及截至 main 的上游）漏写
# `ann["is_tcgen05"] = 1`，导致 `cuda::Gemm::SelectInst` 落进 SM120 NVF4 分支并硬失败
#
#     InternalError: T.mma_gemm_blockscaled() requires an SM120 CUDA target,
#         but got target={..., "arch":"sm_103a"}
#
# 正确修法是**改库的源码**（哪个进程 import 都拿到修好的版本），不是运行时
# monkey-patch、不是 re-exec 注入。本脚本就是那条路径的唯一入口：
#
#   1) 定位 `tilelang/language/gemm_op.py`（`import tilelang` 或 `--tilelang-path`）；
#   2) **AST 检查**该文件里 `tcgen05_gemm_blockscaled` 是否已带 `is_tcgen05` 注解
#      —— 已带 ⇒ 什么都不做（幂等），退出 0；
#   3) 打补丁：源文件 sha256 == 基准 ⇒ 用 `patch -p1`（逐字节精确）；
#      不等（不同小版本 / 已被别的补丁动过）⇒ 退化为**锚点插入**（锚点唯一性由 AST
#      定位的函数体 + 全文计数共同保证），并在输出里说明用了哪条路；
#   4) 写完**重新 AST 复核**：目标函数体里必须恰好出现一次 `ann["is_tcgen05"]`，
#      否则退出 1（不允许"以为打上了"）。
#
# 用法：
#   python3 kernels/tilelang/vendor/apply_tilelang_patch.py --check-only
#   python3 kernels/tilelang/vendor/apply_tilelang_patch.py
#   /opt/dlami/nvme/dsv41_venv/bin/python kernels/tilelang/vendor/apply_tilelang_patch.py
#
# 退出码：0 = 已修复（本次打的 / 本来就修好的）；1 = 失败（信息在 stderr）。
import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(
    HERE, "tilelang-0.1.14", "tcgen05-blockscaled-is_tcgen05.patch"
)

# 上游 v0.1.14 tag 的 `tilelang/language/gemm_op.py` 原始 sha256（627 行）。
PRISTINE_SHA256 = "c775813d5635e39a81df4b45a074f890895032f54d8b696b94b016d50e6426fa"
PRISTINE_LINES = 627

FN = "tcgen05_gemm_blockscaled"
ANNOT = "is_tcgen05"
ANCHOR = '    ann["sf_b_granularity_k"] = int(sf_b_granularity_k)'
INSERT = (
    "\n"
    "    # Request the TCGEN05 lowering explicitly. `T.tcgen05_gemm()` sets this\n"
    "    # annotation; this entry point must too, otherwise `cuda::Gemm::SelectInst`\n"
    "    # never sees `isTcgen05_` and falls through to the \"SFA/SFB are present =>\n"
    "    # NVF4 mma.sync\" branch, which hard-fails with\n"
    "    #   \"T.mma_gemm_blockscaled() requires an SM120 CUDA target\"\n"
    "    # on every Blackwell tcgen05 target (sm_100a / sm_103a).\n"
    '    ann["is_tcgen05"] = 1\n'
)


def find_gemm_op(explicit: str | None) -> str:
    """`tilelang/language/gemm_op.py` 的路径。"""
    if explicit:
        p = os.path.abspath(explicit)
        if os.path.isfile(p):
            return p
        p = os.path.join(p, "language", "gemm_op.py")
        if os.path.isfile(p):
            return p
        sys.exit(f"[fail] no gemm_op.py under --tilelang-path={explicit}")
    try:
        import tilelang  # noqa: PLC0415 — 探测，故意延迟
    except ImportError as e:
        sys.exit(
            f"[fail] cannot import tilelang from {sys.executable}: {e}\n"
            "       pass --tilelang-path <dir> (e.g. ~/.local/lib/python3.12/"
            "site-packages/tilelang) or run this with the interpreter that owns it."
        )
    p = os.path.join(os.path.dirname(os.path.abspath(tilelang.__file__)), "language", "gemm_op.py")
    if not os.path.isfile(p):
        sys.exit(f"[fail] tilelang at {tilelang.__file__} has no language/gemm_op.py")
    return p


def fn_state(src: str) -> str:
    """AST 判定目标函数里的注解状态。

    返回 `"fixed"` / `"missing"`；函数不存在 ⇒ `"no-fn"`（上游改了结构，必须重新推导，
    不允许猜）。用 AST 而不是字符串搜索：`is_tcgen05` 在 `tcgen05_gemm()` 里也有，
    全文 grep 会把「隔壁函数有」误判成「这个函数有」——那正是这个 bug 的形状。
    """
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == FN), None
    )
    if fn is None:
        return "no-fn"
    for node in ast.walk(fn):
        # `ann["is_tcgen05"] = 1`
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if (
            isinstance(tgt, ast.Subscript)
            and isinstance(tgt.slice, ast.Constant)
            and tgt.slice.value == ANNOT
        ):
            return "fixed"
    return "missing"


def run_patch(path: str, dry: bool) -> tuple[bool, str]:
    """用 `patch -p1` 打 vendored diff（从补丁文件的第一个 `--- a/` 起）。"""
    if shutil.which("patch") is None:
        return False, "the `patch` binary is not installed"
    with open(PATCH) as f:
        lines = f.readlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("--- a/")), None)
    if start is None:
        return False, f"{PATCH} has no unified-diff body"
    body = "".join(lines[start:])
    # diff 的路径是 `a/tilelang/language/gemm_op.py`，`-p1` 去掉 `a/` ⇒ cwd 必须是
    # **包目录的父目录**（site-packages / venv 的 lib/pythonX.Y/site-packages）。
    root = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    argv = ["patch", "-p1", "--forward"]
    if dry:
        argv.append("--dry-run")
    r = subprocess.run(
        argv, cwd=root, input=body, text=True, capture_output=True
    )
    out = (r.stdout + r.stderr).strip().replace("\n", " | ")
    return r.returncode == 0, out


def main() -> int:
    ap = argparse.ArgumentParser(description="vendored TileLang tcgen05-blockscaled fix")
    ap.add_argument("--tilelang-path", default=None, help="tilelang 包目录（默认 import 定位）")
    ap.add_argument("--check-only", action="store_true", help="只检查，不改文件（AOT 前置校验用）")
    ap.add_argument(
        "--require-pristine",
        action="store_true",
        help="源文件 sha256 不等于 v0.1.14 基准就失败（想锁定版本时用）",
    )
    args = ap.parse_args()

    path = find_gemm_op(args.tilelang_path)
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    src = raw.decode()
    lines = raw.count(b"\n")

    state = fn_state(src)
    print(f"[vendor] gemm_op.py      : {path}")
    print(f"[vendor] sha256           : {sha}")
    print(f"[vendor] lines            : {lines}" + ("" if lines == PRISTINE_LINES else
                                                     f"  (pristine v0.1.14 = {PRISTINE_LINES})"))
    print(f"[vendor] {FN}.{ANNOT} : {state}")

    if state == "no-fn":
        print(
            f"[fail] {FN}() is gone from this gemm_op.py — the vendored patch cannot be\n"
            "       verified. Re-derive it against the installed version (see "
            "kernels/tilelang/vendor/README.md §3) instead of guessing.",
            file=sys.stderr,
        )
        return 1
    if state == "fixed":
        print("[vendor] already fixed — nothing to do (idempotent)")
        return 0
    if args.check_only:
        print(
            "[fail] the installed tilelang is UNFIXED. Run without --check-only:\n"
            f"       {sys.executable} {os.path.abspath(__file__)}",
            file=sys.stderr,
        )
        return 1
    if sha != PRISTINE_SHA256:
        msg = (
            f"[warn] sha256 != pristine v0.1.14 ({PRISTINE_SHA256[:16]}…) — this tree has been\n"
            "       modified (another patch, or a different revision)."
        )
        if args.require_pristine:
            print(msg, file=sys.stderr)
            return 1
        print(msg)
        print("[warn] falling back to anchored insertion (patch -p1 skipped)")

    method = "patch -p1"
    if sha == PRISTINE_SHA256:
        ok, out = run_patch(path, dry=False)
        print(f"[vendor] patch -p1        : {'OK' if ok else 'FAILED'}  {out}")
        if not ok:
            print("[warn] `patch` failed — falling back to anchored insertion")
            method = "anchor"
    else:
        method = "anchor"

    if method == "anchor":
        # 锚点唯一性：全文恰好一次，且落在目标函数体内（AST 已确认函数存在）。
        if src.count(ANCHOR) != 1:
            print(
                f"[fail] the injection anchor occurs {src.count(ANCHOR)} times (need 1):\n"
                f"           {ANCHOR}\n"
                "       gemm_op.py changed; re-derive the vendored patch.",
                file=sys.stderr,
            )
            return 1
        tree = ast.parse(src)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == FN
        )
        lo, hi = fn.lineno, fn.end_lineno
        anchor_line = next(
            i + 1 for i, l in enumerate(src.splitlines()) if l == ANCHOR
        )
        if not (lo <= anchor_line <= hi):
            print(
                f"[fail] the anchor (line {anchor_line}) is OUTSIDE {FN}() "
                f"(lines {lo}-{hi}) — refusing to inject into the wrong function.",
                file=sys.stderr,
            )
            return 1
        bak = path + ".pre-tcgen05-bs-fix"
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
            print(f"[vendor] backup           : {bak}")
        src = src.replace(ANCHOR, ANCHOR + INSERT, 1)
        with open(path, "w") as f:
            f.write(src)
        print("[vendor] anchored insert  : OK")

    # ---- 复核（无论走哪条路）：AST 重读，必须恰好修好 ----
    with open(path) as f:
        after = f.read()
    if fn_state(after) != "fixed":
        print("[fail] post-write AST check FAILED — the annotation is still absent", file=sys.stderr)
        return 1
    new_sha = hashlib.sha256(after.encode()).hexdigest()
    print(f"[vendor] method           : {method}")
    print(f"[vendor] new sha256       : {new_sha}")
    print(f"[vendor] verified         : {FN}() now sets ann[\"{ANNOT}\"] = 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
