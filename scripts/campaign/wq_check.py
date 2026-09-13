#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wq_check.py — 推理正确性验收判据的机械化检查器（纯本地 CPU，零第三方依赖）。

背景（用户硬性红线，见 docs/agent/moe-bs-crash-investigation.md §39/§40）：
  1. 模型输出**不能重复、不能乱码**（两者同级）。已知事故形态：
     ① 纯制表符/空白       例: '\\n\\t\\t\\t\\t\\t\\t…'
     ② 单字符/单数字循环   例: ', 6, 6, 7, 7, 7, 7, 7, 7, 7,'
     ③ 混合乱码            例: ' OR "OW  或者是: \\t (  + \\t:   +   :    \\t. 在 \\t:'
  2. 数字任务数数：prompt「请从 1 数到 100，每个数字单独一行」，**只对前 61 行有效**
     ⇒ 前 61 行必须严格递增 1..61。
  3. 必须与 EAGER 对照（同 prompt / 同权重 / 关掉被验特性的那次运行）。
     **退化与 EAGER 一致 ⇒ 视为模型行为，不算我们的 bug**（文档确立的判读规则）。

用法
  wq_check.py --text-file <f> [--expect-count 61] [--eager-file <f2>]
  wq_check.py --log <armrun_*.log> [armrun_*.log ...] [--expect-count 61]
  wq_check.py --selftest

输入文本会自动"先反转义再判定"：若内容被 '…' 或 "…" 包裹且含 \\n / \\t / \\x.. 等
转义（arm_run.sh 的 `[NAME] OUT: <repr>` 行就是这种），会先反转义；原始文本文件直接判。
可用 --no-unescape / --raw 关闭该行为。

退出码：0 = 全部 PASS/WARN；1 = 至少一个 FAIL；2 = 用法错误。
"""

import argparse
import os
import re
import sys
import unicodedata

# --------------------------------------------------------------------------- #
# 阈值（集中定义，便于调参；边界情形宁可 WARN 也不 FAIL）
# --------------------------------------------------------------------------- #
TH = {
    # (a) 重复
    "repeat_len_fail": 12,      # 最长"重复≥3次"子串长度 ≥12 → FAIL
    "repeat_len_warn": 6,
    "repeat_density_fail": 0.40,  # L*count/len ≥0.40 → FAIL（文本基本上就是复读）
    "repeat_density_warn": 0.20,
    "char_share_fail": 0.60,    # 单一非空白字符占比 >60% → FAIL
    "char_share_warn": 0.45,
    "digit_share_fail": 0.55,   # 单一数字占全部数字的比例 >55%（数字≥8个才判）→ FAIL
    "digit_share_warn": 0.45,
    "token_loop_fail": 0.25,    # 最常见 token 占全部 token 比例 ≥25% 且出现≥5次 → FAIL
    "token_loop_min": 5,
    # (b) 乱码
    "tab_ratio_fail": 0.30,     # 制表符占比 >30% → FAIL
    "tab_ratio_warn": 0.15,
    "ctrl_ratio_fail": 0.05,    # 非空白控制字符占比 >5%（且≥2个）→ FAIL
    "readable_fail": 0.40,      # 可读字符（字母/数字/CJK/常用标点，不含空白）占比 <40% → FAIL
    "readable_warn": 0.55,
    "alnum_cjk_fail": 0.25,     # 纯字母/数字/CJK 占比 <25% → FAIL（纯符号垃圾）
    "alnum_cjk_warn": 0.40,
}

CJK_RANGES = (
    (0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF), (0xFF00, 0xFFEF), (0xAC00, 0xD7AF),
)
PUNCT = set(",.;:!?'\"()[]{}<>+-*/=_%&@#$~^|\\`" + "，。、；：！？（）【】《》“”‘’—…·「」")
WS = set(" \t\n\r\v\f\u00a0\u3000")


def is_cjk(ch):
    o = ord(ch)
    for lo, hi in CJK_RANGES:
        if lo <= o <= hi:
            return True
    return False


# --------------------------------------------------------------------------- #
# 反转义（兼容 Python repr / 日志里的转义文本）
# --------------------------------------------------------------------------- #
_SIMPLE = {
    "n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\",
    "'": "'", '"': '"', "a": "\a", "b": "\b", "f": "\f", "v": "\v",
}


def unescape(s):
    """手动反转义，保留 CJK（不用 codecs.unicode_escape，它会把 CJK 打成 latin-1 乱码）。"""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            k = s[i + 1]
            if k in _SIMPLE:
                out.append(_SIMPLE[k]); i += 2; continue
            if k in "xuU":
                width = {"x": 2, "u": 4, "U": 8}[k]
                h = s[i + 2:i + 2 + width]
                if len(h) == width:
                    try:
                        out.append(chr(int(h, 16))); i += 2 + width; continue
                    except ValueError:
                        pass
            if k == "N" and i + 2 < n and s[i + 2] == "{":
                j = s.find("}", i + 3)
                if j > 0:
                    try:
                        out.append(unicodedata.lookup(s[i + 3:j])); i = j + 1; continue
                    except KeyError:
                        pass
            out.append(c); i += 1; continue          # 未知转义：保留原样
        out.append(c); i += 1
    return "".join(out)


def maybe_unescape(raw, force=False, disable=False):
    """若整段（或整行）被成对引号包裹，则剥引号并反转义。返回 (text, unescaped: bool)。"""
    if disable:
        return raw, False
    t = raw
    if t.endswith("\n"):
        t = t[:-1]
    ts = t.strip()
    if len(ts) >= 2 and ts[0] in "'\"" and ts[-1] == ts[0]:
        return unescape(ts[1:-1]), True
    if force:
        return unescape(raw), True
    return raw, False


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def longest_repeat(s, min_count=3, max_n=256):
    """最长子串 L 使其（某个）L-gram 出现 ≥min_count 次。返回 (L, count, sample)。

    单调性：若长度 n+1 的串出现 ≥k 次，则其长度 n 的前缀也出现 ≥k 次
    ⇒ 一旦某 n 的最大计数 <min_count 即可提前终止（避免 O(n²) 全扫）。
    """
    if len(s) < 2:
        return 0, 0, ""
    best = (0, 0, "")
    n_max = min(max_n, max(1, len(s) // 2))
    for n in range(2, n_max + 1):
        counts = {}
        for i in range(len(s) - n + 1):
            sub = s[i:i + n]
            counts[sub] = counts.get(sub, 0) + 1
        if not counts:
            break
        sub, cnt = max(counts.items(), key=lambda kv: kv[1])
        if cnt < min_count:
            break                       # 单调 ⇒ 更长的也不会达标
        best = (n, cnt, sub)
    return best


def tokenize(s):
    return [t for t in re.split(r"[\s,;|]+", s) if t]


def metrics(text):
    m = {}
    L = len(text)
    m["len"] = L
    m["lines"] = len(text.splitlines())
    if L == 0:
        return m
    m["tab"] = text.count("\t")
    m["tab_ratio"] = m["tab"] / L
    ctrl = sum(1 for c in text if unicodedata.category(c) == "Cc" and c not in "\t\n\r")
    m["ctrl"] = ctrl
    m["ctrl_ratio"] = ctrl / L
    unprint = sum(1 for c in text if unicodedata.category(c) in ("Cf", "Co", "Cs", "Cn"))
    m["unprint"] = unprint
    readable = alnum_cjk = 0
    nonspace = {}
    digits = {}
    for c in text:
        if c.isdigit():
            digits[c] = digits.get(c, 0) + 1
        if c.isalnum() or is_cjk(c):
            alnum_cjk += 1
        if c.isalnum() or is_cjk(c) or c in PUNCT:
            readable += 1
        if c not in WS:
            nonspace[c] = nonspace.get(c, 0) + 1
    m["readable_ratio"] = readable / L
    m["alnum_cjk_ratio"] = alnum_cjk / L
    ns_total = sum(nonspace.values())
    if float(ns_total) > 0:
        ch, cnt = max(nonspace.items(), key=lambda kv: kv[1])
        m["max_char"] = ch
        m["max_char_count"] = cnt
        m["max_char_share"] = cnt / ns_total
    else:
        m["max_char"] = ""
        m["max_char_count"] = 0
        m["max_char_share"] = 1.0
    m["digit_total"] = sum(digits.values())
    if m["digit_total"]:
        d, cnt = max(digits.items(), key=lambda kv: kv[1])
        m["max_digit"] = d
        m["max_digit_count"] = cnt
        m["max_digit_share"] = cnt / m["digit_total"]
    else:
        m["max_digit"] = ""
        m["max_digit_count"] = 0
        m["max_digit_share"] = 0.0
    toks = tokenize(text)
    m["n_tokens"] = len(toks)
    if toks:
        tc = {}
        for t in toks:
            tc[t] = tc.get(t, 0) + 1
        t, cnt = max(tc.items(), key=lambda kv: kv[1])
        m["top_token"] = t
        m["top_token_count"] = cnt
        m["top_token_share"] = cnt / len(toks)
    else:
        m["top_token"] = ""
        m["top_token_count"] = 0
        m["top_token_share"] = 0.0
    Lr, cntr, sample = longest_repeat(text)
    m["repeat_len"] = Lr
    m["repeat_count"] = cntr
    m["repeat_sample"] = sample
    m["repeat_density"] = (Lr * cntr / L) if L else 0.0
    return m


# --------------------------------------------------------------------------- #
# 判据
# --------------------------------------------------------------------------- #
def crit_repeat(text, m):
    """a) 重复：最长重复子串/重复 n-gram + 单字符/单数字占比过高。"""
    res, detail = "PASS", []
    if m["len"] == 0:
        return "FAIL", "empty output"
    fails = []
    if m["repeat_len"] >= TH["repeat_len_fail"]:
        fails.append("long_repeat")
    if m["repeat_density"] >= TH["repeat_density_fail"]:
        fails.append("repeat_density")
    if m["max_char_share"] > TH["char_share_fail"]:
        fails.append("char_share")
    if m["digit_total"] >= 8 and m["max_digit_share"] > TH["digit_share_fail"]:
        fails.append("digit_share")
    if m["top_token_count"] >= TH["token_loop_min"] and m["top_token_share"] >= TH["token_loop_fail"]:
        fails.append("token_loop")
    warns = []
    if m["repeat_len"] >= TH["repeat_len_warn"] or m["repeat_density"] >= TH["repeat_density_warn"]:
        warns.append("long_repeat")
    if m["max_char_share"] > TH["char_share_warn"]:
        warns.append("char_share")
    if m["digit_total"] >= 8 and m["max_digit_share"] > TH["digit_share_warn"]:
        warns.append("digit_share")

    samp = m["repeat_sample"].replace("\t", "\\t").replace("\n", "\\n")[:40]
    detail.append(
        "longest_repeat=%d x%d (density=%.2f, e.g. %r)" % (
            m["repeat_len"], m["repeat_count"], m["repeat_density"], samp))
    ns_total = sum(1 for c in text if c not in WS)
    if ns_total == 0:
        detail.append("max_char_share=n/a (no non-whitespace char at all)")
    else:
        detail.append("max_char_share=%.2f (char %r x%d/%d nonspace)" % (
            m["max_char_share"], m["max_char"], m["max_char_count"], ns_total))
    if m["digit_total"]:
        detail.append("max_digit_share=%.2f (digit %r x%d/%d)" % (
            m["max_digit_share"], m["max_digit"], m["max_digit_count"], m["digit_total"]))
    detail.append("top_token=%r x%d/%d tokens (share=%.2f)" % (
        m["top_token"][:12], m["top_token_count"], m["n_tokens"], m["top_token_share"]))
    if fails:
        res = "FAIL"
        detail.insert(0, "HITS=" + ",".join(fails))
    elif warns:
        res = "WARN"
        detail.insert(0, "soft=" + ",".join(sorted(set(warns))))
    return res, "; ".join(detail)


def crit_garbage(text, m):
    """b) 乱码：控制符/制表符占比、不可打印字符、可读字符占比过低。"""
    if m["len"] == 0:
        return "FAIL", "empty output"
    fails, warns = [], []
    if m["tab_ratio"] > TH["tab_ratio_fail"]:
        fails.append("tab_ratio")
    elif m["tab_ratio"] > TH["tab_ratio_warn"]:
        warns.append("tab_ratio")
    if m["ctrl"] >= 2 and m["ctrl_ratio"] > TH["ctrl_ratio_fail"]:
        fails.append("control_chars")
    if m["readable_ratio"] < TH["readable_fail"]:
        fails.append("readable_low")
    elif m["readable_ratio"] < TH["readable_warn"]:
        warns.append("readable_low")
    if m["alnum_cjk_ratio"] < TH["alnum_cjk_fail"]:
        fails.append("alnum_cjk_low")
    elif m["alnum_cjk_ratio"] < TH["alnum_cjk_warn"]:
        warns.append("alnum_cjk_low")
    if m["unprint"] >= 2:
        fails.append("unprintable")
    elif m["unprint"] == 1:
        warns.append("unprintable")

    detail = "tab_ratio=%.2f (%d) control_ratio=%.3f (%d) unprintable=%d readable_ratio=%.2f alnum_cjk_ratio=%.2f" % (
        m["tab_ratio"], m["tab"], m["ctrl_ratio"], m["ctrl"], m["unprint"],
        m["readable_ratio"], m["alnum_cjk_ratio"])
    if fails:
        return "FAIL", "HITS=" + ",".join(fails) + "; " + detail
    if warns:
        return "WARN", "soft=" + ",".join(sorted(set(warns))) + "; " + detail
    return "PASS", detail


_NUM_LINE = re.compile(r"^\s*(\d{1,7})\s*[.、)）\]]?\s*$")


def extract_numbers(text):
    """逐行抽整数（容忍 markdown 项目符号/反引号/尾随标点）；行数不足时退化为全文扫描。"""
    nums, lines = [], []
    for idx, line in enumerate(text.splitlines(), 1):
        s = line.strip().strip("`*#>-\t ")
        s = s.rstrip(".、)）]】：:〉》").strip()
        if not s:
            continue
        if _NUM_LINE.match(" " + s + " "):
            nums.append(int(s)); lines.append(idx)
    if len(nums) < 3:                       # 非"每行一个数字"格式 → 全文扫描
        nums, lines = [], []
        for mm in re.finditer(r"(?<!\d)(\d{1,7})(?!\d)", text):
            nums.append(int(mm.group(1)))
            lines.append(text[:mm.start()].count("\n") + 1)
    return nums, lines


def count_prefix(nums, N):
    """返回 dict：clean_prefix_len / first_bad_index(0-based,-1=无) / first_bad_line"""
    exp = list(range(1, N + 1))
    bad = -1
    for i in range(min(len(nums), N)):
        if nums[i] != exp[i]:
            bad = i
            break
    return bad


def crit_count(text, m, N):
    """c) 计数：连续整数序列是否严格递增且覆盖 1..N。"""
    nums, lines = extract_numbers(text)
    if not nums:
        return "FAIL", "no integers found; first_bad_line=n/a", {"nums": [], "lines": [], "bad": 0, "N": N}
    bad = count_prefix(nums, N)
    info = {"nums": nums, "lines": lines, "bad": bad, "N": N}
    if len(nums) >= N:
        if bad < 0:
            return "PASS", "first %d lines == 1..%d (strictly increasing)" % (N, N), info
        got = nums[bad]
        return "FAIL", "line %d: got %r, want %d (first_bad_line=%d; head=%s)" % (
            lines[bad] if bad < len(lines) else bad + 1, got, bad + 1, bad + 1,
            nums[:12]), info
    # 不足 N 个：只要求已给出的前缀干净
    if bad < 0:
        return "WARN", "truncated: only %d/%d numbers given, prefix 1..%d clean" % (
            len(nums), N, len(nums)), info
    return "FAIL", "line %d: got %r, want %d (first_bad_line=%d; only %d/%d numbers)" % (
        lines[bad] if bad < len(lines) else bad + 1, nums[bad], bad + 1, bad + 1,
        len(nums), N), info


def eager_parity(arm_info, egr_info, arm_text, egr_text):
    """d) EAGER 对照：计数前缀是否同现。返回 (result, detail, same: bool)。"""
    a, e = arm_info["nums"], egr_info["nums"]
    common = 0
    while common < min(len(a), len(e)) and a[common] == e[common]:
        common += 1
    identical = (arm_text == egr_text)
    ab, eb = arm_info["bad"], egr_info["bad"]
    same = identical or (ab == eb and (a[:ab] == e[:eb]))
    tag = "SAME" if same else "DIFFER"
    detail = ("arm: n=%d first_bad_line=%s | eager: n=%d first_bad_line=%s | "
              "common_prefix=%d | identical_text=%s" % (
                  len(a), (ab + 1) if ab >= 0 else "-", len(e), (eb + 1) if eb >= 0 else "-",
                  common, identical))
    return ("PASS" if same else "FAIL"), tag + "; " + detail, same


# --------------------------------------------------------------------------- #
# 顶层分析
# --------------------------------------------------------------------------- #
def analyze(text, expect_count=61, eager_text=None):
    m = metrics(text)
    crit = []
    r_repeat = crit_repeat(text, m)
    crit.append(("repeat", r_repeat[0], r_repeat[1]))
    r_garb = crit_garbage(text, m)
    crit.append(("garbage", r_garb[0], r_garb[1]))
    r_cnt = crit_count(text, m, expect_count)
    crit.append(("count", r_cnt[0], r_cnt[1]))
    arm_info = r_cnt[2]

    eager_same = None
    exculpated = False
    if eager_text is not None:
        em = metrics(eager_text)
        e_cnt = crit_count(eager_text, em, expect_count)
        r_par = eager_parity(arm_info, e_cnt[2], text, eager_text)
        crit.append(("eager", r_par[0], r_par[1]))
        eager_same = r_par[2]
        if eager_same:
            # EAGER 基线自身也有同样的缺陷（同一处 FAIL）⇒ 按文档规则豁免为「模型行为」
            e_hits = _hits(_eager_crit(eager_text, expect_count))
            if _hits(crit) & e_hits:
                exculpated = True

    results = [r for _, r, _ in crit]
    if all(r == "PASS" for r in results):
        verdict = "PASS"
    elif "FAIL" in results:
        verdict = "FAIL"
    else:
        verdict = "WARN"
    if exculpated and verdict == "FAIL":
        verdict = "PASS"

    return {
        "verdict": verdict,
        "criteria": crit,
        "exculpated": exculpated,
        "eager_same": eager_same,
        "metrics": m,
        "count_info": arm_info,
    }


def _eager_crit(eager_text, expect_count):
    em = metrics(eager_text)
    out = [("repeat",) + crit_repeat(eager_text, em)[:2]]
    out.append(("garbage",) + crit_garbage(eager_text, em)[:2])
    out.append(("count",) + crit_count(eager_text, em, expect_count)[:2])
    return out


def _hits(crit, _=None):
    s = set()
    for name, res, det in crit:
        if res == "FAIL" and det.startswith("HITS="):
            for h in det[5:].split(";")[0].split(","):
                s.add(h.strip())
    return s


def fmt_verdict(r, source, expect_count, unescaped, note=""):
    lines = []
    lines.append("--- %s%s" % (source, " [unescaped]" if unescaped else ""))
    m = r["metrics"]
    lines.append("    len=%d lines=%d (expect_count=%d)" % (m.get("len", 0), m.get("lines", 0), expect_count))
    for name, res, det in r["criteria"]:
        lines.append("    %-8s %-4s %s" % (name + ":", res, det))
    lines.append("    VERDICT: %s%s" % (r["verdict"], (" [exculpated: 退化与 EAGER 一致 ⇒ 模型行为，不算我们的 bug]" if r["exculpated"] else "")))
    if r["eager_same"] is None:
        lines.append("    note: no --eager-file ⇒ 无法区分「模型行为」与「我们的 bug」")
    if note:
        lines.append("    note: " + note)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 日志模式
# --------------------------------------------------------------------------- #
OUT_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*OUT:\s*(?P<body>.*)$")
ARM_KV_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*(?P<key>[A-Za-z_]+):\s*(?P<val>.*)$")
ARM_FLAG_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*(?P<key>SERVE_FAILED|DONE)\s*$")


def extract_arms(log_text):
    """从 arm_run.sh 控制台/日志里抽 `[<name>] OUT: '…'`，同时收集配套判读行。"""
    arms = []
    index = {}
    for raw in log_text.splitlines():
        mm = OUT_RE.match(raw)
        if mm:
            name = mm.group("name")
            body = mm.group("body").rstrip()
            if body == "" or body.startswith("PARSE_FAIL"):
                a = {"name": name, "text": "", "unescaped": False,
                     "note": "OUT empty / PARSE_FAIL"}
            else:
                text, un = maybe_unescape(body)
                a = {"name": name, "text": text, "unescaped": un, "note": ""}
            arms.append(a)
            index[name] = a
            continue
        mk = ARM_KV_RE.match(raw)
        if mk and mk.group("name") in index:
            index[mk.group("name")][mk.group("key").lower()] = mk.group("val").strip()[:160]
            continue
        mf = ARM_FLAG_RE.match(raw)
        if mf and mf.group("name") in index:
            index[mf.group("name")][mf.group("key").lower()] = "yes"
    return arms


def log_evidence(raw):
    """当日志里没有 OUT 行时，仍抽取可用的数值证据（开关回执 / [NC] / step / 失败标记）。"""
    ev = {}
    sw = re.findall(r"\[moe-bs\] (packed fp4 staging|pack geometry|swapAB|canon layout|"
                    r"scale_vec::1X|sf byte order reversed) = (\d)", raw)
    if sw:
        seen = []
        for k, v in sw:
            item = "%s=%s" % (k, v)
            if item not in seen:
                seen.append(item)
        ev["switches"] = " ".join(sorted(seen))
    nc = re.findall(r"\[NC\] WORST[^\n]*", raw)
    if nc:
        ev["nc"] = nc[-1].strip()[:120]
    if "SERVE_FAILED" in raw:
        ev["serve"] = "SERVE_FAILED"
    steps = re.findall(r"\[dsv41\] step pos=\d+: ([0-9.]+)ms", raw)
    if steps:
        ev["step_last"] = steps[-1] + "ms"
    return ev


def run_log_mode(paths, expect_count, eager_path, no_unescape):
    rows = []
    exit_fail = False
    for p in paths:
        try:
            with open(p, "r", errors="replace") as f:
                raw = f.read()
        except OSError as exc:
            print("!!! cannot read %s: %s" % (p, exc)); exit_fail = True; continue
        print("=" * 78)
        print("### LOG %s" % p)
        arms = extract_arms(raw)
        if not arms:
            ev = log_evidence(raw)
            print("    NO-OUT-LINE: 该日志里没有 `[<name>] OUT: '…'` 行 ⇒ 无法判文本"
                  + ("（含 SERVE_FAILED ⇒ 服务没起来）" if ev.get("serve") else ""))
            if ev:
                print("    numeric evidence: " + "; ".join("%s=%s" % kv for kv in ev.items()))
            print("    hint: arm_run.sh 的 OUT 行打到 stdout，不写进 ~/armrun_*.log；"
                  "请 `bash ~/arm_run.sh X … | tee ~/armrun_X.log` 或把该行贴进文本文件。")
            rows.append((os.path.basename(p), "-", "-", "-", "-", "-", "NO-OUT-LINE",
                         (ev.get("nc") or ev.get("switches") or "").replace(" ", "_")[:44]))
            continue
        for a in arms:
            egr = None
            if eager_path:
                with open(eager_path, "r", errors="replace") as f:
                    egr, _ = maybe_unescape(f.read(), disable=no_unescape)
            r = analyze(a["text"], expect_count, egr)
            print(fmt_verdict(r, "%s  arm=%s" % (os.path.basename(p), a["name"]),
                              expect_count, a["unescaped"], a.get("note", "")))
            cd = {}
            res_by = {}
            for name, res, det in r["criteria"]:
                res_by[name] = res
                if res == "FAIL":
                    cd.setdefault(name, []).append(det.split(";")[0])
            rows.append((os.path.basename(p), a["name"], r["verdict"],
                         res_by.get("repeat", "-"), res_by.get("garbage", "-"),
                         res_by.get("count", "-"), res_by.get("eager", "n/a"),
                         (a.get("switches") or "")[:44]))
            if r["verdict"] == "FAIL":
                exit_fail = True
    print("=" * 78)
    print("SUMMARY (arm log verdict table)")
    print("%-26s %-9s %-8s %-8s %-8s %-8s %-8s %s" % (
        "log", "arm", "VERDICT", "repeat", "garbage", "count", "eager", "switches/evidence"))
    for row in rows:
        print("%-26s %-9s %-8s %-8s %-8s %-8s %-8s %s" % row)
    return 0 if not exit_fail else 1


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
MODE1 = "\n" + "\t" * 40 + "\n"
MODE2 = "\n( \t \t \t,  \t, 6, 6, 7, 7, 7, 7, 7, 7, 7,"
MODE3 = ' OR "OW  或者是: \t (  + \t:   +   :    \t. 在 \t:'
CLEAN61 = "\n".join(str(i) for i in range(1, 62)) + "\n"


def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="wq_check_selftest_")
    ok_all = True

    def check(cname, got, want, extra=""):
        nonlocal ok_all
        good = (got == want)
        ok_all = ok_all and good
        print("  [%s] %-4s expect=%-4s got=%-4s %s" % (cname, "OK" if good else "BAD", want, got, extra))

    print("wq_check.py --selftest")
    print("tmp=%s" % tmp)

    print("\n-- case 1: 事故形态① 纯制表符 → FAIL")
    r = analyze(MODE1)
    check("mode1-tabs", r["verdict"], "FAIL",
          "| repeat=%s garbage=%s" % (r["criteria"][0][1], r["criteria"][1][1]))

    print("\n-- case 2: 事故形态② 单数字循环 (6/7 复读) → FAIL")
    r = analyze(MODE2)
    check("mode2-digit-loop", r["verdict"], "FAIL",
          "| repeat=%s" % r["criteria"][0][1])

    print("\n-- case 3: 事故形态③ 混合乱码 → FAIL")
    r = analyze(MODE3)
    check("mode3-mixed", r["verdict"], "FAIL",
          "| garbage=%s" % r["criteria"][1][1])

    print("\n-- case 4: 正常计数 1..61 → PASS")
    r = analyze(CLEAN61, 61)
    check("clean61", r["verdict"], "PASS",
          "| count=%s | repeat_len=%d density=%.2f" % (
              r["criteria"][2][1][:40], r["metrics"]["repeat_len"], r["metrics"]["repeat_density"]))

    print("\n-- case 5: 正常计数 + EAGER 也干净（同现）→ PASS")
    r = analyze(CLEAN61, 61, eager_text=CLEAN61)
    check("clean61+eager", r["verdict"], "PASS", "| eager=%s" % r["criteria"][3][2].split(";")[0])

    print("\n-- case 6: 退化 与 EAGER 完全一致 → PASS(exculpated)")
    r = analyze(MODE3, 61, eager_text=MODE3)
    check("exculpated-eager-same", r["verdict"], "PASS",
          "| exculpated=%s eager=%s" % (r["exculpated"], r["criteria"][3][2].split(";")[0]))

    print("\n-- case 7: 退化 但 EAGER 干净（不一致）→ FAIL（是我们的 bug）")
    r = analyze(MODE1, 61, eager_text=CLEAN61)
    check("eager-differ", r["verdict"], "FAIL", "| eager=%s" % r["criteria"][3][2].split(";")[0])

    print("\n-- case 8: 反转义（文件里是 '\\n\\t...' 字面量）→ 反转义后仍判 FAIL")
    p8 = os.path.join(tmp, "escaped.txt")
    with open(p8, "w") as f:
        f.write("'\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t\\t'\n")
    raw = open(p8).read()
    text, un = maybe_unescape(raw)
    r = analyze(text)
    check("unescape-mode1", r["verdict"], "FAIL", "| unescaped=%s text=%r" % (un, text[:12]))

    print("\n-- case 9: log 模式解析 `[P0] OUT: '…'` → FAIL + 名字=P0")
    p9 = os.path.join(tmp, "armrun_P0.log")
    with open(p9, "w") as f:
        f.write("[serve] dsv41 config ok: 40 layers\n")
        f.write("[dsv41] step pos=1: 190.9ms (5.2 tok/s)\n")
        f.write("[P0] OUT: %r\n" % MODE3)
        f.write("[P0] ERR_COUNT: 0\n")
        f.write("[P0] STEP: step pos=62: 191.00ms (5.2 tok/s)\n")
    arms = extract_arms(open(p9).read())
    check("log-extract", len(arms), 1, "| name=%s" % (arms[0]["name"] if arms else "-"))
    if arms:
        check("log-arm-name", arms[0]["name"], "P0")
        check("log-arm-verdict", analyze(arms[0]["text"])["verdict"], "FAIL")

    print("\n-- case 10: 裸文本文件（无引号包裹，正常计数）→ PASS")
    p10 = os.path.join(tmp, "plain.txt")
    with open(p10, "w") as f:
        f.write(CLEAN61)
    text, un = maybe_unescape(open(p10).read())
    check("plain-noquote", analyze(text)["verdict"], "PASS", "| unescaped=%s" % un)

    print("\n-- case 11: 误报防护 — 带项目符号/句点的计数 → PASS")
    mk = "\n".join("%d." % i for i in range(1, 62)) + "\n"
    check("bulleted-count", analyze(mk)["verdict"], "PASS")

    print("\n-- case 12: 误报防护 — 只数到 10（截断）→ WARN 而非 FAIL")
    check("truncated-count", analyze("\n".join(str(i) for i in range(1, 11)), 61)["verdict"], "WARN")

    print("\n== SELFTEST %s ==" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="机械化的推理正确性验收判据检查器（重复/乱码/计数/EAGER 对照）")
    ap.add_argument("--text-file", help="待判定的文本文件（支持 repr 转义，会自动反转义）")
    ap.add_argument("--log", nargs="+", help="arm_run.sh 日志（含 `[name] OUT: '…'` 行）")
    ap.add_argument("--eager-file", help="EAGER 对照的文本文件（同 prompt/同权重，关掉被验特性）")
    ap.add_argument("--expect-count", type=int, default=61,
                    help="数字任务期望的前 N 行 = 1..N（默认 61）")
    ap.add_argument("--no-unescape", action="store_true", help="不要反转义（按原始文本判）")
    ap.add_argument("--raw", action="store_true", help="强制反转义整段文本")
    ap.add_argument("--selftest", action="store_true", help="跑内置用例")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.text_file and not args.log:
        ap.print_help()
        return 2

    exit_code = 0
    egr_text = None
    if args.eager_file:
        with open(args.eager_file, "r", errors="replace") as f:
            egr_text, un = maybe_unescape(f.read(), disable=args.no_unescape)

    if args.text_file:
        with open(args.text_file, "r", errors="replace") as f:
            raw = f.read()
        text, un = maybe_unescape(raw, force=args.raw, disable=args.no_unescape)
        r = analyze(text, args.expect_count, egr_text)
        print("wq_check.py — 判据：重复 / 乱码 / 计数 / EAGER 对照")
        print("expect_count=%d  eager=%s" % (args.expect_count, args.eager_file or "n/a"))
        print(fmt_verdict(r, args.text_file, args.expect_count, un))
        if r["verdict"] == "FAIL":
            exit_code = 1
        print("SHORT: %s" % r["verdict"])

    if args.log:
        rc = run_log_mode(args.log, args.expect_count, args.eager_file, args.no_unescape)
        exit_code = max(exit_code, rc)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
