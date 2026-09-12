#!/usr/bin/env python3
"""vg_ab2_report.py — the table + 判据 for vg_ab2.sh (the verify-graph A/B).

Reads $LOGDIR/<tag>.{dspark,resp.json,log} and prints:
  * one row per case: verify_ms / draft_ms / mean-k / tok/step / steps, the
    ENGAGED column (`captured` / `failed` / `no`), the emitted text's char count,
    its double-char count and an md5 (the arm-vs-arm TEXT equality proof);
  * the Δ table per prompt: verify_ms(=1) - verify_ms(=0), with BOTH gates
    (the task's 2 ms and the script's 10 ms);
  * a "measurement reached the 50-step gate?" flag — a case with no
    `[dspark] steps=` line is UNMEASURED and cannot carry a verdict.
"""
import hashlib
import json
import os
import re
import sys

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vgab2"
CASES = [
    ("L0", "0", "long"),
    ("L1", "1", "long"),
    ("S0", "0", "yardstick"),
    ("S1", "1", "yardstick"),
]


def field(line, key):
    i = line.find(key)
    if i < 0:
        return None
    m = re.match(r"[-+0-9.eE]+", line[i + len(key):])
    return float(m.group(0)) if m else None


def read(path, binary=False):
    try:
        return open(path, "rb" if binary else "r", errors=None if binary else "ignore").read()
    except OSError:
        return b"" if binary else ""


rows = {}
print("== per-case ==")
print("%-4s %-4s %-9s %9s %8s %8s %8s %7s %-8s %5s %6s %s" % (
    "CASE", "ARM", "CASE-KIND", "verify_ms", "draft_ms", "mean-k", "tok/step",
    "steps", "ENGAGED", "dbl", "chars", "md5(text)[:8]"))

for tag, arm, kind in CASES:
    dspark = read(os.path.join(LOGDIR, tag + ".dspark")).strip()
    log = read(os.path.join(LOGDIR, tag + ".log")) or read(os.path.join(LOGDIR, tag + ".log.kept"))
    engaged = ("captured" if "[verify_graph] captured" in log
               else "failed" if "[verify_graph] capture FAILED" in log else "no")
    content = ""
    try:
        content = json.loads(read(os.path.join(LOGDIR, tag + ".resp.json")))["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        content = ""
        resp_err = str(exc)
    else:
        resp_err = ""
    ch = list(content)
    dbl = sum(1 for i in range(1, len(ch)) if ch[i] == ch[i - 1] and not ch[i].isspace())
    md5 = hashlib.md5(content.encode()).hexdigest()[:8] if content else "NA"
    v, d = field(dspark, "verify="), field(dspark, "draft=")
    k, t, s = field(dspark, "mean-k="), field(dspark, "tok/step="), field(dspark, "steps=")
    steps_lines = len([1 for l in log.splitlines() if "[dsv41] step pos=" in l])
    rows[(arm, kind)] = dict(v=v, d=d, chars=len(content), dbl=dbl, md5=md5,
                             engaged=engaged, k=k, t=t, s=s, steps_lines=steps_lines)
    print("%-4s %-4s %-9s %9s %8s %8s %8s %7s %-8s %5d %6d %s%s" % (
        tag, arm, kind,
        "NA" if v is None else "%.2f" % v,
        "NA" if d is None else "%.2f" % d,
        "NA" if k is None else "%.3f" % k,
        "NA" if t is None else "%.3f" % t,
        "NA" if s is None else "%d" % int(s),
        engaged, dbl, len(content), md5,
        ("  <resp unreadable: %s>" % resp_err) if resp_err else ""))

print()
print("== 判据 ==")
rc = 0
engage_ok = True
text_ok = True
for arm, kind in (("1", "long"), ("1", "yardstick")):
    if rows[(arm, kind)]["engaged"] != "captured":
        print("  ENGAGE  %-9s arm=1 engagement=%s  -> this arm is NOT evidence about the graph"
              % (kind, rows[(arm, kind)]["engaged"]))
        engage_ok = False
for arm, kind in (("0", "long"), ("0", "yardstick")):
    if rows[(arm, kind)]["engaged"] != "no":
        print("  ENGAGE  %-9s arm=0 printed a verify_graph line (%s) — env not honoured"
              % (kind, rows[(arm, kind)]["engaged"]))
        engage_ok = False
if engage_ok:
    print("  ENGAGE  OK: both =1 arms printed 'captured', neither =0 arm did.")

for kind in ("long", "yardstick"):
    a, b = rows[("0", kind)], rows[("1", kind)]
    if a["chars"] != b["chars"] or a["md5"] != b["md5"]:
        print("  TEXT    %-9s DIFFERS: arm0 %d chars dbl=%d md5=%s | arm1 %d chars dbl=%d md5=%s"
              % (kind, a["chars"], a["dbl"], a["md5"], b["chars"], b["dbl"], b["md5"]))
        text_ok = False
    else:
        print("  TEXT    %-9s identical: %d chars, %d double-chars, md5 %s (both arms)"
              % (kind, a["chars"], a["dbl"], a["md5"]))

for kind, gate in (("long", 2.0), ("yardstick", 2.0)):
    a, b = rows[("0", kind)], rows[("1", kind)]
    if a["v"] is None or b["v"] is None:
        missing = [t for t, r in (("arm0", a), ("arm1", b)) if r["v"] is None]
        print("  SPEED   %-9s UNMEASURED (%s printed no '[dspark] steps=' line: %d backbone step(s),"
              " the 50-step gate was not reached) — no verdict from this case" % (kind, ",".join(missing), a["steps_lines"]))
        rc = 2
        continue
    dv = b["v"] - a["v"]
    print("  SPEED   %-9s verify %.2f -> %.2f ms (Δ%+.2f, gain %+.2f ms; task gate >= %.1f ms %s;"
          " plan expectation >= 10 ms %s)"
          % (kind, a["v"], b["v"], dv, -dv, gate, "PASS" if -dv >= gate else "FAIL",
             "PASS" if -dv >= 10.0 else "FAIL"))
    if -dv < gate:
        rc = max(rc, 1)

print()
print("verdict: rc=%d  (0 = both legs hold, 1 = a speed/text leg failed, 2 = a case was unmeasurable)" % rc)
sys.exit(rc)
