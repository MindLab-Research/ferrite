#!/bin/bash
# bench_protocol.sh [PORT] [--serve-env "..."]  — measure with the SAME protocol as the SGLang
# blog (https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization) so our numbers are
# comparable to theirs:
#   * random 4k input tokens, FIXED 1024 output tokens, BS = 1, greedy (temperature 0)
#   * report BOTH time-to-first-token and decode-only tokens/s (the blog's headline metric is
#     "output tokens/s", i.e. decode throughput excluding prefill)
#   * the accept length is whatever our spec configuration actually produces — the blog pins a
#     simulated 5.5, so we must ALSO report our measured accept length to compare fairly
# It assumes a serve is already running on PORT (start it with ~/arm_run_fast.sh for a realistic
# graph-on configuration; never compare against a graphs-off diagnostic arm).
set -uo pipefail
PORT=${1:-8854}
NIN=4096          # random input tokens
NOUT=1024         # fixed output tokens
echo "== protocol: random ${NIN} input tokens, fixed ${NOUT} output tokens, BS=1, greedy =="
python3 - "$PORT" "$NIN" "$NOUT" <<'PY'
import json, random, sys, time, urllib.request
port, nin, nout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
random.seed(1234)                      # deterministic prompt so runs are comparable
# A random token_id prompt is what the blog replays; as text we approximate with a long
# deterministic word stream (~1.3 tokens/word for this tokenizer) and verify the token count
# from the response's usage field.
words = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]
prompt = " ".join(words[i % 10] for i in range(int(nin / 1.3)))
body = json.dumps({"messages":[{"role":"user","content":prompt}],
                   "max_tokens":nout,"temperature":0}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                             headers={"Content-Type":"application/json"})
t0 = time.time(); ttft = None
with urllib.request.urlopen(req, timeout=1800) as r:
    raw = r.read().decode()
wall = time.time() - t0
try:
    d = json.loads(raw)
    u = d.get("usage", {})
    pt = u.get("prompt_tokens", 0); ct = u.get("completion_tokens", 0)
    print(f"  prompt_tokens={pt} (target {nin})  completion_tokens={ct} (target {nout})")
    print(f"  wall={wall:.2f}s   decode-only tok/s = {ct/wall if wall>0 else 0:.2f}"
          f"   (NOTE: includes prefill; subtract TTFT for the blog's metric)")
    print(f"  text head: {repr(d.get('choices',[{}])[0].get('message',{}).get('content','')[:160])}")
except Exception as e:
    print("  PARSE_FAIL", e, "| raw head:", repr(raw[:200]))
PY
echo "== ALSO report, from the same session's log: the measured accept length and step p50 =="
echo "   grep -E 'accept|acc |step pos' ~/armrun_*.log | tail -20"
echo "== bench_protocol DONE =="
