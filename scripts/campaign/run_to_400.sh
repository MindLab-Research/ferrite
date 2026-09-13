#!/bin/bash
# The SPEC-mode chain to 400 tok/s. Supersedes the earlier non-spec version of this script.
#
# WHY REWRITTEN: `[dsv41] step pos=` is only a FULL step when DSV41_SPEC=1 (serve.rs's step_time has two
# branches; the non-spec one advances pos by 1 per step). arm_run.sh's COMMON does not set it, so the
# first version of this script measured ordinary 1-token decode steps (10.19 ms) and the "318 tok/s"
# reported from it was an invalid rescale — retracted. Every measurement below sets DSV41_SPEC=1.
#
# All steps start from the CORRECT path: the hand-written fp4 BS arm destroys the model
# (DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0), proven by the 1..100 probe emitting nothing
# but repeated junk.
#
# The lever, from the code's own comments: verify is ~7000 streaming launches per verify (40 layers x
# ~170 nodes) at ~2.9 us submit each (~20.3 ms of the 28.17 ms), while one graph dispatches at
# ~0.4 us/node (~2.8 ms). SGLang's published verify is 7.3 ms at the same gamma, so ~24.5-26.7 ms
# verify is ~3.9x theirs.
set -uo pipefail
cd "$HOME/ferrite"

judge () {
  local tag="$1" log="$2"
  echo "=================== $tag ==================="
  python3 - "$log" <<'PY'
import re, sys, statistics
txt = open(sys.argv[1], errors='ignore').read()
m = re.findall(r"\] OUT: (.*)", txt)
if m:
    body = m[-1].strip("'\"")
    toks = [x for x in body.replace("\\n", "\n").split("\n") if x.strip()]
    ok = 0
    for i, tk in enumerate(toks[:61]):
        if tk.strip() == str(i + 1): ok += 1
        else: break
    print(f"text: {len(toks)} lines, first {ok} are exactly 1..N")
    print("  head:", toks[:24])
else:
    print("text: (no OUT line)")
v = [float(x) for x in re.findall(r"\[dsv41\] step pos=\d+: ([0-9.]+)ms", txt)]
tail = v[-200:]
if tail:
    s = sorted(tail); med = statistics.median(tail)
    print(f"step: n={len(tail)} p50={med:.2f}ms p10={s[len(s)//10]:.2f} p90={s[9*len(s)//10]:.2f}")
    print(f"  => SPEC step {med:.2f} ms; tok/step = mean-k + 1, so at 2.24 => 3.24 => {3.24*1000/med:.1f} tok/s"
          f" ; at 1.00 => 2.00 => {2.00*1000/med:.1f} tok/s")
else:
    print("step: no step lines")
for pat in (r"\[dsv41\]\s*mean-k[^\n]*", r"accept[^\n]{0,80}", r"\[dspark\][^\n]{0,120}"):
    for mm in re.findall(pat, txt)[-2:]:
        print("  log:", mm[:180])
PY
}

echo "############ P0: the PRODUCTION-LIKE config — both graphs on (one variable pair) ############"
# graph-safety-audit: DSV41_GRAPH_STEP defaults to ON (unwrap_or(true), since 2026-09-11) and
# arm_run.sh's GRAPH_OFF explicitly forces it to 0 — so EVERY earlier arm ran with the whole-step graph
# disabled, which also (side effect) disables ar_v5. That is very likely why the earlier non-spec arms
# showed 10.19 ms where the user's reference is 6.3 ms/step. VERIFY_GRAPH is OFF by default, so P0
# turns both on: the closest thing to the production configuration that exists as an env set.
bash "$HOME/num100.sh" STEP_P0 \
  DSV41_SPEC=1 DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 \
  DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0
judge "P0 (spec + verify graph + step graph)" "$HOME/armrun_STEP_P0.log"

echo "############ P1: P0 + the precision-neutral folds (one variable) ############"
bash "$HOME/num100.sh" STEP_P1F \
  DSV41_SPEC=1 DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 \
  DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 \
  DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1 \
  DSV41_ATTN_MROWS=1 DSV41_COMPRESSOR_PROJ_MROWS=1 \
  DSV41_ENGRAM_PROJ_MROWS=1 DSV41_ENGRAM_GATHER_MROWS=1 \
  DSV41_DRAFT_P3LITE_SEED=1 DSV41_DRAFT_P3LITE_KV=1 DSV41_DRAFT_P3LITE_ATTN=1
judge "P1F (P0 + precision-neutral folds)" "$HOME/armrun_STEP_P1F.log"

echo "############ P2: P0 + the SAME-FORMAT grouped MoE (one variable) ############"
bash "$HOME/num100.sh" STEP_P2G \
  DSV41_SPEC=1 DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 \
  DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 \
  DSV41_EXPERT_GROUPED=1 DSV41_EXPERT_TCGEN05_E4M3=1
judge "P2G (P0 + same-format grouped MoE)" "$HOME/armrun_STEP_P2G.log"

echo "############ P3: P0 + the OLD per-slot MoE (the m=6 side of the MOE_BATCH A/B) ############"
bash "$HOME/num100.sh" STEP_P3O \
  DSV41_SPEC=1 DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 \
  DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 DSV41_MOE_BATCH=0
judge "P3O (P0 + old per-slot MoE)" "$HOME/armrun_STEP_P3O.log"
