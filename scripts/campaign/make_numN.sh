#!/bin/bash
# Make a parameterised counting probe from the verified num100.sh: same runner, same raw-text dump,
# but the count target comes from $N so the defect's boundary can be mapped (the first gap always
# starts at the same generation position, so a sweep over N bounds where it begins).
set -uo pipefail
ssh -o BatchMode=yes ubuntu@43.202.208.136 'python3 - <<PY
import pathlib
src = pathlib.Path.home() / "num100.sh"
s = src.read_text()
old = """BODY='\''{"messages":[{"role":"user","content":"请从1数到100，每个数字单独一行"}],"max_tokens":400,"temperature":0}'\''"""
new = """N=${N:-100}
BODY=$(printf '\''{"messages":[{"role":"user","content":"请从1数到%s，每个数字单独一行"}],"max_tokens":400,"temperature":0}'\'' "$N")"""
assert old in s, "BODY anchor not found"
s2 = s.replace(old, new, 1)
p = pathlib.Path.home() / "numN.sh"
p.write_text(s2)
print("numN.sh written; BODY now parameterised by N")
PY
bash -n ~/numN.sh && echo NUMN_SYNTAX_OK && grep -n "N=" ~/numN.sh | head -3'
