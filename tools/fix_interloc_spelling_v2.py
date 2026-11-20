#!/usr/bin/env python3
"""Definitive spelling normalization: load db_col from a5_spellings.json and rewrite every
'interloc*speaker_id' identifier in code to it (no hand-typed strings).

After this, print codepoint-level diffs so results are verifiable."""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
db_col = json.loads((REPO / "tools" / "a5_spellings.json").read_text(encoding="utf-8"))[
    "db_interlocator_speaker_id_col_exact"
]
print("db_col:", repr(db_col), "codepoints:", [ord(c) for c in db_col][:14])

# Match any interloc<X>_speaker_id-ish token (handles both visual spellings)
rx = re.compile(r"interloc\w*speaker\w*id")

def norm(s):
    return rx.sub(db_col, s)

FILES = [
    "db/database.py",
    "tools/card_rating_system/audit_cards.py",
    "core/scenario_extractor.py",
    "tools/card_rating_system/AGENT_PROMPT.md",
    "tools/card_rating_system/RUBRIC.md",
    "tools/master_synthesis.md",
    "tools/full_fix_plan.md",
]

for f in FILES:
    p = REPO / f
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8")
    new = norm(txt)
    if new != txt:
        p.write_text(new, encoding="utf-8")
        print(f"[patched] {f}")
    else:
        print(f"[clean]   {f}")

# Verify: extract unit.get / card.get / db_col spellings and show codepoints
def show(path, pattern, label):
    txt = (REPO / path).read_text(encoding="utf-8")
    for m in re.finditer(pattern, txt):
        s = m.group(1)
        if "speaker_id" in s and s.startswith("interloc"):
            print(f"  {label} ({path}): {s!r} == db_col? {s == db_col}")

show("db/database.py", r'unit\.get\("([^"]*speaker_id)"\)', "unit.get")
show("tools/card_rating_system/audit_cards.py", r'card\.get\("([^"]*speaker_id)"\)', "card.get")
print(f"  db_col final: {db_col!r}")
