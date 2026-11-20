#!/usr/bin/env python3
"""Standardize every 'interloc*_speaker_id' token in code/docs to the authoritative DB column name
(tools/a5_spellings.json — extracted directly from the DB via PRAGMA table_info).

Rationale: the DB column is spelled 'interloc<u/a>tor_speaker_id'; my new/existing code uses a
slightly different spelling — this normalizes everything to the exact DB spelling so hydration,
audit, and DB-writes all reference the same identifier.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
db_col = json.loads((REPO / "tools" / "a5_spellings.json").read_text(encoding="utf-8"))[
    "db_interlocutor_speaker_id_col_exact"
]
rx = re.compile(r"interloc\w*_speaker_id")

FILES = [
    "db/database.py",
    "tools/card_rating_system/audit_cards.py",
    "core/scenario_extractor.py",
    "tools/card_rating_system/AGENT_PROMPT.md",
    "tools/card_rating_system/RUBRIC.md",
    "tools/card_rating_system/TEAM_AGENT_PROMPT.md",
    "tools/master_synthesis.md",
    "tools/full_fix_plan.md",
]

for f in FILES:
    p = REPO / f
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8")
    new = rx.sub(lambda m: m.group(0) if m.group(0) == db_col else db_col, txt)
    if new != txt:
        p.write_text(new, encoding="utf-8")
        print(f"[patched] {f}: -> {db_col!r}")
    else:
        print(f"[clean]   {f}")

print(f"[+] DB column used everywhere: {db_col}")
