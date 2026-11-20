#!/usr/bin/env python3
"""Compare exact spellings of interloc-related strings used in code vs DB column."""
import json
from pathlib import Path

db_col = json.loads(Path("tools/a5_spellings.json").read_text(encoding="utf-8"))["db_interlocator_speaker_id_col_exact"]
print("DB col       :", db_col, [ord(c) for c in db_col])

for f in [
    "db/database.py",
    "tools/card_rating_system/audit_cards.py",
    "core/scenario_extractor.py",
    "tools/card_rating_system/AGENT_PROMPT.md",
    "tools/card_rating_system/RUBRIC.md",
]:
    txt = Path(f).read_text(encoding="utf-8")
    # find the string most similar to the DB column
    seen = set()
    for line_no, line in enumerate(txt.splitlines(), 1):
        for word in line.split():
            w = word.strip("'\"`()[]{}:,.= ").lower()
            if "interloc" in w and "speaker" in w:
                seen.add((w, line_no, line.strip()[:80]))
    for w, ln, line in sorted(seen):
        print(f"{f}:{ln}: {w}  {'<== SAME' if w == db_col else '<== DIFF'}")
