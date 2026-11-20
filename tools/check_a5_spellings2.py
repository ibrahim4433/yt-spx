#!/usr/bin/env python3
import json
import re
from pathlib import Path

db_col = json.loads(Path("tools/a5_spellings.json").read_text(encoding="utf-8"))[
    "db_interlocator_speaker_id_col_exact"
]

def extract(path, pattern, flags=0):
    txt = Path(path).read_text(encoding="utf-8")
    for m in re.finditer(pattern, txt, flags):
        yield m.group(0)

# db/database.py line 1418: unit.get("...")
for s in extract("db/database.py", r'unit\.get\("([^"]*interloc[^"]*)"\)'):
    print("db.py unit.get:", s == db_col, "->", repr(s))
# audit line 415: card.get("...")
for s in extract("tools/card_rating_system/audit_cards.py", r'card\.get\("([^"]*interloc[^"]*)"\)'):
    print("audit card.get:", s == db_col, "->", repr(s))
# my comment at 1422
for s in extract("db/database.py", r'empty_([^"]*)"'):
    print("db.py comment:", s)
print("db_col:", repr(db_col))
