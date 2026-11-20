#!/usr/bin/env python3
import re
import sqlite3
from pathlib import Path

conn = sqlite3.connect("data/dawah_knowledge.db")
cols = [r[1] for r in conn.execute("PRAGMA table_info(qa_units)").fetchall()]
conn.close()
db_col = next((c for c in cols if c.endswith("speaker_id") and "interloc" in c), None)
print("DB column:", db_col, "| codepoints:", [ord(ch) for ch in (db_col or "")][:25])

src = Path("db/database.py").read_text(encoding="utf-8")
for m in re.finditer(r'"(\w*interloc\w*speaker\w*)"', src):
    s = m.group(1)
    if "speaker_id" in s:
        print("db/database.py string:", s, "| codepoints:", [ord(ch) for ch in s[:25]])

for f in ["tools/card_rating_system/audit_cards.py", "core/scenario_extractor.py", "tools/card_rating_system/RUBRIC.md"]:
    txt = Path(f).read_text(encoding="utf-8")
    for m in re.finditer(r"interloc\w*speaker\w*", txt):
        s = m.group(0)
        print(f"{f}: {s}")
        break
