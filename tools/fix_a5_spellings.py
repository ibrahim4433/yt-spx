#!/usr/bin/env python3
"""Extract exact spellings of interloc-related identifiers and write them to tools/a5_spellings.json
so code/tests can use the exact DB column name. Reads only."""
import sqlite3
from pathlib import Path

conn = sqlite3.connect("data/dawah_knowledge.db")
cols = [r[1] for r in conn.execute("PRAGMA table_info(qa_units)").fetchall()]
conn.close()
interloc_cols = [c for c in cols if "interloc" in c]
out = {
    "db_interloc_cols": interloc_cols,
    "db_interlocutor_speaker_id_col_exact": next((c for c in cols if "speaker_id" in c and "interloc" in c), None),
}
Path("tools/a5_spellings.json").write_text(
    __import__("json").dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("wrote tools/a5_spellings.json")
